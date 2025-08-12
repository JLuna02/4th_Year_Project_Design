from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import cv2
import datetime
# import sqlite3
from pyzbar.pyzbar import decode
import numpy as np 
import tensorflow as tf
import time
from collections import deque, Counter
import sqlite3
import os
import json
from flask_cors import CORS
#gst_pipeline = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), format=NV12, width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! videoconvert ! appsink'

# xcamera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
app = Flask(__name__)
socketio = SocketIO(app)
camera = cv2.VideoCapture(1)
#camera = cv2.VideoCapture("test_video/flag/MVI_5624.MOV")
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

RECORDINGS_FOLDER = 'recordings'
if not os.path.exists(RECORDINGS_FOLDER):
    os.makedirs(RECORDINGS_FOLDER)
    print("MADE DIR!!!!!!!!!!!!!!")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app)

# Function to stream video for playback
def stream_video(filename):
    video_path = os.path.join(RECORDINGS_FOLDER, filename)
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        return "Video not found", 404

    while True:
        ret, frame = video.read()
        if not ret:
            break  # Video finished or error reading

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break  # Error encoding frame

        frame_bytes = buffer.tobytes()

        # Yield frame as a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video.release()

# Add Data to the Database
def send_report(file_name, date_time):
    conn = sqlite3.connect('database/data_monitoring.db')
    curs = conn.cursor()

    # table columns: id, filename, date_time
    # Columns to be filled
    # filename, date_time
    report = """INSERT INTO session_report(filename, date_time) values {}""".format((file_name, date_time))
    curs.execute(report)
    conn.commit()
    conn.close()

def send_clip_report(clip_file_name, clip_date_time, clip_category):
    conn = sqlite3.connect('database/data_monitoring.db')
    curs = conn.cursor()

    # table columns: id, filename, date, time, category
    # Columns to be filled
    # filename, date, time, category
    report = """INSERT INTO clip_report(filename, date_time, category) values {}""".format((clip_file_name, clip_date_time, clip_category))
    curs.execute(report)
    conn.commit()
    conn.close()

# Load video file data from JSON
def load_video_data():
    conn = sqlite3.connect('database/data_monitoring.db')
    curs = conn.cursor()

    # Get the general list of session recordings
    curs.execute("SELECT * FROM session_report")
    session_data = curs.fetchall()

    # Get incidents list of clip recording
    curs.execute("SELECT * FROM clip_report")
    clip_data = curs.fetchall()

    # Prepare json for data storing
    data_json = {"general_videos": [],
             "incident_videos": {
                 "Violence": [],
                 "Panic": [],
                 "Faint": []
             }}
    for i in session_data:
        data_json["general_videos"].append({"filename": i[1], "upload_date": i[2]})
    for i in clip_data:
        match i[3]:
            case 'Brawl':
                data_json["incident_videos"]["Violence"].append({"filename": i[1], "date_time": i[2]})
            case 'Panic':
                data_json["incident_videos"]["Panic"].append({"filename": i[1], "date_time": i[2]})
            case 'Fainting':
                data_json["incident_videos"]["Faint"].append({"filename": i[1], "date_time": i[2]})
    return data_json

# Global variable for VideoWriter object and recording status
video_writer = None
is_recording = False

# Global variables for clip recording
session_writer = None
clip_writer = None
is_clip_recording = False
previous_prediction = "Normal"  # Initialize with "Normal"

# Buffer for instant replay
frame_buffer = deque(maxlen=100)  # Assuming 20 FPS, this gives ~5 seconds

# Directory to save recordings
recordings_folder = 'recordings'
if not os.path.exists(recordings_folder):
    os.makedirs(recordings_folder)

# Directory to save clips
RECORDINGS_CLIP_FOLDER = 'recordings'
if not os.path.exists(RECORDINGS_CLIP_FOLDER):
    os.makedirs(RECORDINGS_CLIP_FOLDER)

# Preprocessing function for each frame
def preprocess_frame(frame, target_size=(112, 112)):
    resized_frame = cv2.resize(frame, target_size)
    return resized_frame.astype(np.float32) / 255.0

def most_frequent_label(predictions_deque):
    if not predictions_deque:
        return "Unknown"
    label_counts = Counter(predictions_deque)
    most_common_label, count = label_counts.most_common(1)[0]
    return most_common_label


def read_qr_code(frame):
    decoded_objects = decode(frame)
    for obj in decoded_objects:
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points]))
            cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
        else:
            cv2.polylines(frame, [np.array(points)], True, (0, 255, 0), 2)
        
        qr_data = obj.data.decode('utf-8')
        # Draw the QR code data on the frame
        cv2.putText(frame, qr_data, (obj.rect.left, obj.rect.top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame, qr_data
    return frame, None

# Function to start recording the video
def start_recording():
    global video_writer, is_recording
    if not is_recording:
        # Get the current time to generate a unique video filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{recordings_folder}/session{timestamp}.mp4"
        filename = f"session{timestamp}.mp4"
        
        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))  # 20 FPS, 640x480 resolution
        is_recording = True
        send_report(filename, timestamp)
        print(f"Recording started and saved as {video_filename}")

# Function to stop recording the video
def stop_recording():
    global video_writer, is_recording
    if is_recording:
        # Release the VideoWriter and save the file
        video_writer.release()
        video_writer = None
        is_recording = False
        print("Recording stopped and saved.")
    else:
        print("No recording is currently active.")

# Function to start recording a clip
def start_clip_recording(current_prediction):
    global clip_writer, is_clip_recording, frame_buffer
    if not is_clip_recording:
        # Get the current time to generate a unique clip filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_filename = f"{RECORDINGS_CLIP_FOLDER}/clip_{timestamp}.mp4"
        clip_file_name = f"clip_{timestamp}.mp4"
        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        clip_writer = cv2.VideoWriter(clip_filename, fourcc, 20.0, (640, 480))  # 20 FPS, 640x480 resolution
        is_clip_recording = True

        send_clip_report(clip_file_name, timestamp, current_prediction)
        
        # Write the buffered frames first (last 5 seconds)
        for buffered_frame in frame_buffer:
            clip_writer.write(buffered_frame)
            
        print(f"Clip recording started and will be saved as {clip_filename}")

# Function to stop recording the clip
def stop_clip_recording():
    global clip_writer, is_clip_recording
    if is_clip_recording:
        # Release the VideoWriter and save the file
        clip_writer.release()
        clip_writer = None
        is_clip_recording = False
        print("Clip recording stopped and saved.")
    else:
        print("No clip recording is currently active.")

# Function to handle automatic clip recording based on prediction changes
def handle_auto_clip_recording(current_prediction):
    global previous_prediction

    # Define the predictions that should trigger recording
    trigger_predictions = ["Brawl", "Fainting", "Panic"]

    # Start recording if the current prediction is in the trigger list and it is different from the previous prediction
    if current_prediction in trigger_predictions and current_prediction != previous_prediction:
        start_clip_recording(current_prediction)
        print(f"Auto clip started: Detected {current_prediction}")

    # Stop recording if the current prediction is not in the trigger list or it changes
    elif previous_prediction in trigger_predictions and current_prediction not in trigger_predictions:
        stop_clip_recording()
        print(f"Auto clip stopped: Prediction changed to {current_prediction}")

    # Update the previous prediction
    previous_prediction = current_prediction

def generate_frames():
    global video_writer, is_recording, clip_writer, is_clip_recording, frame_buffer, previous_prediction
    
    frames = []
    last_capture_time = time.time()
    predict_result = int
    predictions_history = deque(maxlen=3)
    predictions_history_fainting = deque(maxlen=3)
    fainting_level = 0
    main_level = 0
    

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Store the frame in the buffer for instant replay
            frame_buffer.append(frame.copy())
            # ------------- 3D-CNN PREDICTION LOGIC -------------
            current_time = time.time()
            elapsed_time = current_time - last_capture_time

            last_capture_time = current_time

            processed_frame = preprocess_frame(frame)
            frames.append(processed_frame)

            if len(frames) == 16:
                batch_frames = np.expand_dims(np.stack(frames, axis=0), axis=0)

                prediction = model.predict(batch_frames, verbose=(-1))
                #prediction_fainting = model_fainting.predict(batch_frames, verbose=(-1))
                #predicted_class_fainting = np.argmax(prediction_fainting, axis=1)[0]
                #predict_result_fainting = predicted_class_fainting
                predicted_class = np.argmax(prediction, axis=1)[0]
                predict_result = predicted_class
                predictions_history.append(predicted_class)
                #predictions_history_fainting.append(predicted_class_fainting)

                # Confidence
                confidence = prediction[0][predicted_class]  # Get the confidence level of the predicted class
                main_level = confidence
                #confidence_fainting = prediction_fainting[0][predicted_class_fainting]  # Get the confidence level of the predicted class
                #fainting_level = confidence_fainting
                print("DETECTED CLASS!!!!8: ", predicted_class)
                print("CONFIDENCE LEVEL: ", confidence)
                # Check if the confidence is above a certain threshold (e.g., 0.5)
                
                frames = []

            # Get the most frequent label from recent predictions
            stable_prediction = most_frequent_label(predictions_history)
            #stable_prediction_fainting = most_frequent_label(predictions_history_fainting)
            label = ""
            # Map index to labels
            if stable_prediction == 0 and main_level > 0.7:
                label = "Brawl"
            elif stable_prediction == 2:
                label = "Normal"
            elif stable_prediction ==1 and main_level > 0.6:
                label = "Fainting"
            elif stable_prediction == 3:
                label = "Panic"
            else:
                label = "Normal"

            #if stable_prediction_fainting == 1 and fainting_level > 0.8:
            #    label = "Fainting"
            

            # Handle automatic clip recording based on prediction
            handle_auto_clip_recording(label)

            # ------------- DRAW PREDICTION LABEL -------------
            cv2.putText(frame, f"Prediction: {label}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            # If recording is active, write the frame to the video
            if is_recording and video_writer is not None:
                video_writer.write(frame)
                
            # If clip recording is active, write the frame to the clip
            if is_clip_recording and clip_writer is not None:
                clip_writer.write(frame)

            # Optionally emit prediction data to front-end if needed
            socketio.emit('prediction_update', {'label': label})
            #socketio.emit('time_of_prediction', {'time': current_time})

            # Encode the frame and yield it
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

cap = None
fps = 0
paused = False
current_frame = 0
seek_to_frame = None

def open_video():
    global video_replay, fps
    fps = video_replay.get(cv2.CAP_PROP_FPS)
    return video_replay

def video_stream():
    global paused, current_frame, seek_to_frame, video_replay
    video_replay = open_video()
    
    while True:
        if paused:
            time.sleep(0.1)
            continue
        
        if seek_to_frame is not None:
            video_replay.set(cv2.CAP_PROP_POS_FRAMES, seek_to_frame)
            seek_to_frame = None

        success, frame = video_replay.read()
        if not success:
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        current_frame = video_replay.get(cv2.CAP_PROP_POS_FRAMES)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(1 / fps)

def Review_frame(video_path):

    print(f"Received video path: {video_path}")  # Debugging

    # Ensure correct file path format
    video_path = os.path.normpath(video_path)  
    if not os.path.exists(video_path):
        print(f"Error: File does not exist at {video_path}")
        return "Error: Video file not found", 404
    video_path = os.path.abspath(video_path)
    
    review_cap = cv2.VideoCapture(video_path)
    if not review_cap.isOpened():
        print("Error: OpenCV cannot open video file")
        return "Error: Cannot open video file", 

    slowdown_factor = 1
    fps = review_cap.get(cv2.CAP_PROP_FPS)  # Get original FPS
    delay = (1 / fps) * slowdown_factor  # Slow-motion delay
    
    frames = []
    last_capture_time = time.time()
    predict_result = int
    

    while True:
        success, frame = review_cap.read()
        if not success:
            break
        else:
            # ------------- 3D-CNN PREDICTION LOGIC -------------
            current_time = time.time()
            elapsed_time = current_time - last_capture_time

            last_capture_time = current_time

            processed_frame = preprocess_frame(frame)
            frames.append(processed_frame)

            if len(frames) == 16:
                batch_frames = np.expand_dims(np.stack(frames, axis=0), axis=0)

                prediction = model.predict(batch_frames, verbose=(-1))
                predicted_class = np.argmax(prediction, axis=1)[0]
                predict_result = predicted_class
                frames = []

            # Get the most frequent label from recent predictions
            stable_prediction = predict_result
            label = ""
            # Map index to labels
            if stable_prediction == 0:
                label = "Brawl"
            elif stable_prediction == 1:
                label = "Fainting"
            elif stable_prediction == 2:
                label = "Normal"
            elif stable_prediction == 3:
                label = "Panic"
            elif stable_prediction == 4:
                label = "Seizure"

            # ------------- DRAW PREDICTION LABEL -------------
            cv2.putText(frame, f"Prediction: {label}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode the frame and yield it
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Slow-motion effect by sending the same frame multiple times
            for _ in range(slowdown_factor):
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(delay)  # Introduce delay
    review_cap.release()
    cv2.destroyAllWindows()

def release_video():
    global video_replay
    video_replay.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('testing.html')

#@app.route('/home_page')
#def show_main_page():
    #return render_template('testing.html')

# Route for sending report

# Route for logging in, Checking for credentials
@app.route('/check_login', methods=['POST'])
def check_login():
    data = request.json
    username = data.get('userName')
    user_pass = data.get('userPass')
    print("Username: ", username)
    print("Password: ", user_pass)

    conn = sqlite3.connect('database/data_monitoring.db')
    curs = conn.cursor()
    curs.execute("SELECT * FROM admin_user WHERE userName = '{}'".format(username))
    credentials = curs.fetchall()
    conn.close()
    print(credentials)
    if(credentials[0][1] == username and credentials[0][2] == user_pass):
        print(True)
        return jsonify({'verify_login': True})
    else:
        print(False)
        return False

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stop_recording', methods=['POST'])
def stop_record():
    stop_recording()
    return jsonify({"message": "Session recording stopped!"})

@app.route('/api/start_recording', methods=['POST'])
def start_record():
    start_recording()
    return jsonify({"message": "Session recording started!"})

@app.route('/api/start_clip_recording', methods=['POST'])
def start_clip():
    start_clip_recording()
    return jsonify({"message": "Clip recording started!"})

@app.route('/api/stop_clip_recording', methods=['POST'])
def stop_clip():
    stop_clip_recording()
    return jsonify({"message": "Clip recording stopped!"})

@app.route('/api/qr_data', methods=['GET'])
def get_qr_data():
    return jsonify({
        'timestamp': datetime.datetime.now().isoformat(),
        'last_qr_data': 'QR Code Data Here',
        'scan_count': 1
    })

@app.route('/api/camera_status', methods=['GET'])
def camera_status():
    return jsonify({
        'status': 'active',
        'timestamp': datetime.datetime.now().isoformat(),
        'camera_id': 'main_feed'
    })

@app.route('/api/camera_settings', methods=['GET', 'POST'])
def camera_settings():
    if request.method == 'GET':
        return jsonify({
            'resolution': '640x480',
            'brightness': 50,
            'contrast': 50
        })
    elif request.method == 'POST':
        return jsonify({'message': 'Settings updated'})

@app.route('/api/snapshot', methods=['GET'])
def take_snapshot():
    return jsonify({
        'snapshot_id': '12345',
        'timestamp': datetime.datetime.now().isoformat(),
        'url': '/snapshots/latest.jpg'
    })

# API Route to Get General Video Files
@app.route('/get_videos', methods=['GET'])
def get_videos():
    video_data = load_video_data()
    return jsonify(video_data['general_videos'])

# API Route to Get Video Files for a specific incident type
@app.route('/get_videos/<incident_type>', methods=['GET'])
def get_videos_by_incident(incident_type):
    video_data = load_video_data()
    # Return the videos for the requested incident type
    # Check if the incident type exists in the data
    incident_videos = video_data['incident_videos']
    if incident_type in incident_videos:
        return jsonify(incident_videos[incident_type])
    else:
        return jsonify([])  # Return an empty list if the incident type is not found

# API Route to Get Total Counts for Each Incident Type
@app.route('/get_total_counts', methods=['GET'])
def get_total_counts():
    video_data = load_video_data()
    counts = {
        "Violence": len(video_data['incident_videos'].get("Violence", [])),
        "Panic": len(video_data['incident_videos'].get("Panic", [])),
        "Faint": len(video_data['incident_videos'].get("Faint", []))
    }
    return jsonify(counts)

# This is for viewing video on a video player
# This route will handle the dynamic video loading
@app.route('/view_video/<filename>')
def view_video(filename):
    global video_replay
    filename = "recordings/" + filename
    print("Streaming: ", filename)
    video_replay = cv2.VideoCapture(filename)
    return Response(video_stream(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)

    # Debugging: Print file save path
    print(f"Saving uploaded file to: {video_path}")

    video.save(video_path)

    return jsonify({"video_path": video_path})

# Stream annotated video
@app.route('/video_feed_test')
def video_feed_test():
    print("Upload Clicked!")
    video_path = request.args.get('video_path')
    if not video_path:
        return "No video path provided", 400
    
    # Normalize path for cross-platform compatibility
    video_path = os.path.normpath(video_path)
    print(video_path)

    return Response(Review_frame(video_path), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pause', methods=['POST'])
def pause():
    global paused
    paused = True
    return 'Paused', 200

@app.route('/resume', methods=['POST'])
def resume():
    global paused
    paused = False
    return 'Resumed', 200

@app.route('/seek', methods=['POST'])
def seek():
    global seek_to_frame
    time_in_seconds = float(request.form['time'])
    seek_to_frame = int(time_in_seconds * fps)  # Convert time to frame number
    return 'Seeking', 200

@app.route('/api/check_trigger', methods=['GET'])
def check_trigger():
    global dialog_triggered
    # Return the status of whether the dialog should be triggered
    return jsonify({'trigger': dialog_triggered})

@app.route('/delete_video/<filename>', methods=['DELETE'])
def delete_video(filename):
    # Delete the video file
    video_path = os.path.join(RECORDINGS_FOLDER, filename)
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Deleted file: {video_path}")
    else:
        print(f"File not found: {video_path}")

    # Delete the video entry from the database
    conn = sqlite3.connect('database/data_monitoring.db')
    curs = conn.cursor()

    # Try deleting from both session_report and clip_report tables
    try:
        curs.execute("DELETE FROM session_report WHERE filename = ?", (filename,))
        curs.execute("DELETE FROM clip_report WHERE filename = ?", (filename,))
        conn.commit()
        print(f"Deleted database entry for: {filename}")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({"success": False, "message": "Failed to delete video from database."}), 500
    finally:
        conn.close()

    return jsonify({"success": True, "message": f"Video '{filename}' deleted successfully."})

@app.route('/close_video/<filename>', methods=['POST'])
def close_video():
    print("Close video clicked!")
    global video_replay
    if video_replay is not None:
        video_replay.release()
        cv2.destroyAllWindows()
        print("Video closed successfully.")
    else:
        print("No video to close.")
    return jsonify({"message": "Video closed successfully."})

if __name__ == "__main__":
    model_path = 'saved_model/modified_brawl_detection4.keras'  # For Brawl Model
    #model_path = 'saved_model/second_chance7.keras' # For review model
    #model_path = 'saved_model/modified_2_class4.keras'  # For Fainting Model
    model = tf.keras.models.load_model(model_path)
    #model_fainting_path = 'saved_model/modified_2_class4.keras'
    #model = tf.keras.models.load_model(model_fainting_path)
    print("Model loaded successfully!")
    print(model.input_shape)

    video_path = 0
    #predict_live_video(model, video_path, frame_count=20, capture_interval=0.2, smoothing_window=5)

    socketio.run(app, host="0.0.0.0")
