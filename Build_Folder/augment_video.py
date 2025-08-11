import cv2
import os

def rotate_video(input_file, output_dir):
    # Load the video file using OpenCV
    cap = cv2.VideoCapture(input_file)
    
    # Get the original video's properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get the base name of the file to create the output file names
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare the video writers for rotated videos
    #rotated_right_writer = cv2.VideoWriter(os.path.join(output_dir, f"{name}_rotated_right{ext}"), fourcc, fps, (frame_height, frame_width))
    #rotated_top_writer = cv2.VideoWriter(os.path.join(output_dir, f"{name}_rotated_top{ext}"), fourcc, fps, (frame_width, frame_height))
    rotated_left_writer = cv2.VideoWriter(os.path.join(output_dir, f"{name}_rotated_left{ext}"), fourcc, fps, (frame_height, frame_width))
    #
    #flipped_horiz_writer = cv2.VideoWriter(os.path.join(output_dir, f"{name}_flipped_horiz{ext}"), fourcc, fps, (frame_width, frame_height))

    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate frames and write to respective video files
        #rotated_right = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #rotated_top = cv2.rotate(frame, cv2.ROTATE_180)
        rotated_left = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #flipped_horiz = cv2.flip(frame, 1)  # Horizontal flip


        #rotated_right_writer.write(rotated_right)
        #rotated_top_writer.write(rotated_top)
        rotated_left_writer.write(rotated_left)
        #flipped_horiz_writer.write(flipped_horiz)

    # Release video resources
    cap.release()
    #rotated_right_writer.release()
    #rotated_top_writer.release()
    rotated_left_writer.release()
    #flipped_horiz_writer.release()


    print(f"Rotation completed for: {base_name}")

def process_videos_in_folder(input_folder, output_folder):
    # List all video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    # Process each video file in the folder
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        rotate_video(video_path, output_folder)

if __name__ == "__main__":
    # Path to the folder containing your videos
    input_folder = "no_no_Edit"  # Change this to your folder path with videos
    output_folder = "no_no_Augment"  # Folder where rotated videos will be saved

    # Process the videos in the folder
    process_videos_in_folder(input_folder, output_folder)
