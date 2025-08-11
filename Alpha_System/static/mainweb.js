const socket = io();
let previousPrediction = "Normal"; // Initialize with "Normal"

socket.on('prediction_update', function(data) {
    // Refresh the total counts
    const prediction = data.label.trim(); // Get the current prediction

    // Only trigger a notification if the prediction has changed
    if (prediction !== previousPrediction) {
        if (prediction !== "Normal") {
            handlePredictionChange(prediction); // Trigger the notification
            logIncident(prediction)
        }
        previousPrediction = prediction; // Update the previous prediction
    }
});

// Function to close the incident modal
function deleteClip(filename, incidentType) {
    if (confirm(`Are you sure you want to delete the clip "${filename}"?`)) {
        fetch(`/delete_video/${filename}`, { method: 'DELETE' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(data.message);

                    // Refresh the modal to reflect the deletion
                    openIncidentModal(incidentType);
                } else {
                    alert("Failed to delete the clip.");
                }
            })
            .catch(error => console.error("Error deleting clip:", error));
    }
}

// Function to show the selected section
function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.style.display = 'none';
    });
        
    document.getElementById(sectionId).style.display = 'block';
    
    // Show search bar only in the Video File section
    let searchBar = document.getElementById("videoSearchBar");
    searchBar.style.display = (sectionId === "video") ? "block" : "none";
    
    // Load video files when entering the Video File section
    if (sectionId === 'video') {
        loadVideoFiles();
    }
}
function editRow(button) {
    const row = button.closest("tr");
    const agencyCell = row.cells[0];
    const hotlineCell = row.cells[1];

    if (button.textContent === "Edit") {
        // Make cells editable
        const agency = agencyCell.textContent;
        const hotline = hotlineCell.textContent;

        agencyCell.innerHTML = `<input type="text" value="${agency}">`;
        hotlineCell.innerHTML = `<input type="text" value="${hotline}">`;

        button.textContent = "Save";
    } else {
        // Save the new values
        const agencyInput = agencyCell.querySelector("input").value;
        const hotlineInput = hotlineCell.querySelector("input").value;

        agencyCell.textContent = agencyInput;
        hotlineCell.textContent = hotlineInput;

        button.textContent = "Edit";
    }
}

function addEmergencyRow() {
    const tableBody = document.querySelector("#emergencyTable tbody");
    const newRow = document.createElement("tr");

    // New row is not editable by default
    newRow.innerHTML = `
        <td>New Agency</td>
        <td>000-0000</td>
        <td>
            <button onclick="editRow(this)">Edit</button>
            <button onclick="deleteRow(this)">Delete</button>
        </td>
    `;

    tableBody.appendChild(newRow);
}
function saveNewRow(button) {
    const row = button.closest("tr");
    const agencyInput = row.cells[0].querySelector("input").value;
    const hotlineInput = row.cells[1].querySelector("input").value;

    if (!agencyInput || !hotlineInput) {
        alert("Please fill in both fields.");
        return;
    }

    row.cells[0].textContent = agencyInput;
    row.cells[1].textContent = hotlineInput;
    row.cells[2].innerHTML = `<button onclick="editRow(this)">Edit</button>`;
}
function deleteRow(button) {
    const row = button.closest("tr");
    if (confirm("Are you sure you want to delete this row?")) {
        row.remove();
    }
}
function addEmergencyRow() {
    const tableBody = document.querySelector("#emergencyTable tbody");
    const newRow = document.createElement("tr");

    newRow.innerHTML = `
        <td contenteditable="true">New Agency</td>
        <td contenteditable="true">000-0000</td>
        <td>
            <button onclick="editRow(this)">Edit</button>
            <button onclick="deleteRow(this)">Delete</button>
        </td>
    `;

    tableBody.appendChild(newRow);
}

// Load video files from the database
function loadVideoFiles() {
    fetch('get_videos') 
    .then(response => {
        if (!response.ok) {
            throw new Error("Failed to fetch videos");
        }
        return response.json();
    })
    .then(data => {
        let tableBody = document.querySelector("#videoTable tbody");
        tableBody.innerHTML = ""; // Clear existing data

        data.forEach(video => {
            let row = document.createElement("tr");
            row.innerHTML = `
                <td>${video.filename}</td>
                <td>${video.upload_date}</td>
                <td>
                    <button onclick="playVideo('${video.filename}')">Play</button>
                    <button onclick="deleteVideo('${video.filename}')">Delete</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    })
    .catch(error => console.error("Error loading videos:", error));
}

// Function to open the incident modal and load video data based on the incident type
function openIncidentModal(incidentType) {
    console.log(`/get_videos/${incidentType}`);
    fetch(`/get_videos/${incidentType}`)
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to load data');
            }
        })
        .then(data => {
            // Show the modal with the appropriate data
            const modal = document.getElementById('incidentModal');
            const tableBody = document.getElementById('incidentTableBody');

            // Clear the table
            tableBody.innerHTML = '';

            // Populate the table with data
            data.forEach(video => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${video.filename}</td>
                    <td>${video.date_time}</td>
                    <td>
                        <button onclick="playVideo('${video.filename}')">Play</button>
                        <button onclick="deleteClip('${video.filename}', '${incidentType}')">Delete</button>
                    </td>
                `;
                tableBody.appendChild(row);
            });

            // Show the modal
            modal.style.display = 'block';
        })
        .catch(error => {
            console.error('Error loading videos:', error);
        });
}

// Function to close the incident modal
function closeIncidentModal() {
    const modal = document.getElementById('incidentModal');
    modal.style.display = 'none';
}

// Toggle navigation visibility on clicking the toggle bar
let togglebar = document.querySelector('.togglebar');
let navigation = document.querySelector('.navigation');
let main = document.querySelector('.main');

togglebar.onclick = function(){
    navigation.classList.toggle('active');
    main.classList.toggle('active');
}

// Highlight the active navigation item
let list = document.querySelectorAll('.navigation ul li');
let activeSection = document.querySelector('.navigation ul li a[href="#HOME"]')?.parentElement;

function activeLink() {
    list.forEach((item) => item.classList.remove('hovered'));
    this.classList.add('hovered');
}

// On hover, highlight the hovered item
list.forEach((item) => {
    item.addEventListener('mouseover', activeLink);
    item.addEventListener('mouseout', function () {
        list.forEach((el) => el.classList.remove('hovered'));
        if (activeSection) activeSection.classList.add('hovered'); 
    });

    // Set the active section when clicked
    item.addEventListener('click', function () {
        list.forEach((el) => el.classList.remove('hovered'));
        this.classList.add('hovered');
        activeSection = this; 
    });
});

// Load home section by default when the page loads
window.onload = function() {
    showSection('dashboard');
};
// Function to fetch total counts for incidents (Violence, Panic, Faint)
function loadTotalCounts() {
    fetch('get_total_counts', { method: 'GET' })  // API to get total counts
    .then(response => response.json())
    .then(data => {
        //console.log(data);
        //console.log();

        // Get the table body
        //let tableBody = document.querySelector("#totalCountsTable tbody");
        //tableBody.innerHTML = ""; // Clear existing data
        
        // Loop through the counts and populate the table
        for (let incident in data) {
            let countBody = document.getElementById(`count${incident}`);
            countBody.innerHTML = data[incident]; // Clear existing data
            //let row = document.createElement("tr");
            //row.innerHTML = `
            //    <td>${incident}</td>
            //    <td>${data[incident]}</td>
            //`;
            //tableBody.appendChild(row);
        }
    })
    .catch(error => console.error("Error loading total counts:", error));
}

// Call the loadTotalCounts function when the page loads or whenever needed
// document.querySelector("#loadTotalCountsButton").addEventListener("click", loadTotalCounts);


// Call the loadTotalCounts function when the page loads or whenever needed
// document.querySelector("#loadTotalCountsButton").addEventListener("click", loadTotalCounts);

// Function to fetch video files based on the selected incident type (Violence, Panic, Faint)
function loadVideosByIncident(incidentType) {
    fetch(`get_videos/${incidentType}`)
    .then(response => response.json())
    .then(data => {
        console.log(data);
        let tableBody = document.querySelector("#incidentVideoTable tbody");
        tableBody.innerHTML = ""; // Clear existing data

        if (data.length === 0) {
            tableBody.innerHTML = "<tr><td colspan='3'>No videos available for this incident type.</td></tr>";
        }

        // Loop through the video data and populate the table
        data.forEach(video => {
            console.log(video.filename);
            let row = document.createElement("tr");
            row.innerHTML = `
                <td>${video.filename}</td>
                <td>${video.dateTime}</td>
                <td>
                    <button onclick="playVideo('${video.filename}')">Play</button>
                    <button onclick="deleteVideo('${video.filename}')">Delete</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    })
    .catch(error => console.error("Error loading videos:", error));
}

function playVideo(filename) {
    console.log(`Playing video: ${filename}`);

    const videoPlayer = document.getElementById('videoPlayer');
    var videoSource = document.getElementById('videoReplay');

    // Dynamically set the video path
    console.log("Video source URL: /view_video/${filename}");
    videoSource.src =`view_video/${filename}`;

    //videoPlayer.src = `/static/${filename}`;
    //videoPlayer.play(); // Start playing the video

    // Show the video modal
    const videoModal = document.getElementById('videoModal');
    videoModal.style.display = 'block'; // Open the modal to show the video
}

// Pause the video
function pauseVideo() {
    fetch('/pause', { method: 'POST' })
        .then(response => response.text())
        .then(data => console.log(data))
        .catch(error => console.error('Error pausing video:', error));
}

// Resume the video
function resumeVideo() {
    fetch('/resume', { method: 'POST' })
        .then(response => response.text())
        .then(data => console.log(data))
        .catch(error => console.error('Error resuming video:', error));
}

// Seek the video to the provided time (in seconds)
function seekVideo() {
    const seekTime = document.getElementById('seekTime').value;
    if (seekTime) {
        fetch('/seek', {
            method: 'POST',
            body: new FormData().append('time', seekTime)
        })
        .then(response => response.text())
        .then(data => console.log(data));
    } else {
        alert('Please enter a valid seek time');
    }
}

// Function to close the video modal
function closeVideoModal() {
    const videoModal = document.getElementById('videoModal');
    var stop_viewing = `/stop_view_Video`;
    videoModal.style.display = 'none';
    fetch(`/close_video/${filename}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Remove the video from the table
                    const row = document.querySelector(`#videoTable tbody tr:has(td:contains("${filename}"))`);
                    row.remove();
                } else {
                    alert("Failed to delete video.");
                }
            })
            .catch(error => {
                console.error('Error deleting video:', error);
                alert('Error deleting video.');
            });
}

// Function to close the incident modal (the one for total counts)
function closeIncidentModal() {
    const modal = document.getElementById('incidentModal');
    modal.style.display = 'none';
}

function start_record_route() {
    fetch('/api/start_recording', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);  // Show the response message
    })
    .catch(error => console.error('Error:', error));
}

function triggerRoute() {
    fetch('/api/stop_recording', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);  // Show the response message
    })
    .catch(error => console.error('Error:', error));
}


function showNotification(message, type = "alert") {
    {const container = document.getElementById("notification-container");
    const notification = document.createElement("div");

    notification.innerText = message;
    notification.style.padding = "15px 20px";
    notification.style.borderRadius = "8px";
    notification.style.boxShadow = "0 4px 8px rgba(0,0,0,0.1)";
    notification.style.color = "#fff";
    notification.style.fontWeight = "bold";
    notification.style.fontSize = "14px";
    notification.style.animation = "fadeInOut 5s ease forwards";

    // Custom colors for alert types
    if (type === "danger") {
        notification.style.backgroundColor = "#e74c3c"; // red
    } else if (type === "warning") {
        notification.style.backgroundColor = "#f39c12"; // orange
    } else {
        notification.style.backgroundColor = "#3498db"; // blue
    }

    container.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);}
}

// Example usage when prediction changes
function handlePredictionChange(prediction) {
    loadTotalCounts(); // Refresh the total counts
    if (prediction === "Brawl") {
        showNotification("⚠️ Violence detected!", "danger");
    } else if (prediction === "Panic") {
        showNotification("⚠️ Panic detected!", "warning");
    } else if (prediction === "Fainting") {
        showNotification("⚠️ Someone fainted!", "warning");
    }
}


// Function to add video details to the table
function addVideoToTable(filename, uploadDate) {
    const tableBody = document.querySelector('#videoTable tbody');

    // Create a new row with video details
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${filename}</td>
        <td>${uploadDate}</td>
        <td>
            <button onclick="playVideo('${filename}')">Play</button>
            <button onclick="deleteVideo('${filename}')">Delete</button>
        </td>
    `;

    // Append the row to the table body
    tableBody.appendChild(row);
}

/*  Function to add uploaded video details to the table
function addUploadTable(filename, uploadDate) {
    const tableBody = document.querySelector('#videoTable2 tbody');

    // Create a new row with video details
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${filename}</td>
        <td>${uploadDate}</td>
        <td>
            <button onclick="play_ReviewVideo('${filename}')">Play</button>
        </td>
    `;

    // Append the row to the table body
    tableBody.appendChild(row);
}
*/

// Function to play video
/*function playVideo(filename) {
    const videoPlayer = document.createElement('videoReplay');
    videoPlayer.controls = true;
    videoPlayer.src = `/view_video/${filename}`; // Path to video file on the server

    // Show the video player in a modal or new window
    const modal = document.createElement('div');
    modal.classList.add('modal');
    modal.appendChild(videoPlayer);
    document.body.appendChild(modal);

    // Close modal on click
    modal.addEventListener('click', function() {
        document.body.removeChild(modal);
    });

    videoPlayer.play();
}
*/
// Function to delete video from the table (and possibly the server)
function deleteVideo(filename) {
    if (confirm(`Are you sure you want to delete ${filename}?`)) {
        // Send a request to the server to delete the video (optional)
        fetch(`/delete_video/${filename}`, { method: 'DELETE' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Remove the video from the table
                    const row = document.querySelector(`#videoTable tbody tr:has(td:contains("${filename}"))`);
                    row.remove();
                } else {
                    alert("Failed to delete video.");
                }
            })
            .catch(error => {
                console.error('Error deleting video:', error);
                alert('Error deleting video.');
            });
    }
}

function logIncident(incidentType) {
    // Get current date and time
    const now = new Date();
    const timestamp = now.toLocaleString(); // Format: "MM/DD/YYYY, hh:mm:ss AM/PM"
    
    // Create a new table row with the incident data
    const row = document.createElement("tr");
    row.innerHTML = `
        <td>${timestamp}</td>
        <td>${incidentType}</td>
        <td>${incidentType} detected</td> <!-- Change this based on your status -->
    `;
    
    // Append the new row to the table body
    const tableBody = document.getElementById("incidentHistoryBody");
    tableBody.appendChild(row);

    // Ensure only 2 or 3 rows are visible
    const maxRows = 3; // Change this to 2 if you want only 2 rows
    while (tableBody.rows.length > maxRows) {
        tableBody.deleteRow(0); // Remove the oldest row (first row)
    }
    
}

function ShowReview(){
    console.log("triggered");
        const fileInput = document.getElementById('videoUpload');
        const file = fileInput.files[0];
        const currentDate = new Date();
        const formattedDate = currentDate.toLocaleString();
        console.log(file.name)
        //addUploadTable(file.name, formattedDate)
        if (!file) {
            alert("Please select a video file first.");
            return;
        }
        let formData = new FormData();
        formData.append('video', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.video_path) {
                let videoPath = data.video_path.replace(/\\/g, '/'); // Convert Windows path
                var videoReview = document.getElementById('videoPlayerreview');
                videoReview.src = `/video_feed_test?video_path=${videoPath}`;
                videoReview.style.display = 'block';
            }
        })
        .catch(error => console.error('Error:', error));
}

function play_ReviewVideo(review_Filename){
    var videoReview = document.getElementById('videoPlayerreview');
    videoReview.src = `/video_feed_test?video_path=${review_Filename}`;
    videoReview.style.display = 'block';
}

function checkTrigger() {
    fetch('/api/check_trigger')
        .then(response => response.json())
        .then(data => {
            // If the 'trigger' key is true, show a dialog
            if (data.trigger) {
                alert('The dialog is triggered!');
                // Reset the trigger after showing the dialog (optional)
                resetDialog();
            }
        })
        .catch(error => console.error('Error:', error));
}

// Call the function with a specific incident type when needed
//document.querySelector("#violenceButton").addEventListener("click", () => loadVideosByIncident("Violence"));
//document.querySelector("#panicButton").addEventListener("click", () => loadVideosByIncident("Panic"));
//document.querySelector("#faintButton").addEventListener("click", () => loadVideosByIncident("Faint"));

window.addEventListener('DOMContentLoaded', () => {
    // Load immediately on page load
    //loadTotalCounts();

});