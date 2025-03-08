import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv
from pathlib import Path
import os

# Print Supervision version for debugging
sv_version = sv.__version__
print(f"Using Supervision version: {sv_version}")

# Set page configuration
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)

# Define CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">Object Detection & Tracking System</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Upload a video or use your webcam to detect and track objects in real-time.</p>', unsafe_allow_html=True)

# Create sidebar for controls
with st.sidebar:
    st.markdown('<p class="sub-header">Configuration</p>', unsafe_allow_html=True)
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"],
        index=0
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # IOU threshold
    iou_threshold = st.slider(
        "IOU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    # Tracking configuration
    st.markdown('<p class="sub-header">Tracking Settings</p>', unsafe_allow_html=True)
    max_age = st.slider("Max Age (frames)", 1, 150, 30)
    n_init = st.slider("Min Hits to Confirm Track", 1, 10, 3)
    show_trails = st.checkbox("Show Motion Trails", value=True)
    trail_length = st.slider("Trail Length", 5, 100, 30) if show_trails else 0
    
    # Classes to track
    st.markdown('<p class="sub-header">Objects to Detect</p>', unsafe_allow_html=True)
    all_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    default_classes = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]
    selected_classes = st.multiselect(
        "Select Objects to Detect",
        all_classes,
        default=default_classes
    )
    
    # Get class IDs
    if selected_classes:
        selected_class_ids = [all_classes.index(cls) for cls in selected_classes]
    else:
        selected_class_ids = list(range(len(all_classes)))  # All classes if none selected

    # Visualization options
    st.markdown('<p class="sub-header">Visualization</p>', unsafe_allow_html=True)
    show_bboxes = st.checkbox("Show Bounding Boxes", value=True)
    show_labels = st.checkbox("Show Labels", value=True)
    show_conf = st.checkbox("Show Confidence", value=True)

@st.cache_resource
def load_model(model_type):
    """Load and cache the YOLO model"""
    model_path = f"{model_type.lower()}.pt"
    
    # Check if model exists locally, else download it
    if not os.path.exists(model_path):
        model = YOLO(f"{model_type.lower()}")
    else:
        model = YOLO(model_path)
    
    return model

@st.cache_resource
def create_tracker():
    """Create and return a DeepSORT tracker"""
    return DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_cosine_distance=0.4,
        nn_budget=100,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None
    )

# Initialize model and tracker
model = load_model(model_type)
tracker = create_tracker()

# Create a cross-version compatible box annotator
# We're using try/except to handle different versions of supervision
try:
    # First attempt - standard parameters
    box_annotator = sv.BoxAnnotator(thickness=2)
except TypeError as e:
    st.sidebar.warning(f"BoxAnnotator initialization error. Using alternative approach. Error: {e}")
    # Second attempt - custom implementation
    class CustomBoxAnnotator:
        def __init__(self, thickness=2):
            self.thickness = thickness
            
        def annotate(self, scene, detections, labels=None):
            annotated_frame = scene.copy()
            
            if not detections:
                return annotated_frame
                
            # Process each detection
            for i, detection in enumerate(detections):
                # Get bounding box coordinates
                if hasattr(detection, 'xyxy'):
                    # For Supervision's native DetectionWithID
                    x1, y1, x2, y2 = map(int, detection.xyxy)
                elif hasattr(detection, 'tracker_id'):
                    # Also for Supervision's DetectionWithID but different structure
                    x1, y1, x2, y2 = map(int, detection.xyxy)
                else:
                    # For our custom format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, detection)
                
                # Get label if available
                label = labels[i] if labels and i < len(labels) else None
                
                # Generate color based on detection ID if available
                if hasattr(detection, 'tracker_id'):
                    track_id = detection.tracker_id
                    # Consistent color per ID
                    color = ((track_id * 123) % 255, (track_id * 85) % 255, (track_id * 201) % 255)
                else:
                    # Default color
                    color = (0, 255, 0)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.thickness)
                
                # Draw label if provided
                if label:
                    # Position label above the bounding box
                    text_position = (x1, y1 - 10 if y1 > 20 else y1 + 20)
                    
                    # Determine text size
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    
                    # Draw background rectangle for text
                    cv2.rectangle(
                        annotated_frame,
                        (x1, text_position[1] - text_height - 5),
                        (x1 + text_width, text_position[1] + 5),
                        color,
                        -1  # Filled rectangle
                    )
                    
                    # Draw text
                    cv2.putText(
                        annotated_frame,
                        label,
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),  # White text
                        2
                    )
            
            return annotated_frame
    
    # Use our custom annotator
    box_annotator = CustomBoxAnnotator(thickness=2)

# Define a detection container class for compatibility
class DetectionContainer:
    def __init__(self, xyxy, class_id=None, tracker_id=None, confidence=None):
        self.xyxy = xyxy
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.confidence = confidence

# Function to process video frames
def process_frame(frame):
    # Make a copy of the frame for drawing
    annotated_frame = frame.copy()
    
    # Run YOLO detection
    results = model(frame, conf=confidence_threshold, iou=iou_threshold, classes=selected_class_ids)[0]
    
    # Convert detections to DeepSORT format
    detections = []
    
    for r in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, class_id = r
        
        # Skip if class is not in selected classes
        if int(class_id) not in selected_class_ids:
            continue
            
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, int(class_id)))
    
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Dictionary to store trails
    if show_trails and not hasattr(process_frame, "trails"):
        process_frame.trails = {}
    
    # List to store compatible detections for the annotator
    formatted_detections = []
    formatted_labels = []
    
    # Process each track
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        
        # Debug: Print track_id and its type
        print(f"Track ID: {track_id}, Type: {type(track_id)}")
        
        # Get class name
        class_name = all_classes[class_id]
        
        # Create annotation with ID, class and confidence
        label = f"#{track_id} {class_name}"
        
        # Check if track has det_conf attribute and it's not None
        if hasattr(track, 'det_conf') and track.det_conf is not None and show_conf:
            try:
                # Add confidence to label, handling possible errors
                label += f" {float(track.det_conf):.2f}"
            except (ValueError, TypeError):
                # Skip adding confidence if it can't be formatted as float
                pass
            
        # Create a detection container compatible with our annotator
        det = DetectionContainer(
            xyxy=ltrb,
            class_id=class_id,
            tracker_id=track_id,
            confidence=track.det_conf if hasattr(track, 'det_conf') else None
        )
        
        # Add to our lists for later annotation
        formatted_detections.append(det)
        formatted_labels.append(label)
        
        # Draw motion trails if enabled
        if show_trails:
            # Get the center of the bounding box
            bbox = ltrb
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            
            # Convert track_id to string to ensure consistent key type
            track_id_key = str(track_id)
            
            # Initialize trail list for new tracks
            if track_id_key not in process_frame.trails:
                process_frame.trails[track_id_key] = []
                
            # Add current center to trail
            process_frame.trails[track_id_key].append(center)
            
            # Limit trail length
            if len(process_frame.trails[track_id_key]) > trail_length:
                process_frame.trails[track_id_key] = process_frame.trails[track_id_key][-trail_length:]
                
            # Draw trail
            if len(process_frame.trails[track_id_key]) > 1:
                for i in range(1, len(process_frame.trails[track_id_key])):
                    # Ensure track_id is an integer for color calculation
                    try:
                        track_id_int = int(track_id)
                        color = (
                            int((track_id_int * 123) % 255),
                            int((track_id_int * 85) % 255),
                            int((track_id_int * 201) % 255)
                        )
                    except (ValueError, TypeError):
                        # Default color if track_id can't be converted to int
                        color = (0, 255, 0)
                    
                    # Calculate alpha (opacity) based on position in trail
                    alpha = (i / len(process_frame.trails[track_id_key]))
                    thickness = max(1, int(3 * alpha))
                    
                    # Draw line
                    cv2.line(
                        annotated_frame, 
                        process_frame.trails[track_id_key][i-1], 
                        process_frame.trails[track_id_key][i], 
                        color, 
                        thickness
                    )
    
    # Draw all detections at once
    if show_bboxes and formatted_detections:
        for i, det in enumerate(formatted_detections):
            x1, y1, x2, y2 = map(int, det.xyxy)
            
            # Ensure tracker_id is an integer for color calculation
            try:
                if det.tracker_id is not None:
                    tracker_id = int(det.tracker_id)
                    color = (
                        int((tracker_id * 123) % 255),
                        int((tracker_id * 85) % 255),
                        int((tracker_id * 201) % 255)
                    )
                else:
                    # Default color if tracker_id is None
                    color = (0, 255, 0)
            except (ValueError, TypeError):
                # Default color if conversion fails
                color = (0, 255, 0)
            
            # Draw bounding box
            if show_bboxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_labels:
                label = formatted_labels[i]
                # Position label above the bounding box
                text_position = (x1, y1 - 10 if y1 > 20 else y1 + 20)
                
                # Determine text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    annotated_frame,
                    (x1, text_position[1] - text_height - 5),
                    (x1 + text_width, text_position[1] + 5),
                    color,
                    -1  # Filled rectangle
                )
                
                # Draw text
                cv2.putText(
                    annotated_frame,
                    label,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # White text
                    2
                )
    
    return annotated_frame, len(formatted_detections)

# Create tabs for different input options
input_type = st.radio("Select Input Source", ["Upload Video"], horizontal=True)

if input_type == "Upload Video":
    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        
        # Create video capture object
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Display video info
        with st.expander("Video Information", expanded=False):
            cols = st.columns(4)
            with cols[0]:
                st.metric("Width", f"{width}px")
            with cols[1]:
                st.metric("Height", f"{height}px")
            with cols[2]:
                st.metric("FPS", str(fps))
            with cols[3]:
                st.metric("Duration", f"{total_frames//fps}s")
        
        # Frame selection slider
        frame_selection = st.slider("Select Frame", 0, max(0, total_frames-1), 0)
        
        # Seek to selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_selection)
        
        # Read frame
        ret, frame = cap.read()
        
        if ret:
            # Process frame
            processed_frame, num_detections = process_frame(frame)
            
            # Display metrics
            st.metric("Objects Detected", num_detections)
            
            # Display processed frame
            st.image(processed_frame, channels="BGR", caption=f"Frame {frame_selection}/{total_frames-1}")
            
            # Process video button
            if st.button("Process Full Video"):
                # Reset to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create placeholder for processed video frame
                frame_placeholder = st.empty()
                
                # Create placeholders for metrics
                metrics_cols = st.columns(3)
                frame_counter = metrics_cols[0].empty()
                detection_counter = metrics_cols[1].empty()
                fps_counter = metrics_cols[2].empty()
                
                # Process video
                frame_count = 0
                start_time = time.time()
                
                # Clear trails
                if hasattr(process_frame, "trails"):
                    process_frame.trails = {}
                
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                        
                    # Process frame
                    processed_frame, num_detections = process_frame(frame)
                    
                    # Update counters
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # Update progress
                    progress = min(1.0, frame_count / total_frames)
                    progress_bar.progress(progress)
                    
                    # Update status
                    status_text.text(f"Processing video... {frame_count}/{total_frames} frames")
                    
                    # Update metrics
                    frame_counter.metric("Frames Processed", frame_count)
                    detection_counter.metric("Objects Detected", num_detections)
                    fps_counter.metric("Processing FPS", f"{current_fps:.1f}")
                    
                    # Display frame
                    frame_placeholder.image(processed_frame, channels="BGR")
                
                # Final status
                status_text.text(f"Video processing complete. Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
                
                # Clean up
                if hasattr(process_frame, "trails"):
                    process_frame.trails = {}
        
        # Clean up
        cap.release()
        try:
            os.unlink(video_path)
        except:
            pass

# elif input_type == "Webcam":
#     # Webcam settings
#     cam_options = st.selectbox("Camera", ["Default Camera", "Camera 1", "Camera 2"])
    
#     # Map selection to camera index
#     camera_dict = {"Default Camera": 0, "Camera 1": 1, "Camera 2": 2}
#     cam_index = camera_dict[cam_options]
    
#     # Create placeholders
#     frame_placeholder = st.empty()
#     status_text = st.empty()
    
#     # Create metrics columns
#     metrics_cols = st.columns(3)
#     frame_counter = metrics_cols[0].empty()
#     detection_counter = metrics_cols[1].empty()
#     fps_counter = metrics_cols[2].empty()
    
#     # Start/Stop button
#     start_button = st.button("Start Camera")
    
#     # Add a stop button that will become visible only after starting
#     stop_placeholder = st.empty()
    
#     if start_button:
#         # Hide the start button after clicking
#         start_button = False
        
#         # Show stop button
#         stop_button = stop_placeholder.button("Stop Camera")
        
#         # Open webcam
#         cap = cv2.VideoCapture(cam_index)
        
#         # Clear trails
#         if hasattr(process_frame, "trails"):
#             process_frame.trails = {}
            
#         # Processing loop
#         frame_count = 0
#         start_time = time.time()
        
#         # Stream until stop button is pressed
#         while not stop_button:
#             # Read frame
#             ret, frame = cap.read()
            
#             if not ret:
#                 status_text.text("Error reading from webcam. Please check your camera connection.")
#                 break
                
#             # Process frame
#             processed_frame, num_detections = process_frame(frame)
            
#             # Update counters
#             frame_count += 1
#             if frame_count % 10 == 0:  # Update FPS every 10 frames
#                 elapsed_time = time.time() - start_time
#                 current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
#                 # Update metrics
#                 frame_counter.metric("Frames Processed", frame_count)
#                 detection_counter.metric("Objects Detected", num_detections)
#                 fps_counter.metric("Processing FPS", f"{current_fps:.1f}")
            
#             # Display frame
#             frame_placeholder.image(processed_frame, channels="BGR")
            
#             # Check if stop button was pressed
#             stop_button = stop_placeholder.button("Stop Camera", key='stop_button')
            
#             # Small delay to prevent high CPU usage
#             time.sleep(0.01)
        
#         # Clean up
#         cap.release()
        
#         # Clear trails
#         if hasattr(process_frame, "trails"):
#             process_frame.trails = {}
            
#         # Reset UI
#         stop_placeholder.empty()
#         status_text.text("Camera stopped.")
        

# Footer
st.markdown("---")
st.markdown('<p class="info-text">Object Detection powered by YOLOv8 | Object Tracking powered by DeepSORT</p>', unsafe_allow_html=True)

# Requirements section
with st.expander("Installation Requirements"):
    st.code("""
# Create a requirements.txt file with the following packages:
streamlit
opencv-python-headless
numpy
ultralytics
deep-sort-realtime
supervision
    """)
    
    st.markdown("""
    ### Installation Instructions
    1. Create a new virtual environment
    2. Create a requirements.txt file with the packages listed above
    3. Install the requirements: `pip install -r requirements.txt`
    4. Run the app: `streamlit run app.py`
    
    ### Troubleshooting
    - If you encounter version incompatibilities with the Supervision library, try installing an older version:
      `pip install supervision==0.8.0` or `pip install supervision==0.11.1`
    - For webcam issues on Windows, you might need to use `opencv-python` instead of `opencv-python-headless`
    """)    