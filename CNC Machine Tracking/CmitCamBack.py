import cv2
import os
import csv
import streamlit as st
from datetime import datetime, timedelta
from ultralytics import YOLO
import base64

# Function to encode the background image in base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set the background image ONLY on the sides
def set_background_image(image_path):
    base64_image = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/jpeg;base64,{base64_image}) no-repeat center fixed;
            background-size: cover;
        }}

        /* Keep the middle section fully white, like the original */
        .block-container {{
            background-color: white;
            width: 80%;
            margin: auto;
            padding: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set up result folder
result_folder = os.path.expanduser("~/Downloads/ML_Results")
os.makedirs(result_folder, exist_ok=True)
csv_file = os.path.join(result_folder, "tracking_results.csv")

# Load YOLO Model
yolo_model = YOLO("yolov8n.pt")

# Create CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Start Time", "End Time", "Duration (s)", "Event", "Stillstand Type", "Image Path"])

def save_event_to_csv(event_type, stillstand_type, start_time, duration, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"{event_type}_{timestamp}.jpg"
    image_path = os.path.join(result_folder, image_filename)
    cv2.imwrite(image_path, frame)

    if start_time:
        end_time = start_time + timedelta(seconds=duration)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([start_time.strftime("%H:%M:%S"), end_time.strftime("%H:%M:%S"), duration, event_type, stillstand_type, image_path])

def detect_tool_with_yolo(frame):
    results = yolo_model(frame)
    detected_boxes = []

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if conf > 0.5:
                detected_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return detected_boxes[0] if detected_boxes else None

def is_valid_bbox(bbox, frame_width, frame_height):
    """Check if the bounding box is valid and within frame boundaries"""
    if not bbox:
        return False
    
    x, y, w, h = bbox
    if (x < 0 or y < 0 or x + w > frame_width or y + h > frame_height or
        w <= 0 or h <= 0):
        return False
    
    return True

def track_tool_and_detect_events(video_source, skip_frames=5):
    global start_time, tool_out_of_frame
    start_time = datetime.now()
    tool_out_of_frame = False

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Fehler: Video oder Kamera konnte nicht geöffnet werden.")
        return

    ret, first_frame = cap.read()
    if not ret:
        st.error("Fehler: Das erste Frame konnte nicht gelesen werden.")
        return

    # Get frame dimensions
    frame_height, frame_width = first_frame.shape[:2]
    
    # Try to detect tool in the first frame
    initial_bbox = detect_tool_with_yolo(first_frame)
    
    # If no tool detected, use center of frame
    if not initial_bbox or not is_valid_bbox(initial_bbox, frame_width, frame_height):
        # Use a reasonable default ROI in the center of the frame
        center_x, center_y = frame_width // 2, frame_height // 2
        roi_size = min(frame_width, frame_height) // 4
        initial_bbox = (
            max(0, center_x - roi_size // 2),
            max(0, center_y - roi_size // 2),
            min(roi_size, frame_width - 1),
            min(roi_size, frame_height - 1)
        )
        st.warning("Kein Werkzeug erkannt. Starte Tracking im Bildzentrum.")

    # Initialize CSRT tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(first_frame, initial_bbox)
    
    frame_counter = 0
    last_positions = []
    still_frame_count = 0  

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % skip_frames != 0:
            continue

        orig_frame = frame.copy()
        frame = cv2.resize(frame, (640, 360))
        frame_height, frame_width = frame.shape[:2]
        
        success, werkzeug_bbox = tracker.update(frame)

        if success and is_valid_bbox(werkzeug_bbox, frame_width, frame_height):
            x, y, w, h = [int(v) for v in werkzeug_bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if tool_out_of_frame:
                tool_out_of_frame = False
                duration = (datetime.now() - start_time).total_seconds()
                save_event_to_csv("Bewegung", "-", start_time, duration, frame)

            movement_threshold = 5  
            last_positions.append((x, y))
            if len(last_positions) > 10:
                last_positions.pop(0)

            # Only calculate movement if we have enough positions
            if len(last_positions) > 1:
                movement = max(abs(last_positions[-1][0] - last_positions[0][0]), 
                              abs(last_positions[-1][1] - last_positions[0][1]))

                if movement < movement_threshold:
                    still_frame_count += 1
                else:
                    still_frame_count = 0  

                if still_frame_count >= 10 and not tool_out_of_frame:
                    duration = (datetime.now() - start_time).total_seconds()
                    save_event_to_csv("Stillstand", "Still", start_time, duration, frame)
                    tool_out_of_frame = True
                    start_time = datetime.now()
        else:
            # Try to detect the tool with YOLO
            detected_bbox = detect_tool_with_yolo(frame)

            if detected_bbox and is_valid_bbox(detected_bbox, frame_width, frame_height):
                x, y, w, h = detected_bbox
                
                # Make sure bbox is within frame boundaries
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                w = max(1, min(w, frame_width - x))
                h = max(1, min(h, frame_height - y))
                
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                tool_out_of_frame = False
                
                # Draw the detected bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                continue

            if not tool_out_of_frame:
                tool_out_of_frame = True
                duration = (datetime.now() - start_time).total_seconds()
                save_event_to_csv("Stillstand", "Werkzeugwechseln", start_time, duration, frame)
                # Draw status text
                cv2.putText(frame, "Tool not visible", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR", caption="Werkzeugbewegung verfolgen")

        # Add a small delay to reduce CPU usage
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_available_cameras():
    index = 0
    available_cameras = []
    
    # Try up to 2 cameras (to avoid long detection times)
    while index < 2:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        camera_name = "Laptop-Kamera" if index == 0 else "Externe Kamera"
        available_cameras.append(camera_name)
        cap.release()
        index += 1

    return available_cameras if available_cameras else ["Keine Kamera gefunden"]

# Main Streamlit App
st.title("🛑 Stillstandserkennung")

option = st.radio("Wählen Sie eine Option", ["📂 Video hochladen", "📹 Live-Kameraerkennung"])

if option == "📂 Video hochladen":
    uploaded_file = st.file_uploader("Laden Sie eine Videodatei hoch", type=["mp4", "avi", "mov"])
    if uploaded_file:
        try:
            video_path = os.path.join(result_folder, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success("Datei erfolgreich hochgeladen!")

            # Show a start button
            if st.button("Tracking starten"):
                with st.spinner("Tracking läuft..."):
                    track_tool_and_detect_events(video_path)
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten der Datei: {str(e)}")

elif option == "📹 Live-Kameraerkennung":
    cameras = get_available_cameras()
    selected_camera = st.selectbox("Wählen Sie eine Kamera", options=cameras)

    if selected_camera != "Keine Kamera gefunden" and st.button("Kamera starten"):
        camera_index = 0 if selected_camera == "Laptop-Kamera" else 1
        with st.spinner("Kamera wird initialisiert..."):
            track_tool_and_detect_events(camera_index)