import streamlit as st
import cv2
import numpy as np
from PIL import Image

def load_age_detection_model():
    """Load age detection neural network from local files"""
    return cv2.dnn.readNetFromCaffe(
        'AI-human-face-detection\\deploy.prototxt', 
        'AI-human-face-detection\\age_net.caffemodel'
    )

def detect_age_and_faces(image, age_net):
    """Detect faces and estimate ages"""
    # Age categories
    age_list = ['(0-3)', '(4-7)', '(8-13)', '(14-20)','(21-24)', 
                '(25-32)','(33-37)', '(38-43)','(44-47)', '(48-53)','(53-59)', '(60-100)']
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30)
    )
    
    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        
        # Prepare face for age detection
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), 
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )
        
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        
        # Draw rectangle and age
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, age, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0, 255, 0), 2)
    
    return image

def main():
    st.title("📸 Age Detection System")
    
    # Load age detection model
    try:
        age_net = load_age_detection_model()
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return

    # Add mode selection
    mode = st.radio("Select Mode", ["Real-time Detection", "Image Upload"])

    if mode == "Real-time Detection":
        # Initialize video capture
        if 'cap' not in st.session_state:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create placeholder for video frame
        frame_placeholder = st.empty()
        
        # Add a stop button
        stop_button = st.button("Stop")

        if not stop_button:
            while True:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    st.session_state.cap.release()
                    break

                # Convert frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame for age detection
                processed_frame = detect_age_and_faces(frame, age_net)
                
                # Display the processed frame
                frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
        else:
            # Release the webcam
            st.session_state.cap.release()
            # Clear the frame
            frame_placeholder.empty()
            st.experimental_rerun()

    else:  # Image Upload mode
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert image to RGB if it's not
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Process image
            processed_image = detect_age_and_faces(img_array, age_net)
            
            # Display original and processed images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Processed Image")
                st.image(processed_image, use_column_width=True)

if __name__ == "__main__":
    main()