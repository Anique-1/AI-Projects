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
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
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
    st.title("📸 Image Age Detection")
    
    # Load age detection model
    try:
        age_net = load_age_detection_model()
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Detect ages
        result_image = detect_age_and_faces(img_array, age_net)
        
        # Display original and result images
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Original Image")
            st.image(image)
        
        with col2:
            st.header("Age Detection Result")
            st.image(result_image)

if __name__ == "__main__":
    main()