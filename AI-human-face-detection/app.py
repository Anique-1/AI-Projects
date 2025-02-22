import streamlit as st
import cv2
import numpy as np

def load_age_detection_model():
    """Load pre-trained age detection model"""
    age_net = cv2.dnn.readNetFromCaffe(
        'AI-human-face-detection\\deploy.prototxt', 
        'AI-human-face-detection\\age_net.caffemodel'
    )
    return age_net

def detect_age(image, age_net):
    """Detect age in the given image"""
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (227, 227), 
        (78.4263377603, 87.7689143744, 114.895847746), 
        swapRB=False
    )
    
    age_net.setInput(blob)
    age_preds = age_net.forward()
    
    return age_list[age_preds[0].argmax()]

def detect_faces_and_ages(image, age_net):
    """Detect faces and their ages in the image"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        age = detect_age(face_img, age_net)
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, age, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0, 255, 0), 2)
    
    return image

def main():
    st.title("🧓 Live Age Detection")
    
    # Load age detection model once
    age_net = load_age_detection_model()
    
    # Camera input
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    
    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect faces and ages
        processed_frame = detect_faces_and_ages(frame, age_net)
        
        # Convert BGR to RGB for Streamlit
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Display processed frame
        FRAME_WINDOW.image(processed_frame_rgb)
    
    else:
        st.write('Stopped')
    
    # Release camera when done
    camera.release()

if __name__ == "__main__":
    main()