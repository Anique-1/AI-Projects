import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tempfile
import time


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def extract_pose_features(landmarks):
    
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    
    
    torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    hip_angle = calculate_angle(left_shoulder, left_hip, right_hip)
    knee_angle = calculate_angle(left_hip, left_knee, right_knee)
    
    
    vertical_displacement = np.mean([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
    
    return [torso_angle, hip_angle, knee_angle, vertical_displacement]


def train_fall_detection_model():
    
    np.random.seed(42)
    n_samples = 1000
    
    
    normal_poses = np.random.normal(loc=[170, 170, 170, 0.3], scale=[10, 10, 10, 0.05], size=(n_samples//2, 4))
    normal_labels = np.zeros(n_samples//2)
    
    
    fall_poses = np.random.normal(loc=[60, 90, 90, 0.7], scale=[20, 20, 20, 0.1], size=(n_samples//2, 4))
    fall_labels = np.ones(n_samples//2)
    
    X = np.vstack([normal_poses, fall_poses])
    y = np.hstack([normal_labels, fall_labels])
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def process_video(video_file):
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    
    model, scaler = train_fall_detection_model()
    
    
    video_placeholder = st.empty()
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            
            mp_drawing.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
           
            features = extract_pose_features(results.pose_landmarks.landmark)
            
            
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            
            
            status = "FALL DETECTED!" if prediction == 1 else "Normal"
            color = (255, 0, 0) if prediction == 1 else (0, 255, 0)
            cv2.putText(frame_rgb, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2, cv2.LINE_AA)
            
       
        video_placeholder.image(frame_rgb)
        time.sleep(1/fps)
    
    cap.release()

def main():
    st.title("AI-Powered Human Fall Detection System")
    
    st.write("""
    This application uses MediaPipe Pose Detection and Machine Learning to detect falls in video footage.
    Upload a video to analyze it for potential falls.
    """)
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        st.write("Processing video...")
        process_video(uploaded_file)
        st.write("Processing complete!")

if __name__ == "__main__":
    main()