import streamlit as st
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
from land_change import *
from threat import *
from species_mont import *
#from land_change import *

# Load Models
#st.sidebar.info("Loading AI models... Please wait.")
species_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
species_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50").eval()
yolo_model = YOLO("Species_Habit Monitoring\\yolov8n.pt")
threat_model = pipeline("image-classification", model="nateraw/vit-base-beans")

# Habitat Analysis Model
class HabitatAnalyzer:
    def __init__(self):
        self.CLASSES = ['vegetation', 'water', 'urban', 'barren']
    
    def analyze_vegetation(self, image_array):
        ndvi = (image_array[:, :, 3] - image_array[:, :, 0]) / (image_array[:, :, 3] + image_array[:, :, 0] + 1e-8)
        return ndvi
    
    def detect_land_changes(self, image1, image2):
        return cv2.absdiff(image1, image2)

class SpeciesMonitor:
    def detect_species(self, image):
        inputs = species_processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = species_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        top_preds = torch.topk(probs, 5)
        results = []
        for score, idx in zip(top_preds.values[0], top_preds.indices[0]):
            results.append({
                'species': species_model.config.id2label[idx.item()],
                'confidence': score.item() * 100
            })
        return results

    def count_population(self, image):
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        results = yolo_model(img_array)
        
        # Draw bounding boxes
        annotated_img = img_array.copy()
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calculate density
        total_pixels = img_array.shape[0] * img_array.shape[1]
        count = len(results[0].boxes)
        density = count / total_pixels if total_pixels > 0 else 0
        
        return {
            'count': count,
            'density': density,
            'marked_image': annotated_img
        }

    def assess_health(self, image):
        # Simplified health assessment based on visual features
        img_array = np.array(image)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Calculate basic health indicators
        brightness = np.mean(hsv[:,:,2])
        saturation = np.mean(hsv[:,:,1])
        color_variance = np.std(hsv[:,:,0])
        
        # Normalize scores to 0-100 range
        health_score = min(100, (brightness * 0.4 + saturation * 0.3 + color_variance * 0.3) * 100 / 255)
        
        # Determine status based on score
        if health_score >= 80:
            status = "Excellent"
        elif health_score >= 60:
            status = "Good"
        elif health_score >= 40:
            status = "Fair"
        else:
            status = "Poor"
        
        return {
            'status': status,
            'score': health_score,
            'indicators': {
                'brightness': brightness,
                'saturation': saturation,
                'color_variance': color_variance
            }
        }

def detect_threat(image, labels):
    results = threat_model(image)
    for result in results:
        if result['label'] in labels and result['score'] > 0.5:
            return f"{result['label']} Detected with confidence {result['score']:.2f}"
    return "No Threat Detected"

def main():
    habitat_analyzer = HabitatAnalyzer()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select an Analysis Type:",
                             ["Species Monitoring", "land Change Detection", "Animal Monitoring", "Threat Detection"])

    if option == "Species Monitoring":
        st.title("Species Identification")
        monitoring_system = SpeciesMonitoringSystem()
    
    # File uploader
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add a progress bar
            progress_bar = st.progress(0)
            
            with st.spinner("Analyzing image..."):
                # Create three columns for results
                col1, col2, col3 = st.columns(3)
                
                # Species Detection
                progress_bar.progress(30)
                species_results = monitoring_system.detect_species(image)
                
                # Population Count
                progress_bar.progress(60)
                count, marked_image = monitoring_system.count_population(image)
                
                # Health Assessment
                progress_bar.progress(90)
                health_status, health_score, health_indicators = monitoring_system.assess_health(image)
                
                # Display results in columns
                with col1:
                    st.subheader("🔍 Species Detection")
                    for species, confidence in species_results:
                        st.write(f"**{species.title()}**")
                        st.progress(confidence/100)
                        st.caption(f"Confidence: {confidence:.1f}%")
                
                with col2:
                    st.subheader("👥 Population Count")
                    st.write(f"**Detected Animals:** {count}")
                    st.image(marked_image, caption="Detection Visualization", use_column_width=True)
                
                with col3:
                    st.subheader("💪 Health Assessment")
                    st.write(f"**Status:** {health_status}")
                    st.write(f"**Overall Score:** {health_score:.1f}/100")
                    
                    # Display health indicators
                    for indicator, value in health_indicators.items():
                        st.write(f"**{indicator}:**")
                        st.progress(value/100)
                        st.caption(f"{value:.1f}%")
            
            # Complete progress bar
            progress_bar.progress(100)
            
            # Add analysis details
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Analysis Details")
            st.sidebar.text(f"Analyzed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.sidebar.text(f"Image size: {image.size}")
            
            # Add download buttons for results
            st.markdown("---")
            st.subheader("📊 Export Results")
            
            # Create a summary of results
            summary = f"""Wildlife Monitoring Analysis Report
            Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

            Species Detection Results:
            {'-' * 30}
            """
            for species, confidence in species_results:
                summary += f"\n{species.title()}: {confidence:.1f}% confidence"
            
            summary += f"""\n\nPopulation Count:
            {'-' * 30}
            Total detected: {count} individuals

            Health Assessment:
            {'-' * 30}
            Status: {health_status}
            Overall Score: {health_score:.1f}/100
            """
            for indicator, value in health_indicators.items():
                summary += f"\n{indicator}: {value:.1f}%"
            
            # Create download button
            st.download_button(
                label="Download Analysis Report",
                data=summary,
                file_name="wildlife_analysis_report.txt",
                mime="text/plain"
            )

    elif option == "land Change Detection":
        st.title("🌍 Land Change Detection")
        #st.write("Upload two images to detect changes over time.")

        uploaded_file2 = st.file_uploader("Upload first image", type=['tif', 'png', 'jpg'])
        uploaded_file3 = st.file_uploader("Upload second image", type=['tif', 'png', 'jpg'])

        if uploaded_file2 is not None and uploaded_file3 is not None:
            detect_land_changes(uploaded_file2, uploaded_file3)

    elif option == "Animal Monitoring":
        st.title("Animal Monitoring")
        uploaded_file4 = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4"])
        
        if uploaded_file4:
            if uploaded_file4.type.startswith("image"):
                file_bytes = np.asarray(bytearray(uploaded_file4.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                if image is None:
                    st.error("Error loading image. Please upload a valid image file.")
                else:
                    results = yolo_model(image)
                    for result in results:
                        for box in result.boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                    st.image(image, caption="Detected Animals", channels="BGR")
                    st.write(f"Estimated Count: {len(results[0].boxes)}")
            
            elif uploaded_file4.type.startswith("video"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file4.read())
                cap = cv2.VideoCapture(tfile.name)
                
                if not cap.isOpened():
                    st.error("Error loading video. Please upload a valid video file.")
                else:
                    stframe = st.empty()
                    st.write("Processing video...")
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame = cv2.resize(frame, (640, 480))
                        results = yolo_model(frame)
                        
                        for result in results:
                            for box in result.boxes.xyxy:
                                x1, y1, x2, y2 = map(int, box[:4])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        stframe.image(frame, channels="BGR")
                        time.sleep(0.03)
                    
                    cap.release()

    elif option == "Threat Detection":
        st.title("Threat Detection and Prevention")
        st.sidebar.header("Choose Threat Detection")
        detection_option = st.sidebar.selectbox(
            "Select an option", 
            ["Poaching Alerts"]
        )
        
        if detection_option in ["Poaching Alerts"]:
            uploaded_file7 = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

            if uploaded_file7:
                image = Image.open(uploaded_file7)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if detection_option == "Poaching Alerts":
                    st.subheader("🎯 Poaching Activity Detection")
                    
                    with st.spinner("Analyzing image for potential poaching activities..."):
                        # Use YOLO model to detect objects that might indicate poaching
                        results = yolo_model(image)
                        
                        # Define objects that might indicate poaching
                        poaching_objects = ['person', 'gun', 'knife', 'truck', 'car']
                        detections = {}
                        
                        # Process YOLO results
                        for result in results:
                            for box in result.boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                label = result.names[cls]
                                
                                if label in poaching_objects and conf > 0.3:  # Confidence threshold
                                    detections[label] = conf
                        
                        # Display results
                        if detections:
                            for obj, conf in detections.items():
                                st.progress(conf)
                                st.write(f"{obj.title()}: {conf*100:.1f}% confidence")
                            
                            # Display annotated image
                            annotated_img = np.array(image)
                            for result in results:
                                for box in result.boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            st.image(annotated_img, caption="Detected Objects", use_column_width=True)
                            
                            if any(conf > 0.7 for conf in detections.values()):
                                st.error("⚠️ High-risk poaching activity detected! Alert sent to authorities.")
                        else:
                            st.success("No suspicious activities detected.")
                            
                # elif detection_option == "Invasive Species Detection":
                #     st.subheader("🌿 Invasive Species Detection")
                    
                #     with st.spinner("Analyzing image for invasive species..."):
                #         # Use the threat model for invasive species detection
                #         detection_result = threat_model(image)
                        
                #         # Process and display results
                #         significant_detections = [
                #             result for result in detection_result 
                #             if result['score'] > 0.5  # Confidence threshold
                #         ]
                        
                #         if significant_detections:
                #             for detection in significant_detections:
                #                 confidence = detection['score']
                #                 species = detection['label']
                                
                #                 st.progress(confidence)
                #                 st.write(f"{species.title()}: {confidence*100:.1f}% confidence")
                            
                #             # Display historical data
                #             st.subheader("Historical Sightings")
                #             dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
                #             data = pd.DataFrame({
                #                 'Date': dates,
                #                 'Sightings': np.random.randint(1, 10, size=10)
                #             })
                            
                #             # Create plot using matplotlib instead of plotly
                #             fig, ax = plt.subplots()
                #             ax.plot(data['Date'], data['Sightings'])
                #             ax.set_title('Invasive Species Sightings Trend')
                #             ax.set_xlabel('Date')
                #             ax.set_ylabel('Sightings')
                #             plt.xticks(rotation=45)
                #             plt.tight_layout()
                #             st.pyplot(fig)
                #         else:
                #             st.success("No invasive species detected.")

if __name__ == "__main__":
    main()