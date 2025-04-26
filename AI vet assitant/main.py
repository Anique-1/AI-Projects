import streamlit as st
import speech_recognition as sr
import requests
from gtts import gTTS
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SERPAPI_KEY = "ENTER SERPAPI"  # Add SERPAPI_KEY to your .env file
AIMLAPI_KEY = "AIMLAPI" # Add AIMLAPI_KEY to your .env file

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to transcribe audio
def transcribe_audio():
    try:
        with sr.Microphone() as source:
            st.write("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.write("Listening... Speak about your pet's symptoms.")
            audio = recognizer.listen(source, timeout=5)
            st.write("Processing audio...")
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"

# Function to generate vet suggestion using AIMLAPI
def generate_vet_suggestion(text):
    try:
        url = "https://api.aimlapi.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIMLAPI_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": f"My pet is showing the following symptoms: {text}. What could be the issue?"
                }
            ],
            "max_tokens": 100
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating suggestion: {e}"

# Function to find nearby vet clinics using SerpAPI
def find_vet_clinics(location="New York, NY"):
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_maps",
            "q": "veterinary clinic",
            "ll": "@40.7128,-74.0060,15z",  # Default to New York coordinates
            "type": "search",
            "api_key": SERPAPI_KEY
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        clinics = []
        for place in data.get("local_results", [])[:3]:  # Limit to top 3 results
            name = place.get("title")
            address = place.get("address")
            phone = place.get("phone", "Not available")
            clinics.append({"name": name, "address": address, "phone": phone})
        return clinics
    except Exception as e:
        return [{"name": "Error", "address": f"Could not fetch clinics: {e}", "phone": "N/A"}]

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    sound_file = BytesIO()
    tts.write_to_fp(sound_file)
    return sound_file

# Streamlit app
def main():
    st.set_page_config(page_title="AI Veterinary Assistant", layout="wide")
    st.title("🐾 AI Veterinary Assistant")
    st.write("Describe your pet's symptoms via audio or text, and get suggestions and nearby vet clinics.")

    # Input method selection
    input_method = st.radio("Choose input method:", ("Text", "Audio"))

    # Initialize session state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = ""
    if "clinics" not in st.session_state:
        st.session_state.clinics = []

    # Handle input
    if input_method == "Text":
        st.session_state.user_input = st.text_area("Enter your pet's symptoms:", height=100)
        if st.button("Submit Text"):
            if st.session_state.user_input:
                st.session_state.suggestion = generate_vet_suggestion(st.session_state.user_input)
                st.session_state.clinics = find_vet_clinics()
            else:
                st.warning("Please enter symptoms.")
    else:
        if st.button("Record Audio"):
            st.session_state.user_input = transcribe_audio()
            if "Could not" not in st.session_state.user_input:
                st.session_state.suggestion = generate_vet_suggestion(st.session_state.user_input)
                st.session_state.clinics = find_vet_clinics()
            else:
                st.error(st.session_state.user_input)

    # Display results
    if st.session_state.user_input:
        st.subheader("Your Input:")
        st.write(st.session_state.user_input)

    if st.session_state.suggestion:
        st.subheader("Suggested Diagnosis:")
        st.write(st.session_state.suggestion)
        # Play audio response
        audio_file = text_to_speech(st.session_state.suggestion)
        st.audio(audio_file)

    if st.session_state.clinics:
        st.subheader("Nearby Veterinary Clinics:")
        for clinic in st.session_state.clinics:
            st.write(f"**{clinic['name']}**")
            st.write(f"Address: {clinic['address']}")
            st.write(f"Phone: {clinic['phone']}")
            st.write("---")

if __name__ == "__main__":
    main()