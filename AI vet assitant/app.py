import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import googlemaps
from gtts import gTTS
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_MAPS_API_KEY = "AIzaSyCcVNkwFOnyiUS1XsuZmhjYKQBGu1ztX3U"      

# Initialize Google Maps client
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# Initialize LLM pipeline (using BART for text generation)
llm = pipeline("text2text-generation", model="facebook/bart-large")

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

# Function to generate vet suggestion using LLM
def generate_vet_suggestion(text):
    prompt = f"My pet is showing the following symptoms: {text}. What could be the issue?"
    response = llm(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return response

# Function to find nearby vet clinics
def find_vet_clinics(location="New York, NY"):
    try:
        places = gmaps.places_nearby(
            location=location,
            keyword="veterinary clinic",
            radius=10000  # 10km radius
        )
        clinics = []
        for place in places.get("results", [])[:3]:  # Limit to top 3 results
            name = place.get("name")
            address = place.get("vicinity")
            # Get phone number via place details
            place_id = place.get("place_id")
            details = gmaps.place(place_id=place_id)
            phone = details.get("result", {}).get("formatted_phone_number", "Not available")
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