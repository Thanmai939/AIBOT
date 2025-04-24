import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
import transcribe_audio
import tempfile
import os
from utils import get_answer

# Initialize floating features for the interface
float_init()

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How may I assist you today?"}]
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""

initialize_session_state()
st.title("AI Chatbot 🤖")

def handle_text_input():
    if st.session_state.text_prompt:
        st.session_state.messages.append({"role": "user", "content": st.session_state.text_prompt})
        with st.chat_message("user"):
            st.write(st.session_state.text_prompt)
        st.session_state["text_prompt"] = ""  # Clear the text input after processing

footer_container = st.container()
with footer_container:
    col1, col2 = st.columns([1, 2])  # Adjust width as needed

    with col1:
        audio_bytes = audio_recorder(key="audio_recorder")

    with col2:
        text_input = st.text_input("Or type your message here:", key="text_prompt", on_change=handle_text_input,
                                    value=st.session_state.get("text_prompt", ""))

# Display the stored messages on the interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    print("Audio received")
    with st.spinner("Transcribing..."):
        try:
            
            transcript = transcribe_audio.transcribe_audio(audio_bytes)
            print("Thanmai")
            if transcript:
                st.session_state["text_prompt"] = transcript # Fill the text field
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            print(f"Error during transcription: {e}")

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

footer_container.float("bottom: 0rem;")