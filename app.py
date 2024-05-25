import torch
from transformers import BitsAndBytesConfig, pipeline
import streamlit as st
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import numpy as np
import io

# Initialize the models and configurations
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("text-generation", model=model_id, model_kwargs={"quantization_config": quantization_config})

# Function to transcribe audio
def transcribe(audio):
    try:
        if audio is None:
            raise ValueError("No audio input provided.")
        
        audio = torch.tensor(audio)
        mel = torch.log(torch.abs(torch.stft(audio, n_fft=2048, hop_length=512, window=torch.hann_window(2048))))
        result = pipe(mel)
        return result[0]['generated_text']
    except Exception as e:
        return str(e)

# Function to convert text to speech
def text_to_speech(text, file_path):
    try:
        audioobj = gTTS(text=text, lang='en', slow=False)
        audioobj.save(file_path)
        return file_path
    except Exception as e:
        return str(e)

# Streamlit interface
st.title("Learn OpenAI Whisper: Voice processing with Whisper and Llava")
st.write("Record your voice and receive audio response.")

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []
    
    def recv(self, frame):
        self.audio_frames.append(frame.to_ndarray().astype(np.float32))
        return True

    def post_process(self):
        audio_data = np.concatenate(self.audio_frames)
        transcription = transcribe(audio_data)
        st.text(f"Transcription: {transcription}")

        prompt = f"{transcription}\nASSISTANT:"
        outputs = pipe(prompt, max_new_tokens=200)
        raw_output = outputs[0]["generated_text"].strip()
        assistant_response = raw_output.replace("USER:", "").replace("ASSISTANT:", "").strip()
        st.text(f"ChatGPT Output: {assistant_response}")

        audio_response_path = text_to_speech(assistant_response, "response.mp3")
        st.audio(audio_response_path)

webrtc_ctx = webrtc_streamer(key="audio-recorder", audio_processor_factory=AudioRecorder)

if webrtc_ctx.state.playing:
    st.text("Recording...")
