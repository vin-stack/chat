import torch
from transformers import BitsAndBytesConfig, pipeline
import whisper
import streamlit as st
from gtts import gTTS
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np

# Initialize the models and configurations
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("text-generation", model=model_id, model_kwargs={"quantization_config": quantization_config})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium", device=DEVICE)

# Function to transcribe audio
def transcribe(audio):
    try:
        if audio is None:
            raise ValueError("No audio input provided.")
        
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        return result.text
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

class AudioRecorder(VideoTransformerBase):
    def __init__(self):
        self.sampling_rate = 44100
        self.sample_width = 2
        self.channels = 1
        self.duration = 10
        self.audio = np.array([], dtype=np.int16)

    def transform(self, frame):
        self.audio = np.append(self.audio, np.frombuffer(frame.to_ndarray(format="wav"), dtype=np.int16))
        if len(self.audio) >= self.sampling_rate * self.duration:
            st.stop()
        return frame

webrtc_ctx = webrtc_streamer(key="audio-recorder", video_transformer_factory=AudioRecorder)

if webrtc_ctx.state.playing:
    st.text("Recording...")

if webrtc_ctx.state == "stopped":
    audio = whisper.load_audio(io.BytesIO(self.audio.tobytes()), sample_rate=self.sampling_rate)
    transcription = transcribe(audio)
    st.text(f"Transcription: {transcription}")

    prompt = f"{transcription}\nASSISTANT:"
    outputs = pipe(prompt, max_new_tokens=200)
    raw_output = outputs[0]["generated_text"].strip()
    assistant_response = raw_output.replace("USER:", "").replace("ASSISTANT:", "").strip()
    st.text(f"ChatGPT Output: {assistant_response}")

    audio_response_path = text_to_speech(assistant_response, "response.mp3")
    st.audio(audio_response_path)
