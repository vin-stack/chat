# -*- coding: utf-8 -*-

#!pip install -q -U transformers==4.37.2
#!pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
#!pip install -q git+https://github.com/openai/whisper.git
#!pip install -q gTTS

import torch
from transformers import BitsAndBytesConfig, pipeline

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline("image-to-text",
                model=model_id,
                model_kwargs={"quantization_config": quantization_config})

import whisper
import warnings
from gtts import gTTS

import locale
print(locale.getlocale())  # Before running the pipeline
# Run the pipeline
print(locale.getlocale())  # After running the pipeline

import nltk
nltk.download('punkt')
from nltk import sent_tokenize

import datetime

warnings.filterwarnings("ignore")

import numpy as np

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")

import whisper
model = whisper.load_model("medium", device=DEVICE)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

import re

def transcribe(audio):

    # Check if the audio input is None or empty
    if audio is None or audio == '':
        return ('','',None)  # Return empty strings and None audio file

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    result_text = result.text

    return result_text

def text_to_speech(text, file_path):
    language = 'en'

    audioobj = gTTS(text=text,
                    lang=language,
                    slow=False)

    audioobj.save(file_path)

    return file_path

#!ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 10 -q:a 9 -acodec libmp3lame Temp.mp3

import streamlit as st

def process_audio(audio_path):
    # Process the audio file
    speech_to_text_output = transcribe(audio_path)
    
    # Assuming you want the assistant to repeat what was said
    assistant_output = speech_to_text_output
    
    # Convert the assistant output to speech
    processed_audio_path = text_to_speech(assistant_output, "Temp3.mp3")

    return speech_to_text_output, assistant_output, processed_audio_path

# Streamlit UI
st.title("Voice Assistant")

audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"])

if audio_file:
    st.audio(audio_file, format='audio/wav')

    # Process the audio file
    speech_to_text_output, assistant_output, processed_audio_path = process_audio(audio_file)

    st.write("Speech to Text Output:")
    st.write(speech_to_text_output)

    st.write("Assistant Output:")
    st.write(assistant_output)

    st.write("Assistant Response Audio:")
    st.audio(processed_audio_path, format='audio/mp3')

