import streamlit as st
import torch
import whisper
import sounddevice as sd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from deep_translator import GoogleTranslator
import soundfile as sf


st.set_page_config(
    page_title="Multilingual Sentiment Chatbot", layout="centered")

st.image("Logo.png", use_column_width=True)

translator = GoogleTranslator(source='auto', target='en')

# Define device for model inference (CPU or GPU)
device = torch.device('cuda')

# Load Whisper model
whisper_model = whisper.load_model("medium").to(device)


# Load fine-tuned DistilBERT model  for sentiment classification
def load_finetuned_model():
    model_directory = r"FinetunedModel"
    tokenizer = DistilBertTokenizer.from_pretrained(
        "C:\Users\Mohamed Maklad\Desktop\Graduation Project\FinetunedModel")
    model = DistilBertForSequenceClassification.from_pretrained(
        "C:\Users\Mohamed Maklad\Desktop\Graduation Project\FinetunedModel").to(device)
    return tokenizer, model


tokenizer_bert, model_bert = load_finetuned_model()

# Function to record audio for voice input


def record_audio(duration=5, fs=16000):
    st.write("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs,
                   channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    return np.squeeze(audio)

# Transcribe the audio using Whisper


def transcribe_audio(audio):
    st.write("Transcribing audio...")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    result = whisper.decode(whisper_model, mel)
    return result.text

# Classify sentiment using the fine-tuned DistilBERT model


def classify_text(transcription):
    st.write(f"Classifying sentiment for text: {transcription}")
    # Translate to English if Arabic is detected
    if any('\u0600' <= char <= '\u06FF' for char in transcription):
        transcription = translator.translate(transcription)

    inputs = tokenizer_bert(
        transcription, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model_bert(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    labels = ['negative', 'positive']
    return labels[predicted_class.item()]


##################################################################################################################
# Streamlit app
##################################################################################################################
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Take text input
if prompt := st.chat_input("Enter your text or press the mic to record audio:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = classify_text(prompt)
    with st.chat_message("assistant"):
        st.markdown(f"Sentiment: {response}")
    st.session_state.messages.append(
        {"role": "assistant", "content": f"Sentiment: {response}"})

# Take voice input using a button
if st.button("Record Voice Input"):
    audio_data = record_audio(duration=5)
    transcription = transcribe_audio(audio_data)

    # Display transcription on chat UI
    with st.chat_message("user"):
        st.markdown(f"Transcribed: {transcription}")
    st.session_state.messages.append(
        {"role": "user", "content": f"Transcribed: {transcription}"})

    # Classify sentiment for the transcribed text
    sentiment = classify_text(transcription)
    with st.chat_message("assistant"):
        st.markdown(f"Sentiment: {sentiment}")
    st.session_state.messages.append(
        {"role": "assistant", "content": f"Sentiment: {sentiment}"})
