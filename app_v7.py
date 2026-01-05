import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from piano_transcription_inference import PianoTranscription, sample_rate
import torch

st.set_page_config(page_title="High-Fidelity Piano Transcriber", layout="wide")

# --- TRANSCRIPTION ENGINE ---
def transcribe_piano(audio_path):
    # Load audio at the specific sample rate the model expects
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Initialize the transcriptor (Uses CPU by default for VPS compatibility)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transcriptor = PianoTranscription(device=device)
    
    # This generates a high-fidelity MIDI file directly
    output_midi_path = "output_transcription.mid"
    transcriptor.transcribe(audio, output_midi_path)
    
    return output_midi_path

# --- UI ---
st.title("🎹 Deep Learning Piano Transcription")
st.info("Best for Classical, Jazz, and Arpeggiated Piano (like Für Elise)")

uploaded_file = st.file_uploader("Upload Piano MP3", type=["mp3", "wav"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- Spectrogram ---
    y, sr = librosa.load(temp_path)
    fig, ax = plt.subplots(figsize=(10, 3))
    # Mel-spectrogram is better for visualizing piano energy
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel', ax=ax)
    st.pyplot(fig)

    with st.spinner("AI is transcribing every individual note..."):
        try:
            midi_file = transcribe_piano(temp_path)
            
            with open(midi_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Full Piano MIDI",
                    data=f,
                    file_name="transcribed_piano.mid",
                    mime="audio/midi"
                )
            st.success("Success! This MIDI contains every note played, not just chords.")
            
        except Exception as e:
            st.error(f"Transcription Error: {e}")
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)