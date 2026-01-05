import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from piano_transcription_inference import PianoTranscription, sample_rate
from mido import MidiFile
import torch

st.set_page_config(page_title="Chordia V8", layout="wide")

# --- FUNCTIONS ---

def get_note_name(midi_number):
    """Converts MIDI number to musical note name (e.g., 60 -> C4)."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    name = notes[midi_number % 12]
    return f"{name}{octave}"

def parse_midi_to_list(midi_path):
    """Extracts a human-readable list of notes from the generated MIDI."""
    mid = MidiFile(midi_path)
    note_events = []
    current_time = 0
    
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                # Convert ticks to seconds (assuming standard 480 ticks per beat at 120bpm)
                # For simplicity, we use the model's direct output timing
                note_events.append({
                    "Timestamp (s)": round(current_time / 480.0, 2), # Approximation
                    "Note": get_note_name(msg.note),
                    "Intensity": msg.velocity
                })
    return sorted(note_events, key=lambda x: x['Timestamp (s)'])

# --- UI SETUP ---

st.title("🎹 Chordia V8")
st.markdown("Designed for complex pieces**. Upload, Analyze, and Play.")

uploaded_file = st.file_uploader("Upload Piano MP3/WAV", type=["mp3", "wav"])

if uploaded_file:
    # Save file to disk
    temp_audio = f"temp_{uploaded_file.name}"
    with open(temp_audio, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. Visualization (Always show after upload)
    y, sr = librosa.load(temp_audio)
    fig, ax = plt.subplots(figsize=(12, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax, cmap='magma')
    st.pyplot(fig)
    st.audio(uploaded_file)

    # 2. Analyze Button
    if st.button("🚀 Run Deep Analysis", type="primary"):
        with st.spinner("AI is listening to every note... This takes 10-30 seconds."):
            try:
                # Transcription Engine
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                transcriptor = PianoTranscription(device=device)
                midi_output = "transcription.mid"
                
                # Run the model
                audio, _ = librosa.load(temp_audio, sr=sample_rate, mono=True)
                transcriptor.transcribe(audio, midi_output)
                
                # Store results in session state
                st.session_state['midi_ready'] = midi_output
                st.session_state['note_list'] = parse_midi_to_list(midi_output)
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"Error: {e}")

    # 3. Results Section (Only shows if analysis is done)
    if 'midi_ready' in st.session_state:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📜 Note-by-Note Guide")
            st.info("Follow these notes in order to play the piece.")
            notes_df = pd.DataFrame(st.session_state['note_list'])
            st.dataframe(notes_df, height=400, use_container_width=True)

        with col2:
            st.subheader("📥 Export Data")
            # Download MIDI
            with open(st.session_state['midi_ready'], "rb") as f:
                st.download_button(
                    label="⬇️ Download MIDI File",
                    data=f,
                    file_name="piano_transcription.mid",
                    mime="audio/midi",
                    width='stretch'
                )
            
            # Download Text Notes
            csv = notes_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Letter Notes (CSV)",
                data=csv,
                file_name="notes_guide.csv",
                mime="text/csv",
                width='stretch'
            )

    # Cleanup
    if os.path.exists(temp_audio):
        os.remove(temp_audio)