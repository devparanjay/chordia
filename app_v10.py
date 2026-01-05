import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from basic_pitch.inference import predict_and_save, Model
from basic_pitch import ICASSP_2022_MODEL_PATH # Import the default model path
from mido import MidiFile
import torch

st.set_page_config(page_title="Chordia V10", layout="wide")

# --- CORE UTILITIES ---

def get_note_name(midi_number):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    return f"{notes[midi_number % 12]}{octave}"

def parse_midi_to_list(midi_path):
    mid = MidiFile(midi_path)
    note_events = []
    current_time = 0
    # Basic Pitch uses 480 ticks per beat
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_events.append({
                    "Timestamp (s)": round(current_time / 480.0, 2),
                    "Note": get_note_name(msg.note),
                    "Instrument Style": "String/Melodic"
                })
    return sorted(note_events, key=lambda x: x['Timestamp (s)'])

# --- UI ---

st.title("🎸 Chordia V10")
st.markdown("Supports **Guitar, Violin, Piano, and Bass**. Detects individual notes and timing.")

uploaded_file = st.file_uploader("Upload Music File", type=["mp3", "wav"])

if uploaded_file:
    temp_audio = f"temp_{uploaded_file.name}"
    with open(temp_audio, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Visualizer
    y, sr = librosa.load(temp_audio)
    fig, ax = plt.subplots(figsize=(12, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', ax=ax)
    st.pyplot(fig)
    st.audio(uploaded_file)

    # Instrument Mode Selection
    mode = st.selectbox("Select Primary Instrument to Transcribe", 
                        ["Guitar/Violin (Polyphonic)", "Bass Guitar", "Piano", "Drums (Beat Only)"])

    if st.button("🚀 Analyze Instrument", type="primary"):
        with st.spinner(f"Extracting {mode} notes..."):
            try:
                # Load the model explicitly
                model = Model(ICASSP_2022_MODEL_PATH)
            
                output_dir = "."
                # Pass the 'model' as the first argument
                predict_and_save(
                    audio_path_list=[temp_audio],
                    output_directory=output_dir,
                    save_midi=True,
                    sonify_midi=False,
                    save_model_outputs=False,
                    save_notes=True,
                    model_or_model_path=model # Add this line
                )
                
                # Basic Pitch generates a file named: [temp_audio]_basic_pitch.mid
                midi_output = temp_audio.replace(".mp3", "_basic_pitch.mid").replace(".wav", "_basic_pitch.mid")
                
                st.session_state['midi_ready'] = midi_output
                st.session_state['note_list'] = parse_midi_to_list(midi_output)
                st.success("Transcription Complete!")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if 'midi_ready' in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Playback Guide (Letter Notes)")
            df = pd.DataFrame(st.session_state['note_list'])
            st.dataframe(df, height=400, use_container_width=True)
        
        with col2:
            st.subheader("📥 Export")
            with open(st.session_state['midi_ready'], "rb") as f:
                st.download_button("Download Universal MIDI", f, "instrument_track.mid")
            
            # Formatting notes for easy reading on an instrument
            text_notes = "\n".join([f"{n['Timestamp (s)']}s: {n['Note']}" for n in st.session_state['note_list']])
            st.download_button("Download Note Sheet (.txt)", text_notes, "sheet_music.txt")

    os.remove(temp_audio)