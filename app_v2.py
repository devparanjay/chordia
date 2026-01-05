import streamlit as st
import autochord
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from midiutil import MIDIFile
from io import BytesIO

st.set_page_config(page_title="Chordia V2", layout="wide")

# --- Enhanced MIDI Logic ---
def create_midi(chord_data):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    
    note_map = {'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 
                'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 
                'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71}

    for start, end, chord_str in chord_data:
        if chord_str in ['N', '']: continue
        
        # Split "C:maj" into root="C", quality="maj"
        parts = chord_str.split(':')
        root_name = parts[0]
        quality = parts[1] if len(parts) > 1 else 'maj'
        
        if root_name in note_map:
            root_note = note_map[root_name]
            duration = max(0.1, end - start)
            
            # Determine intervals based on quality
            third = 4 if 'maj' in quality else 3
            fifth = 7
            
            # Add Notes (Track, Channel, Pitch, Time, Duration, Volume)
            midi.addNote(0, 0, root_note, start, duration, 80)      # Root
            midi.addNote(0, 0, root_note + third, start, duration, 65) # 3rd
            midi.addNote(0, 0, root_note + fifth, start, duration, 65) # 5th
            
    bin_data = BytesIO()
    midi.writeFile(bin_data)
    return bin_data.getvalue()

# --- Streamlit UI ---
st.title("🎵 Pro Chord Analyzer & MIDI Generator")

uploaded_file = st.file_uploader("Upload MP3 for Analysis", type=["mp3", "wav"])

if uploaded_file:
    # Save file locally for processing
    temp_filename = f"temp_{uploaded_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Layout: Top bar for Audio Player
    st.audio(uploaded_file)
    
    with st.spinner("Processing Audio with Deep Learning..."):
        # 1. Run Autochord
        chords = autochord.recognize(temp_filename)
        
        # 2. Extract Spectrogram Data
        y, sr = librosa.load(temp_filename)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Layout: Two columns for Visuals and Data
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Frequency Spectrogram")
        fig, ax = plt.subplots(figsize=(12, 5))
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax, cmap='magma')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Log-Frequency Power Spectrogram")
        st.pyplot(fig)

    with col2:
        st.subheader("Detected Progression")
        df = pd.DataFrame(chords, columns=["Start (s)", "End (s)", "Chord"])
        st.dataframe(df, width='stretch', height=400)
        
        # MIDI Export Button
        midi_data = create_midi(chords)
        st.download_button(
            label="⬇️ Download MIDI File",
            data=midi_data,
            file_name=f"{uploaded_file.name}_chords.mid",
            mime="audio/midi",
            use_container_width=True
        )

    # Clean up
    if os.path.exists(temp_filename):
        os.remove(temp_filename)