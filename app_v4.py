import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from chord_extractor.extractors import Chordino
from midiutil import MIDIFile
from io import BytesIO

st.set_page_config(page_title="Chordia V4", layout="wide")

# --- CORE ENGINE: CHORDINO (NNLS-CHROMA) ---
def extract_high_accuracy_chords(file_path):
    # Initialize Chordino (The high-accuracy engine)
    chordino = Chordino(roll_on=True) # roll_on improves detection of chord changes
    
    # Extract chords - returns a list of ChordChange objects
    # Each object has .chord (str), .timestamp (float)
    raw_chords = chordino.extract(file_path)
    
    # Format for our app
    processed_chords = []
    for i in range(len(raw_chords) - 1):
        start = raw_chords[i].timestamp
        end = raw_chords[i+1].timestamp
        chord_val = raw_chords[i].chord
        if chord_val not in ['N', '']:
            processed_chords.append((start, end, chord_val))
            
    return processed_chords

# --- ENHANCED MIDI LOGIC (Handles Triads & 7ths) ---
def create_precise_midi(chord_data):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    
    note_map = {'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 
                'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 
                'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71}

    for start, end, chord_str in chord_data:
        # Clean string: e.g., "Cmaj7" or "Amin"
        root = ""
        for n in ["C#", "Db", "D#", "Eb", "F#", "Gb", "G#", "Ab", "A#", "Bb", "C", "D", "E", "F", "G", "A", "B"]:
            if chord_str.startswith(n):
                root = n
                break
        
        if root in note_map:
            root_midi = note_map[root]
            duration = max(0.2, end - start)
            
            # Voicing Logic
            notes = [root_midi] # Always add root
            
            if "min" in chord_str.lower() or "m" in chord_str[len(root):len(root)+1]:
                notes.append(root_midi + 3) # Minor 3rd
            else:
                notes.append(root_midi + 4) # Major 3rd
                
            notes.append(root_midi + 7) # Perfect 5th
            
            if "7" in chord_str:
                notes.append(root_midi + 10) # Simple Dominant 7th
                
            for n in notes:
                midi.addNote(0, 0, n, start, duration, 70)
                
    bin_data = BytesIO()
    midi.writeFile(bin_data)
    return bin_data.getvalue()

# --- UI ---
st.title("🎼 Chordia V4")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file)

    with st.spinner("Running NNLS-Chroma Analysis..."):
        try:
            # 1. High Accuracy Extraction
            chords = extract_high_accuracy_chords(temp_path)
            
            # 2. Visuals
            y, sr = librosa.load(temp_path)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(12, 4))
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
                st.pyplot(fig)
                
            with col2:
                df = pd.DataFrame(chords, columns=["Start", "End", "Chord"])
                st.dataframe(df, width='stretch')
                
                midi_out = create_precise_midi(chords)
                st.download_button("Download Corrected MIDI", midi_out, "output.mid", "audio/midi")
                
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)