import streamlit as st
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from chord_extractor.extractors import Chordino
from midiutil import MIDIFile
from io import BytesIO

st.set_page_config(page_title="Pro AI Transcription Lab", layout="wide")

# --- CORE LOGIC: CHORD TO NOTES ---
def get_chord_notes(chord_str):
    """Maps chord names to MIDI intervals."""
    note_base = {'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 
                 'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 
                 'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 71, 'B': 71}
    
    root = ""
    for n in sorted(note_base.keys(), key=len, reverse=True):
        if chord_str.startswith(n):
            root = n
            break
    if not root: return []
    
    root_midi = note_base[root]
    intervals = [0, 4, 7] # Default Major
    
    c_low = chord_str.lower()
    if 'min' in c_low or 'm' in chord_str[len(root):len(root)+2]:
        intervals[1] = 3 # Minor 3rd
    if '7' in chord_str:
        intervals.append(10)
    if 'maj7' in c_low:
        intervals[-1] = 11
        
    return [root_midi + i for i in intervals]

# --- INVERSION ENGINE (Voice Leading) ---
def apply_inversion(current_notes, previous_avg):
    """Shifts notes by octaves to stay near the previous chord's pitch."""
    if previous_avg == 0:
        return current_notes
    
    # Try different octave offsets (-1, 0, +1) to find the closest match
    best_notes = current_notes
    min_diff = float('inf')
    
    for octave_shift in [-12, 0, 12]:
        shifted = [n + octave_shift for n in current_notes]
        avg = sum(shifted) / len(shifted)
        diff = abs(avg - previous_avg)
        if diff < min_diff:
            min_diff = diff
            best_notes = shifted
            
    return best_notes

def create_inverted_midi(chord_data):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    
    prev_avg = 60 # Start near Middle C
    for start, end, chord_val in chord_data:
        raw_notes = get_chord_notes(chord_val)
        if not raw_notes: continue
        
        # Apply voice leading logic
        inverted_notes = apply_inversion(raw_notes, prev_avg)
        prev_avg = sum(inverted_notes) / len(inverted_notes)
        
        duration = max(0.1, end - start)
        for note in inverted_notes:
            midi.addNote(0, 0, int(note), start, duration, 75)
            
    bin_data = BytesIO()
    midi.writeFile(bin_data)
    return bin_data.getvalue()

# --- STREAMLIT UI ---
st.title("🎹 Chordia V6")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- 1. Audio Processing & Spectrogram ---
    y, sr = librosa.load(temp_path)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Log-Frequency Spectrogram")
        fig, ax = plt.subplots(figsize=(10, 4))
        # Use CQT (Constant-Q Transform) which is better for music than STFT
        C = np.abs(librosa.cqt(y, sr=sr))
        librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), 
                                 sr=sr, x_axis='time', y_axis='cqt_note', ax=ax, cmap='magma')
        st.pyplot(fig)
        st.audio(uploaded_file)

    # --- 2. Chord Extraction & MIDI ---
    with st.spinner("Analyzing Harmonies & Applying Voice Leading..."):
        chordino = Chordino(roll_on=True)
        raw_output = chordino.extract(temp_path)
        
        chords = []
        for i in range(len(raw_output) - 1):
            if raw_output[i].chord not in ['N', '']:
                chords.append((raw_output[i].timestamp, raw_output[i+1].timestamp, raw_output[i].chord))

    with col2:
        st.subheader("Transcription")
        df = pd.DataFrame(chords, columns=["Start", "End", "Chord"])
        st.dataframe(df, height=300, use_container_width=True)
        
        midi_data = create_inverted_midi(chords)
        st.download_button(
            label="⬇️ Download Inverted MIDI",
            data=midi_data,
            file_name="pro_transcription.mid",
            mime="audio/midi",
            width='stretch'
        )

    os.remove(temp_path)