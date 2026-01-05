import streamlit as st
import os
import pandas as pd
from chord_extractor.extractors import Chordino
from midiutil import MIDIFile
from io import BytesIO

# --- ACCURATE VOICING ENGINE ---
def get_chord_notes(chord_str):
    """Parses chord strings into MIDI note lists across the piano range."""
    # Base MIDI numbers for Octave 4 (Middle C starts at 60)
    note_base = {'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 
                 'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 
                 'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71}
    
    # Identify the root
    root = ""
    for n in sorted(note_base.keys(), key=len, reverse=True):
        if chord_str.startswith(n):
            root = n
            break
    
    if not root: return []
    
    # Start with Octave 3 for a fuller piano sound
    root_midi = note_base[root] - 12 
    intervals = [0, 7] # Root and Fifth
    
    # Determine Quality
    c_low = chord_str.lower()
    if 'min' in c_low or 'm' in chord_str[len(root):len(root)+2]:
        intervals.append(3) # Minor 3rd
    else:
        intervals.append(4) # Major 3rd
        
    # Extensions
    if '7' in chord_str:
        intervals.append(10)
    if 'maj7' in c_low:
        intervals[-1] = 11
        
    return [root_midi + i for i in intervals]

def create_pro_midi(chord_data):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    
    for start, end, chord_val in chord_data:
        midi_notes = get_chord_notes(chord_val)
        duration = max(0.2, end - start)
        for note in midi_notes:
            # Velocity 70 for a soft piano touch
            midi.addNote(0, 0, note, start, duration, 70)
            
    bin_data = BytesIO()
    midi.writeFile(bin_data)
    return bin_data.getvalue()

# --- STREAMLIT UI ---
st.title("🎹 Chordia V5")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Transcribing to 88-key Piano MIDI..."):
        chordino = Chordino(roll_on=True)
        raw_chords = chordino.extract(temp_path)
        
        # Convert to list of (start, end, chord)
        chords = []
        for i in range(len(raw_chords) - 1):
            chords.append((raw_chords[i].timestamp, raw_chords[i+1].timestamp, raw_chords[i].chord))

        df = pd.DataFrame(chords, columns=["Start", "End", "Chord"])
        st.table(df.head(15)) # Preview first 15

        midi_data = create_pro_midi(chords)
        st.download_button("⬇️ Download 88-Key MIDI", midi_data, "piano_chords.mid")

    os.remove(temp_path)