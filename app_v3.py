import streamlit as st
import autochord
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import os
from midiutil import MIDIFile
from io import BytesIO

# Page Setup
st.set_page_config(page_title="Pro Chord AI Lab", layout="wide")

# --- CORE ENGINE: IMPROVED ACCURACY ---
def process_and_recognize(file_path):
    """
    Cleans audio by removing percussion and normalizing volume 
    before sending it to the Deep Learning model.
    """
    # 1. Load Audio
    y, sr = librosa.load(file_path)

    # 2. Pre-emphasis (Boosts high frequencies to clarify note transients)
    y_filt = librosa.effects.preemphasis(y)

    # 3. HPSS (Isolate Harmonics)
    # This is the most critical step for accuracy in songs with drums.
    y_harmonic = librosa.effects.hpss(y_filt, margin=2.0)[0]

    # 4. Normalization (Ensures consistent volume for the AI)
    y_norm = librosa.util.normalize(y_harmonic)

    # 5. Save Processed Audio to a temporary WAV for Autochord
    clean_path = "temp_clean.wav"
    sf.write(clean_path, y_norm, sr)

    # 6. Deep Learning Recognition
    chords = autochord.recognize(clean_path)

    # Cleanup
    if os.path.exists(clean_path):
        os.remove(clean_path)
        
    return chords, y, sr

# --- MIDI EXPORT LOGIC ---
def create_midi(chord_data):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    
    note_map = {'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 
                'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 
                'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71}

    for start, end, chord_str in chord_data:
        if chord_str in ['N', '']: continue
        
        parts = chord_str.split(':')
        root_name = parts[0]
        quality = parts[1] if len(parts) > 1 else 'maj'
        
        if root_name in note_map:
            root_node = note_map[root_name]
            duration = max(0.1, end - start)
            
            # Determine intervals: Minor (3 semitones) vs Major (4 semitones)
            third = 3 if 'min' in quality else 4
            
            # Add Triad
            midi.addNote(0, 0, root_node, start, duration, 80)      
            midi.addNote(0, 0, root_node + third, start, duration, 65) 
            midi.addNote(0, 0, root_node + 7, start, duration, 65) 
            
    bin_data = BytesIO()
    midi.writeFile(bin_data)
    return bin_data.getvalue()

# --- STREAMLIT UI ---
st.title("🎵 Pro AI Chord Detector")
st.markdown("Advanced harmonic separation for high-accuracy transcription.")

uploaded_file = st.file_uploader("Upload Audio (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file:
    # Save original upload
    input_path = f"in_{uploaded_file.name}"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file)
    
    with st.spinner("🧠 Performing Harmonic Separation & Analysis..."):
        try:
            chords, raw_audio, sr = process_and_recognize(input_path)
            
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Spectral Analysis")
                # Create Spectrogram
                fig, ax = plt.subplots(figsize=(12, 5))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(raw_audio)), ref=np.max)
                img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax, cmap='viridis')
                plt.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig)

            with col2:
                st.subheader("Chord Progression")
                df = pd.DataFrame(chords, columns=["Start", "End", "Chord"])
                st.dataframe(df, width='stretch', height=400)
                
                # MIDI Button
                midi_bytes = create_midi(chords)
                st.download_button(
                    label="⬇️ Download MIDI",
                    data=midi_bytes,
                    file_name="transcription.mid",
                    mime="audio/midi",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Analysis failed: {e}")
        
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)