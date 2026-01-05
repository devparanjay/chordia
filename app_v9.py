import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from basic_pitch.inference import predict_and_save
from mido import MidiFile
import torch

st.set_page_config(page_title="Chordia V9", layout="wide")

# --- GUITAR TAB LOGIC ---
def midi_to_tab(midi_note):
    """
    Translates a MIDI note to the most likely String and Fret.
    Standard Tuning: E2 (40), A2 (45), D3 (50), G3 (55), B3 (59), E4 (64)
    """
    # Define strings by their starting MIDI note
    strings = [
        {"name": "e (High)", "start": 64},
        {"name": "B", "start": 59},
        {"name": "G", "start": 55},
        {"name": "D", "start": 50},
        {"name": "A", "start": 45},
        {"name": "E (Low)", "start": 40},
    ]
    
    # Logic: Find the first string where the note can be played (fret 0-22)
    for s in strings:
        if midi_note >= s["start"]:
            fret = midi_note - s["start"]
            if 0 <= fret <= 22:
                return f"{s['name']} | Fret: {fret}"
    
    return "Out of Range"

def get_note_name(midi_number):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    return f"{notes[midi_number % 12]}{octave}"

# --- MIDI PARSER ---
def parse_midi_to_list(midi_path, instrument_type):
    mid = MidiFile(midi_path)
    note_events = []
    current_time = 0
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                data = {
                    "Timestamp (s)": round(current_time / 480.0, 2),
                    "Note": get_note_name(msg.note),
                }
                if instrument_type == "Guitar":
                    data["Guitar Tab"] = midi_to_tab(msg.note)
                note_events.append(data)
    return sorted(note_events, key=lambda x: x['Timestamp (s)'])

# --- UI ---
st.title("🎸 Chordia V9")
st.markdown("Convert any audio to **MIDI, Letter Notes, or Guitar Tabs**.")

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])

if uploaded_file:
    temp_audio = f"temp_{uploaded_file.name}"
    with open(temp_audio, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Visualizer (Spectrogram)
    y, sr = librosa.load(temp_audio)
    fig, ax = plt.subplots(figsize=(10, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    st.pyplot(fig)
    st.audio(uploaded_file)

    # User Selection
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        inst_mode = st.radio("Instrument Mode", ["Piano/Violin", "Guitar"])
    
    # Analyze Button
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("AI processing frequencies and tabulature..."):
            try:
                predict_and_save(
                    audio_path_list=[temp_audio],
                    output_directory=".",
                    save_midi=True,
                    sonify_midi=False,
                    save_model_outputs=False,
                    save_notes=True
                )
                
                midi_output = temp_audio.replace(".mp3", "_basic_pitch.mid").replace(".wav", "_basic_pitch.mid")
                st.session_state['midi_ready'] = midi_output
                st.session_state['note_list'] = parse_midi_to_list(midi_output, inst_mode)
                st.success("Analysis Complete!")
            except Exception as e:
                st.error(f"Analysis Error: {e}")

    # Results
    if 'note_list' in st.session_state:
        st.divider()
        st.subheader("📝 Transcription Results")
        
        df = pd.DataFrame(st.session_state['note_list'])
        st.dataframe(df, width='stretch', height=400)

        # Download Buttons
        btn1, btn2 = st.columns(2)
        with btn1:
            with open(st.session_state['midi_ready'], "rb") as f:
                st.download_button("⬇️ Download MIDI", f, "output.mid", width='stretch')
        with btn2:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Notes/Tabs (CSV)", csv, "tabs.csv", width='stretch')

    # Cleanup (Recommended to do on script exit or after download)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)