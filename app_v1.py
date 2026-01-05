import streamlit as st
import autochord
import os
import pandas as pd

st.set_page_config(page_title="Chordia", layout="wide")
st.title("🎼 Advanced Chord Transcription")

uploaded_file = st.file_uploader("Upload MP3", type=["mp3", "wav"])

if uploaded_file:
    # Save the file temporarily
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("AI is analyzing harmonic structures..."):
        # AutoChord returns: [(start, end, chord), ...]
        chords = autochord.recognize(file_path)
        
        # Format results into a clean Table
        df = pd.DataFrame(chords, columns=["Start (s)", "End (s)", "Chord"])
        
    st.success("Transcription Complete")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df, height=400)
    with col2:
        st.audio(uploaded_file)
        # Visual representation (Simplified text timeline)
        progression = " ➔ ".join([c[2] for c in chords if c[2] != 'N'][:20])
        st.info(f"**Key Progression Preview:** {progression}...")

    os.remove(file_path) # Cleanup