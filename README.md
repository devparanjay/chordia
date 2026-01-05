# Chordia

## Installation

### Requirements

1. uv
2. python3.11 (virtualenv using pyenv)
3. ./latest/requirements.txt
4. wget

### Installation

1. `pyenv virtualenv 3.11 chordia-venv`
2. `pyenv activate chordia-venv`
3. `uv pip install -R ./latest/requirements.txt`

### Usage

1. `streamlit run ./latest/app_v10.py`
   Replace `app_v10.py` with the latest version available.
2. Once the app is running int he browser, upload a MP3 or WAV file.
3. It will take some time to download the models and other dependecies the first time you run it.
4. Once you click the "Analyze" button, Chordia will produce a spectogram.
5. Once the spectogram of the file is visible, select the primary instrument in the music/song for transcription/detection and click "Analyze Instrument".
6. Wait for a few seconds; might take longer in slower environments. Keep an eye on your terminal for any errors or issues.
7. Once the analysis is done, Chordia will produce a Playback Guide with Letter Notes in order with their timestamps.
8. There's also options to download the Letter Notes as TXT and for downloading a MIDI file based on Chordia's analysis of the uploaded file.

### TODO

* [ ] Improve detection of notes in complicated modern music.
* [ ] Improve the analysis for songs that also have vocals over the music (maybe via voice and music isolation).
* [ ] Add feature to view and download the sheet music based on Chordia's analysis of the file.
* [ ] Add more features to ease and aid user's journey of learning their music and instrument.
* [ ] Add further classification for different instruments with more claristy and fine-tuning.
* [ ] Improve the format of the Playback Guide to be more useful for complicated music.
