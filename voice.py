
# Import the required modules
import pyaudio
import wave
import os
import numpy as np
import torch
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from google.cloud import texttospeech

# Define some constants
CHUNK = 1024 # Number of frames per buffer
FORMAT = pyaudio.paInt16 # Audio format
CHANNELS = 1 # Number of channels
RATE = 16000 # Sampling rate
RECORD_SECONDS = 10 # Duration of recording
WAVE_FILE = "voice.wav" # Name of the output wave file
MODEL_PATH = "sv2tts" # Path to the SV2TTS model
TEXT = "Hello, this is my voice." # Text to read

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream for recording
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Start recording
print("Recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Stop recording
print("Done recording.")
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded voice as a wave file
wf = wave.open(WAVE_FILE, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(frames))
wf.close()

# Load the SV2TTS model
print("Loading model...")
encoder.load_model(MODEL_PATH)
synthesizer = Synthesizer(os.path.join(MODEL_PATH, "synthesizer"))
vocoder.load_model(os.path.join(MODEL_PATH, "vocoder"))

# Encode the voice as a voice embedding
print("Encoding voice...")
preprocessed_wav = encoder.preprocess_wav(WAVE_FILE)
original_wav, sampling_rate = librosa.load(WAVE_FILE)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
embed = encoder.embed_utterance(preprocessed_wav)

# Synthesize the text as a mel spectrogram
print("Synthesizing text...")
texts = [TEXT]
embeds = [embed]
mels = synthesizer.synthesize_spectrograms(texts, embeds)
mel = np.concatenate(mels, axis=1)

# Generate the speech as a waveform
print("Generating speech...")
waveform = vocoder.infer_waveform(mel)

# Initialize Google Cloud Text-to-Speech client
client = texttospeech.TextToSpeechClient()

# Set the text input to be synthesized
synthesis_input = texttospeech.SynthesisInput(text=TEXT)

# Build the voice request, select the language code and the ssml
# voice gender
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config)

# The response's audio_content is binary.
with open("output.mp3", "wb") as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')
