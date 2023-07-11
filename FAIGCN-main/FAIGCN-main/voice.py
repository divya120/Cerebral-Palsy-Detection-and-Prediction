import sounddevice as sd
import numpy as np
import wave

# Set the audio recording parameters
duration = 12.5 # recording duration in seconds
sample_rate = 44100  # sample rate in Hz
channels = 1  # number of audio channels (1 for mono, 2 for stereo)
filename = 'output.wav'  # name of the output WAV file

# Define the callback function for recording audio
def callback(indata, frames, time, status):
    pass  # no-op

# Open a new WAV file for writing
with wave.open(filename, 'w') as wavefile:
    # Set the WAV file parameters
    wavefile.setnchannels(channels)
    wavefile.setsampwidth(2)  # 2 bytes (16 bits)
    wavefile.setframerate(sample_rate)

    # Start recording audio
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        print(f"Recording audio for {duration} seconds...")

        # Record audio data and write it to the WAV file
        frames_to_record = int(duration * sample_rate)
        frames_recorded = 0
        while frames_recorded < frames_to_record:
            frames_remaining = frames_to_record - frames_recorded
            frames = min(frames_remaining, 2048)  # record in chunks of 2048 frames
            audio_data = sd.rec(frames, samplerate=sample_rate, channels=channels)
            wavefile.writeframesraw(audio_data.tobytes())
            frames_recorded += frames

    print(f"Recording complete. Audio saved as '{filename}'.")

import matplotlib.pyplot as plt
file_path = "/home/luna/Documents/FAIGCN-main/output.wav"
with wave.open(file_path, 'r') as wave_file:
    frames = wave_file.readframes(-1)
    signal = np.frombuffer(frames, dtype=np.int16)

# Create time array
frame_rate = wave_file.getframerate()
duration = wave_file.getnframes() / float(frame_rate)
time = np.linspace(0, duration, num=len(signal))

# Plot waveform
plt.figure(figsize=(10, 5))
plt.plot(time, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.show()
print("hello")
from pydub import AudioSegment
from pydub.playback import play

# Load the audio file
filename = "output.wav"
audio = AudioSegment.from_file(filename)

# Extract acoustic features
duration = audio.duration_seconds
frame_rate = audio.frame_rate
channels = audio.channels
samples = audio.get_array_of_samples()
print("Duration:",duration)
print("Frame Rate:",frame_rate)
print("Channels:",channels)
print("Sample Length:",len(samples))

# Define threshold values for each feature (you can adjust these based on your specific use case)
duration_threshold = 3     # Example threshold for duration in seconds
frame_rate_threshold = 8000 # Example threshold for frame rate
channels_threshold = 1     # Example threshold for number of channels
samples_threshold = 10000   # Example threshold for number of samples

# Check for voice distortion
is_distorted = False
if duration < duration_threshold or frame_rate < frame_rate_threshold or channels < channels_threshold or len(samples) < samples_threshold:
    is_distorted = True

# Print the result
if is_distorted:
    print("Voice distortion related to cerebral palsy detected.")
    val = True
else:
    print("Voice distortion related to cerebral palsy not detected.")
    val = False

# Import necessary libraries
import numpy as np
import pandas as pd
from pydub import AudioSegment
import tempfile
import io
import librosa

# Load and preprocess the audio data
# Assuming you have audio data in a suitable format (e.g., WAV files)
 # Replace with the actual path to your audio file
audio_file = 'output.wav'  # Replace with the actual path to your audio file
# Load the audio file using PyDub
audio = AudioSegment.from_file(audio_file)

# Play the audio to load it into memory
play(audio)

# Save the audio to a temporary file
with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
    audio.export(temp_file.name, format='wav')

    # Load the temporary file with librosa and extract the audio signal
    audio_signal, fs = librosa.load(temp_file.name, sr=None)

# Apply Short-Time Fourier Transform (STFT) to the audio signal
Zxx = librosa.stft(audio_signal, n_fft=1024, hop_length=512)  # Use librosa to perform STFT on audio signal

# Extract features from STFT
# Depending on the specific characteristics of voice distortion in cerebral palsy, you may need to extract specific features such as spectral characteristics, pitch, formants, etc.
# Example: Extract magnitude of STFT as a feature
magnitude = np.abs(Zxx)

# Perform analysis on the extracted features
# Depending on the specific requirements of your task, you can apply various analysis techniques such as statistical analysis, machine learning, etc. to determine voice distortion in cerebral palsy.
# Example: Compute statistical features (e.g., mean, median, standard deviation) from the magnitude of STFT
mean_magnitude = np.mean(magnitude, axis=0)
median_magnitude = np.median(magnitude, axis=0)
std_magnitude = np.std(magnitude, axis=0)
print("Mean Magnitude:",mean_magnitude)
print("Median Magnitude:",median_magnitude)
print("Standard Magnitude:",std_magnitude)

# Make decisions based on the analyzed features
# Depending on the specific criteria for determining voice distortion in cerebral palsy, you can set thresholds or criteria based on your analysis results to make decisions or predictions.
# Example: Set a threshold for mean magnitude to determine voice distortion
threshold = 8.6# Set a threshold based on your analysis
print("MMM:",np.max(mean_magnitude))
if np.max(mean_magnitude) > threshold:
    print("Voice distortion detected in cerebral palsy.")
    val_1 = True
else:
    print("No voice distortion detected in cerebral palsy.")
    val_1 = False

# if val_1==True: #and val_1==True:
#     print("Voice distortion related to cerebral palsy detected.")
# # elif val==True and val_1==False:
# #     print("Voice distortion related to cerebral palsy not detected.")
# # elif val==True or val_1==True:
# #     print("Voice distortion related to cerebral palsy detected.")
# else:
#     print("Voice distortion related to cerebral palsy not detected.")