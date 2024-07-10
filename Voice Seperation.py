import pyaudio
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# Function to calculate MFCCs from audio frames
def calculate_mfcc(audio_frame, sr):
    # Convert the audio data to floating-point format
    audio_frame = audio_frame.astype(np.float32)

    mfcc = librosa.feature.mfcc(y=audio_frame, sr=sr, n_mfcc=13)
    return mfcc


# Function to perform DTW calculation
def calculate_dtw_distance(mfcc1, mfcc2):
    distance, _ = fastdtw(mfcc1.T, mfcc2.T, dist=euclidean)
    return distance


# Initialize PyAudio
p = pyaudio.PyAudio()

# Set up audio stream parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Open a stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Initialize reference MFCC with the first incoming audio data
reference_mfcc = calculate_mfcc(np.frombuffer(stream.read(CHUNK), dtype=np.int16), RATE)

while True:
    # Read audio data from the stream
    audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

    # Calculate MFCCs from the incoming audio data
    current_mfcc = calculate_mfcc(audio_data, RATE)

    # Calculate DTW distance between the current and reference MFCCs
    dtw_distance = calculate_dtw_distance(current_mfcc, reference_mfcc)

    # Update the reference MFCC with the current MFCC
    reference_mfcc = current_mfcc

    # Determine the speaker based on DTW distance and a threshold
    threshold = 100  # Adjust the threshold based on your application
    if dtw_distance < threshold:
        print("The speaker is speaking.")
    else:
        print("Different speaker or background noise.")

# Close the stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()
