import pyaudio
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, fcluster
import time


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


# Function to save MFCC data to file
def save_mfcc_to_file(mfcc, file_path):
    np.save(file_path, mfcc)


# Function to perform hierarchical clustering
def hierarchical_clustering(mfcc_list, threshold=100):
    distances = np.zeros((len(mfcc_list), len(mfcc_list)))

    # Calculate pairwise DTW distances
    for i in range(len(mfcc_list)):
        for j in range(i + 1, len(mfcc_list)):
            distances[i, j] = calculate_dtw_distance(mfcc_list[i], mfcc_list[j])

    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method='average')
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')

    # Group similar MFCCs together
    grouped_mfccs = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in grouped_mfccs:
            grouped_mfccs[cluster_id] = []
        grouped_mfccs[cluster_id].append(f"detected_sound_{i}.npy")

    return grouped_mfccs


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

# Store MFCCs in a list
mfcc_list = []

# Set the duration to run the code (in seconds)
duration = 20
start_time = time.time()

while time.time() - start_time < duration:
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
        # Save MFCC data for the detected sound or speaker
        save_mfcc_to_file(current_mfcc, f"detected_sound_{len(mfcc_list)}.npy")
        mfcc_list.append(current_mfcc)
        print("Different speaker or background noise. MFCC data saved.")

        # Perform hierarchical clustering and group similar MFCCs
        clustered_mfccs = hierarchical_clustering(mfcc_list)

        # Print the grouped MFCCs
        print("Grouped MFCCs:")
        for cluster_id, mfcc_files in clustered_mfccs.items():
            print(f"Cluster {cluster_id}: {', '.join(mfcc_files)}")

# Close the stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()
