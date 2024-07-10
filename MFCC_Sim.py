import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the first audio file
y1, sr1 = librosa.load(r"D:\Dataset\Music\Train\01 - Saturday Saturday - DownloadMing.SE\B\01 - Saturday Saturday - DownloadMing.SE_B.mp3")
mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)

# Load the second audio file
y2, sr2 = librosa.load(r"D:\Dataset\Music\Train\01 - Saturday Saturday - DownloadMing.SE\C\01 - Saturday Saturday - DownloadMing.SE_C.mp3")
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

# Make sure the dimensions match
min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
mfcc1 = mfcc1[:, :min_frames]
mfcc2 = mfcc2[:, :min_frames]

# Reshape the MFCCs for cosine similarity calculation
mfcc1_flat = mfcc1.flatten().reshape(1, -1)
mfcc2_flat = mfcc2.flatten().reshape(1, -1)

# Calculate cosine similarity
similarity = cosine_similarity(mfcc1_flat, mfcc2_flat)

print(f"Cosine Similarity: {similarity[0][0]}")
