import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load(r"D:\Dataset\Music\Train\01 - Abhi Abhi (Duet) Hum To Haare (Abhi Abhi)-(MyMp3Singer.com)\B\01 - Abhi Abhi (Duet) Hum To Haare (Abhi Abhi)-(MyMp3Singer.com)_B.mp3")

# Compute the MFCCs
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Visualize the MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCCs')
plt.show()
