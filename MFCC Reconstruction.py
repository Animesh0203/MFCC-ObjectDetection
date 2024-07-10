import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio file and extract MFCCs
audio, sr = librosa.load(r"D:\Dataset\Music\Train\01 - Saturday Saturday - DownloadMing.SE\C\01 - Saturday Saturday - DownloadMing.SE_C.mp3")
mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)

# Invert MFCCs to obtain the spectrogram
mel_spec = librosa.feature.inverse.mfcc_to_mel(mfccs)

# Invert the logarithmic operation
power_spec = np.exp(mel_spec)

# Invert the filterbank operation
spectrum = librosa.filters.mel(sr, n_fft=2048, n_mels=128).T @ power_spec

# Invert the Fourier transform
audio_reconstructed = librosa.griffinlim(spectrum, hop_length=512, win_length=2048)

# Save or play the reconstructed audio
librosa.output.write_wav('reconstructed_audio.wav', audio_reconstructed, sr)

# Plot the original and reconstructed audio signals
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
librosa.display.waveshow(audio, sr=sr)
plt.title('Original Audio')

plt.subplot(2, 1, 2)
librosa.display.waveshow(audio_reconstructed, sr=sr)
plt.title('Reconstructed Audio')

plt.tight_layout()
plt.show()