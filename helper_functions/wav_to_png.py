import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
SPECTROGRAM_DPI = 120
DEFAULT_HOP_LENGTH = 1024

class audio():
    def __init__(self, filepath_, hop_length=DEFAULT_HOP_LENGTH):
        self.hop_length = hop_length
        self.waveform, self.sample_rate = librosa.load(filepath_, sr=None)
        self.duration = librosa.get_duration(y=self.waveform, sr=self.sample_rate)
        print(f"Loaded audio file. Duration: {self.duration:.2f} seconds, Sample rate: {self.sample_rate} Hz")

    def plot_spectrogram(self) -> None:
        S = librosa.stft(self.waveform, hop_length=self.hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        plt.figure(figsize=(20, 10))
        librosa.display.specshow(S_db, sr=self.sample_rate, hop_length=self.hop_length, x_axis='time', y_axis='log')
        plt.axis('off')
        plt.tight_layout()

    def write_disk_spectrogram(self, path, dpi=SPECTROGRAM_DPI) -> None:
        self.plot_spectrogram()
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

input_folder = "C:\\Users\\lukas\\Music\\generated_nearest"
output_folder = "C:\\Users\\lukas\\Music\\generated_nearest_spectrograms"

for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)
    sound = audio(input_path)
    
    
    chunk_name = f"{os.path.splitext(file_name)[0]}.png"
    output_path = os.path.join(output_folder, chunk_name)
    sound.write_disk_spectrogram(output_path, dpi=SPECTROGRAM_DPI)
    print(f"Spectrogram saved at {output_path}")