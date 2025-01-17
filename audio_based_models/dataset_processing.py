import os
import torch
from torch.utils.data import Dataset
import torchaudio

class AudioDataset(Dataset):
    def __init__(self,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        
        # Move resampler to the correct device
        self.resampler = None

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, self.audio_files[index])
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            if self.resampler is None:
                self.resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = signal.to(self.device)  # Move signal to same device as resampler
            signal = self.resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


if __name__ == "__main__":
    AUDIO_DIR = r"C:\Users\Lukasz\Music\chopped_30"
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 1440000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    dataset = AudioDataset(
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )
    
    print(f"There are {len(dataset)} samples in the dataset.")
    print(f"Sample shape: {dataset[0].shape}")
