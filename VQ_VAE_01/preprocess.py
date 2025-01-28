import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=16000, target_duration=30,
                 transform=None, max_normalize=True, chunk_length=None):
        """
        Args:
            root_dir (str): Directory with WAV files
            target_sample_rate (int): Target sample rate (default: 16000 = 16kHz)
            target_duration (int): Expected duration in seconds (30)
            transform (callable): Optional transform for spectrograms
            max_normalize (bool): Normalize audio to [-1, 1] range
            chunk_length (int): Split into chunks of this duration (seconds)
        """
        self.root_dir = Path(root_dir)
        self.file_list = list(self.root_dir.glob("*.wav"))
        self.target_sample_rate = target_sample_rate
        self.target_length = target_duration * target_sample_rate
        self.chunk_length = chunk_length * target_sample_rate if chunk_length else None
        self.transform = transform
        self.max_normalize = max_normalize
        self.resampler = None

    def __len__(self):
        return len(self.file_list) * (self.target_length // self.chunk_length if self.chunk_length else 1)

    def _process_audio(self, waveform):
        # Resample if needed
        if self.resampler and waveform.shape[1] != self.target_length:
            waveform = self.resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad/Truncate to exact duration
        if waveform.shape[1] != self.target_length:
            if waveform.shape[1] > self.target_length:
                # Center truncation for full-length audio
                start = (waveform.shape[1] - self.target_length) // 2
                waveform = waveform[:, start:start+self.target_length]
            else:
                # Pad with zeros
                pad_amount = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        
        # Normalize to [-1, 1]
        if self.max_normalize:
            max_val = torch.max(torch.abs(waveform)) + 1e-9
            waveform = waveform / max_val
        
        return waveform

    def __getitem__(self, idx):
        # Handle chunking
        if self.chunk_length:
            file_idx = idx // (self.target_length // self.chunk_length)
            chunk_idx = idx % (self.target_length // self.chunk_length)
        else:
            file_idx = idx
            chunk_idx = None

        file_path = self.file_list[file_idx]
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Initialize resampler if needed
        if sample_rate != self.target_sample_rate and not self.resampler:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
        
        # Process full waveform
        waveform = self._process_audio(waveform)
        
        # Split into chunks if specified
        if self.chunk_length:
            start = chunk_idx * self.chunk_length
            end = start + self.chunk_length
            waveform = waveform[:, start:end]
        
        # Apply transforms (e.g., spectrogram conversion)
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform

# Example 1: Full 30-second processing at 16kHz (480,000 samples)
full_length_loader = DataLoader(
    AudioDataset(
        root_dir="path/to/wav_files",
        target_sample_rate=16000,
        target_duration=30
    ),
    batch_size=8,  # Smaller batch size due to memory constraints
    shuffle=True,
    num_workers=4
)

# Example 2: Split into 3-second chunks (48,000 samples each)
chunked_loader = DataLoader(
    AudioDataset(
        root_dir="path/to/wav_files",
        target_sample_rate=16000,
        target_duration=30,
        chunk_length=3  # 3-second chunks
    ),
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Example 3: Spectrogram processing (30s audio → 2D "image")
spectrogram_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=2048,        # Larger window for better frequency resolution
        hop_length=512,     # Compromise between time and frequency resolution
        n_mels=128         # Higher mel bins for detailed spectral information
    ),
    torchaudio.transforms.AmplitudeToDB()
)

spec_loader = DataLoader(
    AudioDataset(
        root_dir="path/to/wav_files",
        target_sample_rate=16000,
        target_duration=30,
        transform=spectrogram_transform
    ),
    batch_size=16,
    shuffle=True,
    num_workers=4
)