import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from torch_mdct import IMDCT, MDCT, vorbis
SAMPLE_RATE = 16000
mdct = MDCT(win_length=1024, window_fn=vorbis, window_kwargs=None, center=True)
imdct = IMDCT(win_length=1024, window_fn=vorbis, window_kwargs=None, center=True)

def remove_channel_dim(tensor):
    """
    Removes the channel dimension from a tensor of shape [1, height, width]
    to [height, width].
    """
    return tensor.squeeze(0)

def remove_channel_dim_batched(tensor):
    """
    Removes the channel dimension from a tensor of shape [batch_size, 1, height, width]
    to [batch_size, height, width].
    """
    return tensor.squeeze(1)

def add_channel_dim(tensor):
    """
    Adds a channel dimension to a tensor of shape [batch_size, height, width]
    to [batch_size, 1, height, width].
    """
    return tensor.unsqueeze(1)

class AudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None, target_sample_rate=SAMPLE_RATE):
        self.audio_dir = audio_dir
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.transform = transform
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        if waveform.size(1) < self.target_sample_rate:
            padding = self.target_sample_rate - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        spectrogram = mdct(waveform)
        
        #squished_spectogram=remove_channel_dim(spectrogram)
        #print(f"squished", squished_spectogram.shape)
        return spectrogram

# Przykładowe użycie:
audio_dir = r"C:\Users\lukas\Music\youtube_playlist_chopped"
dataset = AudioDataset(audio_dir)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
# Iteracja przez batch'e
for batch in data_loader:
    spectrogram = batch
    print(f"Original shape: {spectrogram.shape}")
    print("MDCT coeff range:", spectrogram.min(), spectrogram.max())
    
    break
  # Should be ~[-1, 1]
# Konwersja pierwszego elementu z powrotem na plik audio
#first_waveform, first_sample_rate = dataset[0]
