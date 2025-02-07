import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from torch_mdct import IMDCT, MDCT, vorbis

mdct = MDCT(win_length=1024, window_fn=vorbis, window_kwargs=None, center=True)
imdct = IMDCT(win_length=1024, window_fn=vorbis, window_kwargs=None, center=True)

class AudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None, target_sample_rate=48000):
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
        
        
        
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        if waveform.size(1) < self.target_sample_rate:
            padding = self.target_sample_rate - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        spectrogram = mdct(waveform)
        return spectrogram

# Przykładowe użycie:
audio_dir = r"C:\Users\lukas\Music\youtube_playlist_chopped"
dataset = AudioDataset(audio_dir)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
# Iteracja przez batch'e
for batch in data_loader:
    spectrogram = batch
    print(spectrogram.shape)
    break

# Konwersja pierwszego elementu z powrotem na plik audio
#first_waveform, first_sample_rate = dataset[0]
#torchaudio.save(r"C:\Users\lukas\Music\test\reconstructeed_audio.wav", first_waveform, first_sample_rate)
