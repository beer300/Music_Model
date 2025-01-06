import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchaudio   
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os
import dataset_processing as dp 
BATCH_SIZE = 8
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
      

        out = self.model(x)
      

        return out.view(-1, 1)
    
class AudioImpactLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude_scale = nn.Parameter(torch.ones(1))
        self.dynamic_range = nn.Parameter(torch.ones(1) * 0.8)
        self.bass_boost = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Ensure everything is on the same device and dtype
        device = x.device
        dtype = x.dtype
        
        # Apply dynamic range compression
        magnitude = torch.abs(x)
        threshold = torch.tensor(0.3, device=device, dtype=dtype)
        ratio = 1.0 - torch.clamp(self.dynamic_range, 0.0, 1.0)
        
        # Handle compression using where
        compressed = torch.where(
            magnitude > threshold,
            threshold + (x - threshold) * ratio.to(device),
            x
        )
        
        # Apply amplitude scaling
        processed = compressed * torch.clamp(self.amplitude_scale, 0.1, 5.0).to(device)
        
        # Apply bass emphasis using frequency domain transform
        if self.bass_boost.item() > 0:
            # Calculate boost factor before FFT
            boost_factor = 1.0 + torch.clamp(self.bass_boost, 0.0, 1.0).to(device)
            
            # Convert to float32 for FFT while keeping device
            processed_float = processed.float()
            fft = torch.fft.rfft(processed_float, dim=-1)
            freq_bins = fft.shape[-1]
            cutoff = int(freq_bins * 0.1)
            
            # Apply boost to low frequencies
            fft[..., :cutoff] *= boost_factor
            
            # Convert back to time domain and original dtype
            processed = torch.fft.irfft(fft, n=processed.shape[-1], dim=-1).to(dtype)
            
        return torch.tanh(processed)  # Ensure output is in [-1, 1]

class Generator(nn.Module):
    def __init__(self, latent_dim=200, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=2, batch_first=True)

        # Convolutional layers for upsampling
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=(1, 7), mode='nearest'),
            nn.Conv2d(hidden_dim, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(64, 32, kernel_size=(1,22), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=3, mode='nearest'),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),


            nn.Tanh()
        )
        
        # Add impact control layer before final output
        self.impact_layer = AudioImpactLayer()

    def forward(self, z):
        # Input shape: (batch_size, latent_dim)
        z = z.unsqueeze(1)  # Add sequence dimension for LSTM (batch_size, seq_len, latent_dim)
        lstm_out, _ = self.lstm(z)  # Output: (batch_size, seq_len, hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # Use the last time step
        lstm_out = lstm_out.view(lstm_out.size(0), -1, 1, 1)  # Reshape for ConvTranspose2d
        x = self.main(lstm_out)
        x = self.impact_layer(x)  # Apply impact control
        
        return x

class GAN(pl.LightningModule):
    def __init__(self, latent_dim=200, lr=0.0002, sample_rate=48000):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator(in_channels=1)
        self.sample_rate = sample_rate
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        real_imgs = batch

        # Generate unique latent vectors for each sample
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, device=self.device)
        fake_imgs = self(z).detach()

        # Train Discriminator
        y_hat_real = self.discriminator(real_imgs)
        y_hat_fake = self.discriminator(fake_imgs)
        real_loss = self.adversarial_loss(y_hat_real, torch.ones_like(y_hat_real))
        fake_loss = self.adversarial_loss(y_hat_fake, torch.zeros_like(y_hat_fake))
        d_loss = 0.5 * (real_loss + fake_loss)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()

        # Train Generator
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, device=self.device)
        fake_imgs = self(z)
        y_hat = self.discriminator(fake_imgs)
        g_loss = self.adversarial_loss(y_hat, torch.ones_like(y_hat))
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()

        self.log('d_loss', d_loss, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # Generate latent vectors similar to training step
        batch_length = torch.randint(4, 8, (1,)).item()
        # Changed: Remove extra dimensions to match training
        z = torch.randn(batch_length, self.hparams.latent_dim, device=self.device)

        # Generate fake images
        fake_imgs = self(z)

        # Scale height (amplitude) dynamically
        scale_factor = torch.rand(1).item() * 2 + 0.5
        fake_imgs = fake_imgs * scale_factor

        # Use actual batch size for real images
        y_hat_real = self.discriminator(batch[:batch_length])
        y_hat_fake = self.discriminator(fake_imgs)

        # Calculate losses
        real_loss = self.adversarial_loss(y_hat_real, torch.ones_like(y_hat_real))
        fake_loss = self.adversarial_loss(y_hat_fake, torch.zeros_like(y_hat_fake))
        d_loss = 0.5 * (real_loss + fake_loss)

        # Generator loss
        g_loss = self.adversarial_loss(y_hat_fake, torch.ones_like(y_hat_fake))

        # Log metrics
        self.log('val_d_loss', d_loss, prog_bar=True)
        self.log('val_g_loss', g_loss, prog_bar=True)

        return {'val_loss': d_loss + g_loss}

    def on_validation_epoch_end(self):
        # Vary impact settings for different parts
        impact_variations = [
            {'amplitude': 1.5, 'dynamic_range': 0.6, 'bass_boost': 0.5},  # Strong impact
            {'amplitude': 0.8, 'dynamic_range': 0.9, 'bass_boost': 0.2},  # Subtle impact
            {'amplitude': 1.2, 'dynamic_range': 0.7, 'bass_boost': 0.4},  # Balanced impact
        ]
        
        num_batches = 8
        combined_audio = []

        for batch_idx in range(num_batches):
            # Cycle through impact variations
            impact = impact_variations[batch_idx % len(impact_variations)]
            self.set_impact(**impact)
            
            z = torch.randn(1, self.hparams.latent_dim, device=self.device)
            spec = self.generator(z)[0, 0].detach().cpu().numpy()
            
            # Normalize and ensure non-repetition
            audio_data = spec.flatten()
            audio_data = np.interp(audio_data, (audio_data.min(), audio_data.max()), [-1, 1])

            # Add rhythmic variations using sine modulation and beat patterns
            rhythm_factor = (batch_idx % 4 + 1) * 0.25  # Unique rhythm factor
            audio_data = np.sin(audio_data * rhythm_factor * np.pi)  # Modulate with sine for beats

            # Append to the final audio stream
            combined_audio.append(audio_data)

        # Concatenate all unique parts
        combined_audio = np.concatenate(combined_audio)

        # Normalize and scale to fit the total duration
        duration = 30  # seconds
        total_samples = self.sample_rate * duration
        combined_audio = np.tile(combined_audio, int(np.ceil(total_samples / len(combined_audio))))[:total_samples]
        combined_audio = (combined_audio * 32767).astype(np.int16)

        # Save WAV file
        wav_path = os.path.join(r'C:\Users\Lukasz\Music\generated_nearest', f"epoch{self.current_epoch}_lstm_beats.wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        write(wav_path, self.sample_rate, combined_audio)

        print(f"[INFO] Saved audio with unique LSTM beats to {wav_path}")

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_d, opt_g]
    
    def set_impact(self, amplitude=None, dynamic_range=None, bass_boost=None):
        """Update impact parameters"""
        if amplitude is not None:
            self.generator.impact_layer.amplitude_scale.data = torch.tensor([amplitude])
        if dynamic_range is not None:
            self.generator.impact_layer.dynamic_range.data = torch.tensor([dynamic_range])
        if bass_boost is not None:
            self.generator.impact_layer.bass_boost.data = torch.tensor([bass_boost])
    
    def get_impact_settings(self):
        """Get current impact settings"""
        return {
            'amplitude': self.generator.impact_layer.amplitude_scale.item(),
            'dynamic_range': self.generator.impact_layer.dynamic_range.item(),
            'bass_boost': self.generator.impact_layer.bass_boost.item()
        }


if __name__ == "__main__":
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.manual_seed(42)
    AUDIO_DIR = r"C:\Users\Lukasz\Music\chopped_30_piano"
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 1440000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=1024,
        n_mels=64
    )
    full_dataset = dp.AudioDataset(
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = GAN(
        latent_dim=200,
        lr=0.001,
        sample_rate=48000
    )
    
    # Set initial impact settings
    model.set_impact(
        amplitude=1.2,      # Slightly boosted overall volume
        dynamic_range=0.75, # Moderate compression
        bass_boost=0.35     # Moderate bass emphasis
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="gan-{epoch:02d}-{val_g_loss:.2f}",
        save_top_k=-1,
        verbose=True,
        monitor="val_g_loss",
        mode="min",
        save_weights_only=True,
        every_n_epochs=1
    )
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )