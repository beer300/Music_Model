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
from dataset_processing import add_channel_dim, remove_channel_dim
from torch_mdct import IMDCT, MDCT, vorbis
mdct = MDCT(win_length=1024, window_fn=vorbis, window_kwargs=None, center=True)
imdct = IMDCT(win_length=1024, window_fn=vorbis, window_kwargs=None, center=True)
BATCH_SIZE = 8



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: 1xHxW (typically 1x64x64 or 1x28x28)
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Additional convolutional layer for deeper features
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            # Feature aggregation
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Enhanced classification head
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        # Apply spectral normalization to convolutional layers
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                nn.utils.spectral_norm(layer)

    def forward(self, x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self, latent_dim=200, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=2, batch_first=True)

        # Convolutional layers for upsampling
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=(2, 7), mode='nearest'),
            nn.Conv2d(hidden_dim, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            # 2 ,7
            
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            #2, 14

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            #4, 28

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            #8, 56

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            #16, 112

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            #32, 224

            nn.ConvTranspose2d(64, 32, kernel_size=(3), stride=(2), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            #64, 224

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            #128, 448
            nn.Upsample(scale_factor=(2,3), mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            #256, 1344
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(size=(512, 2586), mode='nearest'),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            

          
        )

    def forward(self, z):
        # Input shape: (batch_size, latent_dim)
        z = z.unsqueeze(1)  # Add sequence dimension for LSTM (batch_size, seq_len, latent_dim)
        lstm_out, _ = self.lstm(z)  # Output: (batch_size, seq_len, hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # Use the last time step
        lstm_out = lstm_out.view(lstm_out.size(0), -1, 1, 1)  # Reshape for ConvTranspose2d
        x = self.main(lstm_out)
        
        return x

class GAN(pl.LightningModule):
    def __init__(self, latent_dim=200, lr=0.0002, sample_rate=44100):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
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
        # Generate unique batches for the final audio
        num_batches = 8  # More batches for complex beats
        combined_audio = []

        for batch_idx in range(num_batches):
            # Unique latent vector for each batch
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
        #combined_audio = np.concatenate(combined_audio)
        z = torch.randn(1, self.hparams.latent_dim, device=self.device)


        # Normalize and scale to fit the total duration
            # Normalize and scale to fit the total duration


        # Save a single generated waveform for debugging
        z = torch.randn(1, self.hparams.latent_dim, device=self.device)
        spec = self.generator(z)  # Shape: [batch_size, num_channels, num_samples]
        spec = spec[0]  # Extract the first waveform in the batch (shape: [num_channels, num_samples])
        spec_waveform = imdct(spec.cpu())
        print("MDCT coeff range:", spec.min(), spec.max())
        print(f"spec shape", spec.shape)
        print(f"spec_waveform shape", spec_waveform.shape)
        wav_path_diff_conversion = os.path.join(r'C:\Users\lukas\Music\generated_nearest', f"epoch{self.current_epoch}_reconstructed.wav")
        if os.path.exists(wav_path_diff_conversion):
            os.remove(wav_path_diff_conversion)
        torchaudio.save(wav_path_diff_conversion, spec_waveform, self.sample_rate)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_d, opt_g]


if __name__ == "__main__":
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.manual_seed(42)
    AUDIO_DIR = r"C:\Users\lukas\Music\youtube_playlist_chopped — kopia"
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 1440000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    full_dataset = dp.AudioDataset(
                            AUDIO_DIR,
                            )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = GAN(
        latent_dim=200,
        lr=0.0002,
        sample_rate=48000
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