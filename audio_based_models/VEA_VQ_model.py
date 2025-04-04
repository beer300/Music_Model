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
BATCH_SIZE = 16
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # (batch_size, 16, 64, 1407)
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),                           # (batch_size, 16, 32, 703)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),           # (batch_size, 32, 32, 703)
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),                           # (batch_size, 32, 16, 351)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),           # (batch_size, 64, 16, 351)
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),                           # (batch_size, 64, 8, 175)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),          # (batch_size, 128, 8, 175)
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),                           # (batch_size, 128, 4, 87)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),         # (batch_size, 256, 4, 87)
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),                           # (batch_size, 256, 2, 43)

            # Fully connected layers
            nn.Flatten(),                                                   # Flatten for the dense layers
            nn.Linear(256 * 2 * 43, 512),                                   # Additional fully connected layer
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),                                            # Fully connected layer
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),                                             # Fully connected layer
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),                                               # Output layer
            nn.Sigmoid()                                                    # Probability output
        )

    def forward(self, x):
        # Forward pass through the model
        out = self.model(x)
        
        return out
    
class Generator(nn.Module):
    def __init__(self, latent_dim=200, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=2, batch_first=True)
        self.cov = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Upsample(size=(64, 1407), mode='nearest'),
            
        )
        self.final = nn.Sequential(
            
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        # Input shape: (batch_size, latent_dim)
        #print(f"z in: {z.shape}")
        z = z.unsqueeze(1)  # Add sequence dimension for LSTM (batch_size, seq_len, latent_dim)
        #print(f"z after unsqueeze: {z.shape}")
        lstm_out, _ = self.lstm(z)  # Output: (batch_size, seq_len, hidden_dim)
        #print(f"lstm_out: {lstm_out.shape}")
        lstm_out = lstm_out[:, -1, :]  # Use the last time step
        #print(f"lstm_out after slicing: {lstm_out.shape}")
        lstm_out = lstm_out.view(lstm_out.size(0), -1, 1, 1)  # Reshape for ConvTranspose2d
        #print(f"lstm_out after view: {lstm_out.shape}")
        x = self.main(lstm_out)
        #print(f"Generator output shape: {x.shape}")
        x = self.final(x)
        #print(f"Generator Final shape: {x.shape}")
        return x

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=64, num_embeddings=512, num_restarts=3):
        super().__init__()
        self.num_restarts = num_restarts
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(64, 1407), mode='nearest'),
            nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        best_loss = float('inf')
        best_x_recon = None

        for _ in range(self.num_restarts):
            z_e = self.encoder(x)
            z_e_flattened = z_e.view(z_e.size(0), z_e.size(1), -1)
            z_e_flattened = z_e_flattened.permute(0, 2, 1)
            distances = (torch.sum(z_e_flattened**2, dim=2, keepdim=True) 
                        + torch.sum(self.codebook.weight**2, dim=1)
                        - 2 * torch.matmul(z_e_flattened, self.codebook.weight.t()))
            encoding_indices = torch.argmin(distances, dim=-1)
            z_q = self.codebook(encoding_indices).permute(0, 2, 1).view(z_e.shape)
            x_recon = self.decoder(z_q)
            loss = F.mse_loss(x_recon, x)

            if loss < best_loss:
                best_loss = loss
                best_x_recon = x_recon

        return best_x_recon

class GAN(pl.LightningModule):
    def __init__(self, latent_dim=200, lr=0.0002, sample_rate=48000, l2_penalty=0.01, num_restarts=3):
        super().__init__()
        self.save_hyperparameters()
        self.vqvae = VQVAE(in_channels=1, num_restarts=num_restarts)
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator(in_channels=1)
        self.sample_rate = sample_rate
        self.l2_penalty = l2_penalty
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def l2_regularization(self, model):
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += torch.sum(param ** 2)
        return l2_loss

    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        real_imgs = batch

        # VQ-VAE encoding and decoding
        x_recon = self.vqvae(real_imgs)

        # Generate unique latent vectors for each sample
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, device=self.device)
        fake_imgs = self(z).detach()

        # Train Discriminator
        y_hat_recon = self.discriminator(x_recon)
        y_hat_fake = self.discriminator(fake_imgs)
        recon_loss = self.adversarial_loss(y_hat_recon, torch.ones_like(y_hat_recon))
        fake_loss = self.adversarial_loss(y_hat_fake, torch.zeros_like(y_hat_fake))
        d_loss = 0.5 * (recon_loss + fake_loss) + self.l2_penalty * self.l2_regularization(self.discriminator)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()

        # Train Generator
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, device=self.device)
        fake_imgs = self(z)
        y_hat = self.discriminator(fake_imgs)
        g_loss = self.adversarial_loss(y_hat, torch.ones_like(y_hat)) + self.l2_penalty * self.l2_regularization(self.generator)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()

        self.log('d_loss', d_loss, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # Generate latent vectors similar to training step
        batch_length = torch.randint(4, 8, (1,)).item()
        z = torch.randn(batch_length, self.hparams.latent_dim, device=self.device)

        # Generate fake images
        fake_imgs = self.generator(z)

        # Scale height (amplitude) dynamically
        scale_factor = torch.rand(1).item() * 2 + 0.5
        fake_imgs = fake_imgs * scale_factor

        # Use actual batch size for real images
        x_recon = self.vqvae(batch[:batch_length])
        y_hat_recon = self.discriminator(x_recon)
        y_hat_fake = self.discriminator(fake_imgs)

        # Calculate losses
        recon_loss = self.adversarial_loss(y_hat_recon, torch.ones_like(y_hat_recon))
        fake_loss = self.adversarial_loss(y_hat_fake, torch.zeros_like(y_hat_fake))
        d_loss = 0.5 * (recon_loss + fake_loss) + self.l2_penalty * self.l2_regularization(self.discriminator)

        # Generator loss
        g_loss = self.adversarial_loss(y_hat_fake, torch.ones_like(y_hat_fake)) + self.l2_penalty * self.l2_regularization(self.generator)

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
        combined_audio = np.concatenate(combined_audio)

        # Normalize and scale to fit the total duration
        duration = 30  # seconds
        total_samples = self.sample_rate * duration
        combined_audio = np.tile(combined_audio, int(np.ceil(total_samples / len(combined_audio))))[:total_samples]
        combined_audio = (combined_audio * 32767).astype(np.int16)

        # Save WAV file
        wav_path = os.path.join(r'C:\Users\lukas\Music\generated_nearest', f"epoch{self.current_epoch}_lstm_beats.wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        write(wav_path, self.sample_rate, combined_audio)

        print(f"[INFO] Saved audio with unique LSTM beats to {wav_path}")

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
    AUDIO_DIR = r"C:\Users\lukas\Music\youtube_playlist_chopped"
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