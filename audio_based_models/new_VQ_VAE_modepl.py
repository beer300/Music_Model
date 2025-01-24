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
    def __init__(self, in_channels=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=(4, 4))),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self, latent_dim=200, num_embeddings=512, h=16, w=351):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.h = h
        self.w = w

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, num_embeddings, kernel_size=3, padding=1)
        )

    def forward(self, z, hard=True):
        logits = self.main(z)
        code_probs = F.gumbel_softmax(logits, tau=1.0, hard=hard, dim=1)
        return code_probs

class VQVAE(pl.LightningModule):
    def __init__(self, in_channels=1, embedding_dim=64, num_embeddings=512, num_restarts=3, sample_rate=48000):
        super().__init__()
        self.num_restarts = num_restarts
        self.sample_rate = sample_rate
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
        best_indices = None

        for _ in range(self.num_restarts):
            z_e = self.encoder(x)
            z_e_flattened = z_e.view(z_e.size(0), z_e.size(1), -1).permute(0, 2, 1)
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
                best_indices = encoding_indices

        return best_x_recon, best_indices

    def training_step(self, batch, batch_idx):
        x_recon = self.forward(batch)
        loss = F.mse_loss(x_recon, batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_recon = self.forward(batch)
        loss = F.mse_loss(x_recon, batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_validation_epoch_end(self):
        # Save reconstructed VQ-VAE songs as 30-second tracks
        output_dir = r'C:\Users\lukas\Music\vqvae_reconstructed'
        os.makedirs(output_dir, exist_ok=True)
        for i, batch in enumerate(self.trainer.val_dataloaders):
            x_recon = self.forward(batch)
            x_recon = x_recon.detach().cpu().numpy()
            for j, audio in enumerate(x_recon):
                audio = audio.flatten()
                audio = np.tile(audio, int(np.ceil(self.sample_rate * 30 / len(audio))))[:self.sample_rate * 30]
                audio = (audio * 32767).astype(np.int16)
                wav_path = os.path.join(output_dir, f"val_epoch{self.current_epoch}_batch{i}_sample{j}.wav")
                write(wav_path, self.sample_rate, audio)
                print(f"[INFO] Saved reconstructed VQ-VAE audio to {wav_path}")
                
vqvae_model = VQVAE.load_from_checkpoint(vqvae_checkpoint_path)
vqvae_model.freeze()
class GAN(pl.LightningModule):
    def __init__(self,pretrained_vqvae, latent_dim=200, lr=0.0002, sample_rate=48000, l2_penalty=0.001, num_restarts=3, d_updates=1, g_updates=1):
        super().__init__()
        self.save_hyperparameters()
        self.vqvae = pretrained_vqvae
        for param in self.vqvae.parameters():
            param.requires_grad = False
        self.vqvae = VQVAE(in_channels=1, num_restarts=num_restarts, sample_rate=sample_rate)
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator(in_channels=1)
        self.sample_rate = sample_rate
        self.l2_penalty = l2_penalty
        self.d_updates = d_updates
        self.g_updates = g_updates
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

        # Get real code indices and embeddings
        _, real_indices = self.vqvae(real_imgs)
        real_embeddings = self.vqvae.codebook(real_indices).permute(0, 2, 1).view(real_embeddings.size(0), self.vqvae.embedding_dim, self.vqvae.h, self.vqvae.w)

        # Train Discriminator
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, device=self.device)
        fake_code_probs = self.generator(z)
        fake_embeddings = torch.einsum('bnhw, nd -> bdhw', fake_code_probs, self.vqvae.codebook.weight)

        # Discriminator forward
        d_real = self.discriminator(real_embeddings.detach())
        d_fake = self.discriminator(fake_embeddings.detach())

        # Losses
        real_loss = self.adversarial_loss(d_real, torch.ones_like(d_real))
        fake_loss = self.adversarial_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()

        # Train Generator
        fake_embeddings = torch.einsum('bnhw, nd -> bdhw', fake_code_probs, self.vqvae.codebook.weight)
        d_fake = self.discriminator(fake_embeddings)
        g_loss = self.adversarial_loss(d_fake, torch.ones_like(d_fake))
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
        recon_loss = self.adversarial_loss(y_hat_recon, torch.ones_like(y_hat_recon)*0.9)
        fake_loss = self.adversarial_loss(y_hat_fake, torch.ones_like(y_hat_fake)*0.1)
        d_loss = 0.5 * (recon_loss + fake_loss) + self.l2_penalty * self.l2_regularization(self.discriminator)

        # Generator loss
        g_loss = self.adversarial_loss(y_hat_fake, torch.ones_like(y_hat_fake)) + self.l2_penalty * self.l2_regularization(self.generator)

        # VQ-VAE loss
        vqvae_loss = F.mse_loss(x_recon, batch[:batch_length])

        # Log metrics
        self.log('val_d_loss', d_loss, prog_bar=True)
        self.log('val_g_loss', g_loss, prog_bar=True)
        self.log('val_vqvae_loss', vqvae_loss, prog_bar=True)

        return {'val_loss': d_loss + g_loss + vqvae_loss}

    def on_validation_epoch_end(self):
        # Generate unique batches for the final audio
        num_batches = 8  # More batches for complex beats
        combined_audio = []
        z = torch.randn(1, self.hparams.latent_dim, device=self.device)
        fake_code_probs = self.generator(z)
        fake_embeddings = torch.einsum('bnhw, nd -> bdhw', fake_code_probs, self.vqvae.codebook.weight)
        spec = self.vqvae.decoder(fake_embeddings)
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

    # Pretrain VQ-VAE
    vqvae_model = VQVAE(in_channels=1, num_restarts=3, sample_rate=SAMPLE_RATE)
    vqvae_checkpoint_callback = ModelCheckpoint(
        dirpath="vqvae_checkpoints",
        filename="vqvae-{epoch:02d}-{val_loss:.2f}",
        save_top_k=-1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        every_n_epochs=1
    )
    vqvae_trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[vqvae_checkpoint_callback]
    )
    vqvae_trainer.fit(vqvae_model, train_dataloader, val_dataloader)

    # Train GAN
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