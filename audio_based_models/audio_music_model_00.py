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
    
class Generator(nn.Module):
    def __init__(self, latent_dim=200):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(latent_dim//2, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(64, 32, kernel_size=(1,22), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=(1,3), mode='bilinear'),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),





            
        )
        self.final = nn.Sequential(
            
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, z):
        print(f"z in", z.shape)
        z = z.view(-1, self.latent_dim//2, 1, 7)
        print(f"z out", z.shape)
        x = self.main(z)
        print(f"x in", x.shape)
        x = self.final(x)
        print(f"x out", x.shape)
        
       
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

        # Generate unique latent vectors with varying batch lengths
        batch_length = torch.randint(4, 8, (1,)).item()  # Random batch length (e.g., 4 to 8)
        z = torch.randn(batch_length, self.hparams.latent_dim, 1, 7, device=self.device)
        fake_imgs = self(z).detach()

        # Train Discriminator
        y_hat_real = self.discriminator(real_imgs[:batch_length])
        y_hat_fake = self.discriminator(fake_imgs)
        real_loss = self.adversarial_loss(y_hat_real, torch.ones_like(y_hat_real))
        fake_loss = self.adversarial_loss(y_hat_fake, torch.zeros_like(y_hat_fake))
        d_loss = 0.5 * (real_loss + fake_loss)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()

        # Train Generator
        z = torch.randn(batch_length, self.hparams.latent_dim, 1, 7, device=self.device)
        fake_imgs = self(z)
        y_hat = self.discriminator(fake_imgs)
        g_loss = self.adversarial_loss(y_hat, torch.ones_like(y_hat))
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()

        self.log('d_loss', d_loss, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # Generate unique latent vectors for validation with different lengths and heights
        batch_length = torch.randint(4, 8, (1,)).item()  # Random batch length
        z = torch.randn(batch_length, self.hparams.latent_dim, 1, 7, device=self.device)

        fake_imgs = self(z)
        # Scale height (amplitude) dynamically
        scale_factor = torch.rand(1).item() * 2 + 0.5  # Random scaling between 0.5 and 2.5
        fake_imgs = fake_imgs * scale_factor

        y_hat_real = self.discriminator(batch[:batch_length])
        y_hat_fake = self.discriminator(fake_imgs)
        real_loss = self.adversarial_loss(y_hat_real, torch.ones_like(y_hat_real))
        fake_loss = self.adversarial_loss(y_hat_fake, torch.zeros_like(y_hat_fake))
        d_loss = 0.5 * (real_loss + fake_loss)

        y_hat = self.discriminator(fake_imgs)
        g_loss = self.adversarial_loss(y_hat, torch.ones_like(y_hat))

        self.log('val_d_loss', d_loss, prog_bar=True)
        self.log('val_g_loss', g_loss, prog_bar=True)

    def on_validation_epoch_end(self):
        # Generate a sequence of batches with at least three unique heights
        num_batches = 5
        unique_scales = torch.randperm(10)[:3] / 3.0 + 0.5  # At least 3 unique scales (0.5 to ~3.5)
        combined_audio = []

        for batch_idx in range(num_batches):
            batch_length = torch.randint(4, 8, (1,)).item()
            z = torch.randn(batch_length, self.hparams.latent_dim, 1, 7, device=self.device)
            spec = self.generator(z)[0, 0].detach().cpu().numpy()

            # Use a unique scale for this batch
            scale_factor = unique_scales[batch_idx % 3].item()  # Cycle through the 3 unique scales
            spec = spec * scale_factor

            # Flatten and normalize for audio
            audio_data = spec.flatten()
            audio_data = np.interp(audio_data, (audio_data.min(), audio_data.max()), [-1, 1])
            combined_audio.append(audio_data)

        # Concatenate all batches into one audio stream
        combined_audio = np.concatenate(combined_audio)

        # Rescale audio to fit the total duration
        duration = 30  # seconds
        total_samples = self.sample_rate * duration
        combined_audio = np.tile(combined_audio, int(np.ceil(total_samples / len(combined_audio))))[:total_samples]
        combined_audio = (combined_audio * 32767).astype(np.int16)

        # Save WAV file
        wav_path = os.path.join(r'C:\Users\Lukasz\Music\generated', f"epoch{self.current_epoch}_unique_heights.wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        write(wav_path, self.sample_rate, combined_audio)

        print(f"[INFO] Saved audio with unique heights to {wav_path}")

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
    AUDIO_DIR = r"C:\Users\Lukasz\Music\chopped_30"
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 1440000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=1024,
        n_mels=64
    ).to(device)  # Move mel_spectrogram to device
    
    full_dataset = dp.AudioDataset(
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )
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