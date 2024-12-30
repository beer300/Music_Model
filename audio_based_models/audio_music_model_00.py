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
        print(f"x in", x.shape)

        out = self.model(x)
        print(f"x out", x.shape)

        return out.view(-1, 1)
    
class Generator(nn.Module):
    def __init__(self,num_channels=1, latent_dim=200, hidden_dim=512, num_layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.sigmoid = nn.Sigmoid()
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

            nn.Upsample(scale_factor=(1,2), mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(64, 32, kernel_size=(1,22), stride=1, padding=1),

            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=3, mode='bilinear'),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Upsample(size=(16, 350), mode='bilinear'), 
            nn.Conv2d(16,1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Dropout(0.5),  
        )
        self.final = nn.Sequential(
            
            nn.LSTM(input_size=1, hidden_size=1024, batch_first=True),
            
        )
    def forward(self, z):
        print(f"z in", z.shape)
        z = z.view(-1, self.latent_dim // 2, 1, 7)
        print(f"z out", z.shape)
        x = self.main(z)
        print(f"x in", x.shape)
        x = x.view(x.size(0), -1)  # Flatten for LSTM input
        print(f"x view", x.shape)
        x = x.view(x.size(0), -1, 1).contiguous()  # TUTAJ ZMIANA
        print(f"x view", x.shape)
        x, _ = self.final(x)  # TUTAJ ZMIANA
        print(f"x final", x.shape)
        x = x[:, -1, :]  # Last time step output
        print(f"x out", x.shape)
        x = self.sigmoid(x)
        return x



class GAN(pl.LightningModule):
    def _write_disk_spectrogram(self, path, dpi=120):
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
    def __init__(self, latent_dim=200, lr=0.0002, sample_rate=48000):
        super().__init__()
        self.hparams.latent_dim = latent_dim
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator(in_channels=4)
        self.validation_z = torch.randn(1, self.hparams.latent_dim, 1, 7)
    def forward(self, z):
        return self.generator(z)
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        real_imgs = batch
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, 1, 7).type_as(real_imgs)
        fake_imgs = self(z).detach()
        y_hat_real = self.discriminator(real_imgs)
        y_hat_fake = self.discriminator(fake_imgs)
        real_loss = self.adversarial_loss(y_hat_real, torch.ones_like(y_hat_real))
        fake_loss = self.adversarial_loss(y_hat_fake, torch.zeros_like(y_hat_fake))
        d_loss = 0.5 * (real_loss + fake_loss)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, 1, 7).type_as(real_imgs)
        fake_imgs = self(z)
        y_hat = self.discriminator(fake_imgs)
        g_loss = self.adversarial_loss(y_hat, torch.ones_like(y_hat))
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.log('d_loss', d_loss, prog_bar=True, on_epoch=True)
        self.log('g_loss', g_loss, prog_bar=True, on_epoch=True)
    def validation_step(self, batch, batch_idx):
        real_imgs = batch
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, 1, 7).type_as(real_imgs)
        fake_imgs = self(z)
        y_hat_real = self.discriminator(real_imgs)
        y_hat_fake = self.discriminator(fake_imgs)
        real_loss = self.adversarial_loss(y_hat_real, torch.ones_like(y_hat_real))
        fake_loss = self.adversarial_loss(y_hat_fake, torch.zeros_like(y_hat_fake))
        d_loss = 0.5 * (real_loss + fake_loss)
        y_hat = self.discriminator(fake_imgs)
        g_loss = self.adversarial_loss(y_hat, torch.ones_like(y_hat))
        self.log('val_d_loss', d_loss, prog_bar=True, on_epoch=True)
        self.log('val_g_loss', g_loss, prog_bar=True, on_epoch=True)
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_d, opt_g]
    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.main[1].weight)
        spec = self.generator(z)[0, 0].detach().cpu().numpy()
        
        # Initialize and normalize audio data
        audio_data = spec.flatten()
        audio_data = np.interp(audio_data, (audio_data.min(), audio_data.max()), [-1, 1])
        
        # Set audio parameters
        sample_rate = 48000
        duration = 30
        
        # Prepare waveform with proper length
        total_samples = sample_rate * duration
        audio_data = np.tile(audio_data, int(np.ceil(total_samples / len(audio_data))))[:total_samples]
        
        # Safe conversion to 16-bit PCM
        
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Save WAV file
        wav_path = os.path.join(r'C:\Users\Lukasz\Music\generated', f"epoch{self.current_epoch}_sample.wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        write(wav_path, sample_rate, audio_data)
        
        print(f"[INFO] Saved audio to {wav_path}")

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