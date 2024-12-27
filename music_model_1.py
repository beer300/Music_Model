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
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

BATCH_SIZE = 8
data_dir = "\\Users\\lukas\\Music\\spectogram_30"

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith('.png')
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)  # Removed .convert("L")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
 
    transforms.ToTensor(),
])

full_dataset = SpectrogramDataset(image_dir=data_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
       
        out = self.model(x)
       
        return out.view(-1, 1)

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(self.latent_dim//2, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.ConvTranspose2d(32, 16, kernel_size=(8,8), stride=(4,4), padding=(2,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
        )
        self.final = nn.Sequential(
            
            nn.ConvTranspose2d(16, 4, kernel_size=(16,16), stride=(8,8), padding=(4,4)),
            #Snn.Sigmoid()
        )
    def forward(self, z):
       
        z = z.view(-1, self.latent_dim//2, 1, 2)
        
        x = self.main(z)
       
        x = self.final(x)
        x = nn.functional.interpolate(x, size=(1164, 2364), mode='bilinear', align_corners=False)
       
        return x

class GAN(pl.LightningModule):
    def _write_disk_spectrogram(self, path, dpi=120):
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
    def __init__(self, latent_dim=100, lr=0.0002, sample_rate=48000):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator(in_channels=4)
        self.validation_z = torch.randn(1, self.hparams.latent_dim, 1, 2)
    def forward(self, z):
        return self.generator(z)
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        real_imgs = batch
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, 1, 2).type_as(real_imgs)
        fake_imgs = self(z).detach()
        y_hat_real = self.discriminator(real_imgs)
        y_hat_fake = self.discriminator(fake_imgs)
        real_loss = self.adversarial_loss(y_hat_real, torch.ones_like(y_hat_real))
        fake_loss = self.adversarial_loss(y_hat_fake, torch.zeros_like(y_hat_fake))
        d_loss = 0.5 * (real_loss + fake_loss)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim).type_as(real_imgs)
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
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim).type_as(real_imgs)
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
        z = self.validation_z.type_as(self.generator.main[0].weight)
        spec = self.generator(z)[0, 0].detach().cpu().numpy()
        spec = self._center_crop_or_pad(spec, target_h=1024, target_w=2048)


        plt.figure(figsize=(20, 10))  # Changed from (16, 8) to match audio class
        librosa.display.specshow(spec, sr=self.hparams.sample_rate, hop_length=1024, x_axis='time', y_axis='log')
        plt.axis('off')
        plt.tight_layout()

        png_path = os.path.join("\\Users\\lukas\\Documents\\music_model\\generated_spectrograms", f"epoch{self.current_epoch}_spectrogram.png")
        self._write_disk_spectrogram(png_path)

        audio = librosa.griffinlim(
            S=spec,
            n_iter=64,
            hop_length=1024,
            win_length=1024,
            center=True
        )
        sr = self.hparams.sample_rate
        os.makedirs("generated_audio", exist_ok=True)
        wav_path = os.path.join("\\Users\\lukas\\Documents\\music_model\\generated_audio", f"epoch{self.current_epoch}_sample.wav")
        sf.write(wav_path, audio, sr)
        print(f"[INFO] Saved spectrogram to {png_path}")
        print(f"[INFO] Saved audio to {wav_path}")
    def _center_crop_or_pad(self, spec: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = spec.shape
        if h > target_h:
            diff = h - target_h
            spec = spec[diff // 2 : diff // 2 + target_h, :]
        if w > target_w:
            diff = w - target_w
            spec = spec[:, diff // 2 : diff // 2 + target_w]
        h, w = spec.shape
        if h < target_h:
            diff = target_h - h
            top_pad = diff // 2
            bot_pad = diff - top_pad
            spec = np.pad(spec, ((top_pad, bot_pad), (0, 0)), mode="constant", constant_values=0)
        if w < target_w:
            diff = target_w - w
            left_pad = diff // 2
            right_pad = diff - left_pad
            spec = np.pad(spec, ((0, 0), (left_pad, right_pad)), mode="constant", constant_values=0)
        return spec

if __name__ == "__main__":
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.manual_seed(42)
    model = GAN(
        latent_dim=100,
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