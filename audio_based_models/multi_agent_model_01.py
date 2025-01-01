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
    def __init__(self, num_generators=2):
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
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Add adaptive pooling to get fixed size output
        )
        
        # Remove sigmoid activation from validity layer
        self.validity = nn.Linear(256, 1)
        self.generator_label = nn.Sequential(
            nn.Linear(256, num_generators + 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        print(f"x in", x.shape)
        features = self.model(x)
        features = features.view(features.size(0), -1)  # Flatten to (batch_size, features)
        validity = self.validity(features)
        generator_label = self.generator_label(features)
        return validity, generator_label
    
class Generator(nn.Module):
    def __init__(self, latent_dim=200):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=(2,7), mode='nearest'),
            nn.Conv2d(latent_dim, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
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

            nn.Upsample(scale_factor=(1,3), mode='nearest'),
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
        # Reshape z to match the expected input shape
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        x = self.main(z)
        print(f"main out", x.shape)
        x = self.final(x)
        print(f"x out", x.shape)
        return x



class GAN(pl.LightningModule):
    def _write_disk_spectrogram(self, path, dpi=120):
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

    def __init__(self, latent_dim=200, num_generators=2, lr=0.001, sample_rate=48000):
        super().__init__()
        self.hparams.latent_dim = latent_dim
        self.hparams.lr = lr
        self.hparams.sample_rate = sample_rate
        self.num_generators = num_generators
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Create multiple generators
        self.generators = nn.ModuleList([Generator(latent_dim) for _ in range(num_generators)])
        self.discriminator = Discriminator(num_generators=num_generators)
        self.validation_z = torch.randn(1, self.hparams.latent_dim)
        
        # Replace BCELoss with BCEWithLogitsLoss
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.categorical_loss = nn.CrossEntropyLoss()

    def forward(self, z, generator_idx=None):
        if generator_idx is None:
            generator_idx = torch.randint(0, self.num_generators, (1,)).item()
        return self.generators[generator_idx](z)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        opts_g = optimizers[:-1]
        opt_d = optimizers[-1]
        
        real_imgs = batch
        batch_size = real_imgs.size(0)

        # Train Discriminator
        opt_d.zero_grad()
        
        validity_real, label_real = self.discriminator(real_imgs)
        d_real_loss = self.adversarial_loss(validity_real, torch.ones(batch_size, 1, device=self.device))
        real_labels = torch.full((batch_size,), self.num_generators, device=self.device)
        label_real_loss = self.categorical_loss(label_real, real_labels)

        # Fake images from each generator
        d_fake_loss = 0
        label_fake_loss = 0
        fake_imgs_list = []  # Store fake images for each generator
        
        for gen_idx in range(self.num_generators):
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            fake_imgs = self.generators[gen_idx](z).detach()
            fake_imgs_list.append(fake_imgs)
            validity_fake, label_fake = self.discriminator(fake_imgs)
            d_fake_loss += self.adversarial_loss(validity_fake, torch.zeros(batch_size, 1, device=self.device))
            fake_labels = torch.full((batch_size,), gen_idx, device=self.device)
            label_fake_loss += self.categorical_loss(label_fake, fake_labels)

        d_fake_loss /= self.num_generators
        label_fake_loss /= self.num_generators
        
        d_loss = (d_real_loss + d_fake_loss) / 2 + (label_real_loss + label_fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()

        # Train Generators
        for gen_idx, opt_g in enumerate(opts_g):
            opt_g.zero_grad()
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            fake_imgs = self.generators[gen_idx](z)
            validity_fake, label_fake = self.discriminator(fake_imgs)
            
            g_loss = self.adversarial_loss(validity_fake, torch.ones(batch_size, 1, device=self.device))
            fake_labels = torch.full((batch_size,), gen_idx, device=self.device)
            g_label_loss = self.categorical_loss(label_fake, fake_labels)
            
            g_total_loss = g_loss + g_label_loss
            self.manual_backward(g_total_loss)
            opt_g.step()

            self.log(f'g{gen_idx}_loss', g_total_loss, prog_bar=True, on_epoch=True)

        self.log('d_loss', d_loss, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        real_imgs = batch
        batch_size = real_imgs.size(0)
        
        validity_real, label_real = self.discriminator(real_imgs)
        d_real_loss = self.adversarial_loss(validity_real, torch.ones_like(validity_real))
        
        d_fake_losses = []
        g_losses = []
        
        for gen_idx in range(self.num_generators):
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            fake_imgs = self.generators[gen_idx](z)
            validity_fake, label_fake = self.discriminator(fake_imgs)
            
            d_fake_loss = self.adversarial_loss(validity_fake, torch.zeros_like(validity_fake))
            g_loss = self.adversarial_loss(validity_fake, torch.ones_like(validity_fake))
            
            d_fake_losses.append(d_fake_loss)
            g_losses.append(g_loss)
            
            self.log(f'val_g{gen_idx}_loss', g_loss, prog_bar=True, on_epoch=True)
        
        d_loss = (d_real_loss + sum(d_fake_losses) / self.num_generators) / 2
        self.log('val_d_loss', d_loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        # Create list of generator optimizers
        opts_g = [torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999)) 
                 for gen in self.generators]
        # Create discriminator optimizer
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        # Return all optimizers (generator optimizers followed by discriminator optimizer)
        return opts_g + [opt_d]

    def on_validation_epoch_end(self):
        # Generate samples from each generator
        for gen_idx in range(self.num_generators):
            z = self.validation_z.type_as(self.generators[0].main[1].weight)
            spec = self.generators[gen_idx](z)[0, 0].detach().cpu().numpy()
            
            audio_data = spec.flatten()
            audio_data = np.interp(audio_data, (audio_data.min(), audio_data.max()), [-1, 1])
            
            sample_rate = self.hparams.sample_rate
            duration = 30
            
            total_samples = sample_rate * duration
            audio_data = np.tile(audio_data, int(np.ceil(total_samples / len(audio_data))))[:total_samples]
            audio_data = (audio_data * 32767).astype(np.int16)
            
            wav_path = os.path.join(r'C:\Users\Lukasz\Music\generated', 
                                  f"epoch{self.current_epoch}_gen{gen_idx}_sample.wav")
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            write(wav_path, sample_rate, audio_data)
            
            print(f"[INFO] Saved audio from generator {gen_idx} to {wav_path}")

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
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4
    )
    model = GAN(
        latent_dim=200,
        num_generators=2,  # specify number of generators
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