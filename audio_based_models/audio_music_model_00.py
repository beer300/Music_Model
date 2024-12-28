from torch import nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
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
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            
            nn.Upsample(scale_factor=3, mode='bilinear'),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Upsample(scale_factor=3, mode='bilinear'),
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
            nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
        )
        self.final = nn.Sequential(
            
            nn.ConvTranspose2d(16, 4, kernel_size=(16,16), stride=(8,8), padding=(4,4)),
            nn.Sigmoid()
        )
        def forward(self, z):
       
        z = z.view(-1, self.latent_dim//2, 1, 2)
        
        x = self.main(z)
       
        x = self.final(x)
        
       
        return x
    
class GAN(pl.LightningModule):
    def _write_disk_spectrogram(self, path, dpi=120):
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
    def __init__(self, latent_dim=400, lr=0.0002, sample_rate=48000):
        super().__init__()
        self.hparams.latent_dim = latent_dim
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
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, 1, 2).type_as(real_imgs)
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
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim, 1, 2).type_as(real_imgs)
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
        spec = self._center_crop_or_pad(spec, target_h=1164, target_w=2364)

        plt.figure(figsize=(16, 8))
        plt.imshow(librosa.amplitude_to_db(spec, ref=np.max), cmap='inferno', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - Epoch {self.current_epoch}')
        plt.tight_layout()
        png_old_path = os.path.join('/content/drive/My Drive/data/generated_old_spectrograms_model_03', f"epoch{self.current_epoch}_old_spectrogram.png")
        plt.savefig(png_old_path, bbox_inches='tight')
        plt.close()


        plt.figure(figsize=(20, 10))  # Changed from (16, 8) to match audio class
        librosa.display.specshow(spec, sr=self.hparams.sample_rate, hop_length=1024, x_axis='time', y_axis='log')
        plt.axis('off')
        plt.tight_layout()

        png_path = os.path.join('/content/drive/My Drive/data/generated_spectrograms_model_03', f"epoch{self.current_epoch}_spectrogram.png")
        self._write_disk_spectrogram(png_path)

        image = Image.open(png_path)
        image = image.resize((2364, 1164))  # Resize to specified dimensions
        image = image.convert('L')  # Convert to grayscale

        # Normalize pixel data to range [-1, 1]
        pixel_data = np.array(image)
        normalized_data = (pixel_data / 255.0) * 2 - 1

        # Flatten the image data to a 1D array
        audio_data = normalized_data.flatten()

        # Scale the data to fit the hop length
        audio_data = np.interp(audio_data, (audio_data.min(), audio_data.max()), [-1, 1])

        # Repeat or pad to make it a valid waveform for WAV
        total_samples = sample_rate * duration
        audio_data = np.tile(audio_data, int(np.ceil(total_samples / len(audio_data))))[:total_samples]

        # Convert to 16-bit PCM format
        audio_data = (audio_data * 32767).astype(np.int16)

        # Save as a WAV file

        wav_path = os.path.join('/content/drive/My Drive/data/generated_audio_model_03', f"epoch{self.current_epoch}_sample.wav")
        write(wav_path, sample_rate, audio_data)

        print(f"[INFO] Saved spectrogram to {png_path}")
        print(f"[INFO] Saved audio to {wav_path}")