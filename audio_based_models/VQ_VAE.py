# vqvae_model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy  as np
from torch.utils.data import DataLoader, random_split
import torchaudio
from scipy.io.wavfile import write
import dataset_processing as dp

class VQVAE(pl.LightningModule):
    def __init__(self, in_channels=1, embedding_dim=64, num_embeddings=512, 
                 num_restarts=3, sample_rate=48000):
        super().__init__()
        self.save_hyperparameters()
        
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

        for _ in range(self.hparams.num_restarts):
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
        x_recon, _ = self.forward(batch)
        loss = F.mse_loss(x_recon, batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_recon, _ = self.forward(batch)
        loss = F.mse_loss(x_recon, batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)

    def on_validation_epoch_end(self):
        output_dir = r'C:\Users\lukas\Music\vqvae_reconstructed'
        os.makedirs(output_dir, exist_ok=True)
        
        for i, batch in enumerate(self.trainer.val_dataloaders):
            x_recon, _ = self.forward(batch)
            x_recon = x_recon.detach().cpu().numpy()
            for j, audio in enumerate(x_recon):
                audio = audio.flatten()
                audio = np.tile(audio, int(np.ceil(self.hparams.sample_rate * 30 / len(audio))))[:self.hparams.sample_rate * 30]
                audio = (audio * 32767).astype(np.int16)
                wav_path = os.path.join(output_dir, f"val_epoch{self.current_epoch}_batch{i}_sample{j}.wav")
                write(wav_path, self.hparams.sample_rate, audio)
        self.save_checkpoint()

    def save_checkpoint(self):
        checkpoint_dir = r'C:\Users\lukas\Music\vqvae_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'vqvae_epoch{self.current_epoch}.ckpt')
        torch.save(self.state_dict(), checkpoint_path)

def main():
    # Configuration
    BATCH_SIZE = 16
    AUDIO_DIR = r"C:\Users\lukas\Music\youtube_playlist_chopped"
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 1440000
    
    # Create dataset
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
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    vqvae_model = VQVAE(in_channels=1, num_restarts=3, sample_rate=SAMPLE_RATE)
    
    # Configure checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="vqvae_checkpoints",
        filename="vqvae-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(vqvae_model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()