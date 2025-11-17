"""Training script for Vocos Vocoder

Trains the Vocos neural vocoder on audio datasets.
Supports multi-GPU training and various optimization techniques.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
from tqdm import tqdm
import wandb

from sunokiller.vocoders import VocosVocoder
from sunokiller.utils import get_device


class AudioDataset(Dataset):
    """Dataset for loading audio files for vocoder training."""
    
    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 24000,
        segment_length: int = 8192,
        num_mels: int = 80,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.num_mels = num_mels
        
        # Find all audio files
        self.audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            self.audio_files.extend(self.audio_dir.rglob(f'*{ext}'))
        
        print(f"Found {len(self.audio_files)} audio files")
        
        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=num_mels,
        )
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_files[idx]
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Random crop to segment length
        if audio.shape[1] > self.segment_length:
            start = torch.randint(0, audio.shape[1] - self.segment_length, (1,))
            audio = audio[:, start:start + self.segment_length]
        elif audio.shape[1] < self.segment_length:
            # Pad if too short
            padding = self.segment_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        # Compute mel-spectrogram
        mel = self.mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))  # Log scale
        
        return {
            "audio": audio.squeeze(0),
            "mel": mel.squeeze(0),
        }


def train(args):
    """Main training loop."""
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="sunokiller-vocos",
            config=vars(args),
        )
    
    # Setup device
    device = get_device() if args.device == "auto" else args.device
    print(f"Training on device: {device}")
    
    # Create model
    model = VocosVocoder(
        input_channels=args.num_mels,
        dim=args.dim,
        num_layers=args.num_layers,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset and dataloader
    dataset = AudioDataset(
        audio_dir=args.data_dir,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        num_mels=args.num_mels,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.8, 0.99),
        weight_decay=args.weight_decay,
    )
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01,
    )
    
    # Loss functions
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    # Training loop
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        
        for batch in pbar:
            audio = batch["audio"].to(device)
            mel = batch["mel"].to(device)
            
            # Forward pass
            predicted_audio = model(mel)
            
            # Ensure same length
            min_len = min(audio.shape[-1], predicted_audio.shape[-1])
            audio = audio[..., :min_len]
            predicted_audio = predicted_audio[..., :min_len]
            
            # Compute losses
            time_loss = l1_loss(predicted_audio, audio)
            
            # Multi-resolution STFT loss
            stft_loss = 0.0
            for fft_size in [512, 1024, 2048]:
                hop = fft_size // 4
                pred_stft = torch.stft(
                    predicted_audio,
                    n_fft=fft_size,
                    hop_length=hop,
                    return_complex=True,
                )
                true_stft = torch.stft(
                    audio,
                    n_fft=fft_size,
                    hop_length=hop,
                    return_complex=True,
                )
                stft_loss += l1_loss(pred_stft.abs(), true_stft.abs())
            
            stft_loss = stft_loss / 3.0
            
            # Total loss
            loss = time_loss + args.stft_loss_weight * stft_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "time_loss": f"{time_loss.item():.4f}",
                "stft_loss": f"{stft_loss.item():.4f}",
            })
            
            # Log to wandb
            if args.use_wandb and global_step % args.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/time_loss": time_loss.item(),
                    "train/stft_loss": stft_loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                }, step=global_step)
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = Path(args.output_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"vocos_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss / len(dataloader),
            }, checkpoint_path)
            
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = Path(args.output_dir) / "vocos_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Vocos Vocoder")
    
    # Data
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output-dir", type=str, default="outputs/vocos", help="Output directory")
    
    # Model architecture
    parser.add_argument("--num-mels", type=int, default=80, help="Number of mel bins")
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT size")
    parser.add_argument("--hop-length", type=int, default=256, help="Hop length")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--stft-loss-weight", type=float, default=1.0, help="STFT loss weight")
    
    # Audio
    parser.add_argument("--sample-rate", type=int, default=24000, help="Sample rate")
    parser.add_argument("--segment-length", type=int, default=8192, help="Audio segment length")
    
    # System
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Logging
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--log-interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save-interval", type=int, default=10, help="Save interval (epochs)")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
