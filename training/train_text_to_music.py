"""Training script for Text-to-Music Model

Trains the transformer-based text-to-music model on text-audio pairs.
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
import json

from sunokiller.models import TextToMusicModel
from sunokiller.utils import get_device


class TextAudioDataset(Dataset):
    """Dataset for text-audio pairs."""
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 24000,
        max_duration: float = 10.0,
        num_mels: int = 80,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.num_mels = num_mels
        
        # Load metadata (JSON file with text descriptions)
        metadata_file = self.data_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {data_dir}. "
                "Please create a file with format: "
                '{"audio_file.wav": "description of the music", ...}'
            )
        
        with open(metadata_file) as f:
            self.metadata = json.load(f)
        
        self.audio_files = list(self.metadata.keys())
        print(f"Loaded {len(self.audio_files)} text-audio pairs")
        
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
        audio_file = self.audio_files[idx]
        text = self.metadata[audio_file]
        
        # Load audio
        audio_path = self.data_dir / audio_file
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Truncate or pad
        max_samples = int(self.max_duration * self.sample_rate)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        elif audio.shape[1] < max_samples:
            padding = max_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        # Compute mel-spectrogram
        mel = self.mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return {
            "text": text,
            "mel": mel.squeeze(0),
        }


def train(args):
    """Main training loop."""
    
    if args.use_wandb:
        wandb.init(
            project="sunokiller-text-to-music",
            config=vars(args),
        )
    
    device = get_device() if args.device == "auto" else args.device
    print(f"Training on device: {device}")
    
    # Create model
    model = TextToMusicModel(
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_mel_bins=args.num_mels,
        text_encoder_name=args.text_encoder,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    dataset = TextAudioDataset(
        data_dir=args.data_dir,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        num_mels=args.num_mels,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01,
    )
    
    # Loss
    criterion = nn.MSELoss()
    
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        
        for batch in pbar:
            text = batch["text"]
            mel_target = batch["mel"].to(device)
            
            # Encode text
            text_features = model.encode_text(text, device)
            
            # Forward pass with teacher forcing
            mel_pred = model(text_features, mel_target)
            
            # Transpose for loss computation
            mel_pred = mel_pred.transpose(1, 2)  # (B, T, C) -> (B, C, T)
            
            # Compute loss
            loss = criterion(mel_pred, mel_target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if args.use_wandb and global_step % args.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                }, step=global_step)
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = Path(args.output_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"text_to_music_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss / len(dataloader),
            }, checkpoint_path)
            
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = Path(args.output_dir) / "text_to_music_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Text-to-Music Model")
    
    # Data
    parser.add_argument("--data-dir", type=str, required=True, 
                       help="Directory with audio files and metadata.json")
    parser.add_argument("--output-dir", type=str, default="outputs/text_to_music")
    
    # Model
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--ff-dim", type=int, default=3072)
    parser.add_argument("--num-mels", type=int, default=80)
    parser.add_argument("--text-encoder", type=str, default="none",
                       help="T5 encoder name or 'none' for simple embedding")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    # Audio
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-duration", type=float, default=10.0)
    
    # System
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Logging
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=10)
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
