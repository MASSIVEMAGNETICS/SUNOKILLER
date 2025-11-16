"""Utility functions for audio processing."""

import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple


def load_audio(
    path: str,
    sample_rate: int = 24000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate (will resample if different)
        mono: Convert to mono if True
        
    Returns:
        Audio tensor and sample rate
    """
    audio, sr = torchaudio.load(path)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
        sr = sample_rate
    
    # Convert to mono if needed
    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    return audio, sr


def save_audio(
    audio: torch.Tensor,
    path: str,
    sample_rate: int = 24000,
):
    """
    Save audio to file.
    
    Args:
        audio: Audio tensor (C, T)
        path: Output path
        sample_rate: Sample rate
    """
    torchaudio.save(path, audio.cpu(), sample_rate)


def audio_to_mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """
    Convert audio to mel-spectrogram.
    
    Args:
        audio: Audio tensor (B, T) or (T,)
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bins
        sample_rate: Sample rate
        
    Returns:
        Mel-spectrogram (B, n_mels, T) or (n_mels, T)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    ).to(audio.device)
    
    mel_spec = mel_transform(audio)
    
    # Convert to log scale
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    return mel_spec


def normalize_audio(
    audio: np.ndarray,
    target_level: float = -20.0,
) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio array
        target_level: Target level in dB
        
    Returns:
        Normalized audio
    """
    rms = np.sqrt(np.mean(audio ** 2))
    current_level = 20 * np.log10(rms + 1e-8)
    gain = 10 ** ((target_level - current_level) / 20)
    return audio * gain


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
