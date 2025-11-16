"""Vocos: Fourier-based Neural Vocoder Implementation

Based on "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders"
(ICLR 2024) - https://arxiv.org/abs/2306.00814

Key features:
- Direct Fourier coefficient generation instead of time-domain
- 10x faster than traditional neural vocoders
- Minimal quality degradation
- Optimized for low-latency inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for efficient feature extraction."""
    
    def __init__(self, channels: int, intermediate_dim: int, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels)) if layer_scale_init > 0 else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        return x + residual


class VocosBackbone(nn.Module):
    """Backbone network for Vocos using ConvNeXt blocks."""
    
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            ConvNeXtBlock(dim, intermediate_dim)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        
        for layer in self.layers:
            x = layer(x)
            
        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)
        x = x.transpose(1, 2)
        return x


class ISTFTHead(nn.Module):
    """Inverse STFT head for converting Fourier coefficients to audio."""
    
    def __init__(
        self,
        dim: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        padding: str = "same",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Output both magnitude and phase
        out_dim = n_fft + 2  # +2 for DC and Nyquist
        self.out = nn.Conv1d(dim, out_dim, kernel_size=7, padding=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) feature tensor
        Returns:
            audio: (B, audio_len) reconstructed audio
        """
        x = self.out(x)  # (B, n_fft+2, T)
        
        # Split into magnitude and phase components
        mag = x[:, :self.n_fft // 2 + 1]  # (B, n_fft//2+1, T)
        phase = x[:, self.n_fft // 2 + 1:]  # (B, n_fft//2+1, T)
        
        # Apply activation for magnitude (ensure positive)
        mag = torch.exp(mag)
        phase = torch.tanh(phase) * math.pi
        
        # Construct complex STFT
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        stft = torch.complex(real, imag)
        
        # Apply inverse STFT
        audio = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            return_complex=False,
        )
        
        return audio


class VocosVocoder(nn.Module):
    """
    Vocos: Fourier-based neural vocoder.
    
    Converts mel-spectrograms or other acoustic features to high-quality audio
    using direct Fourier coefficient generation.
    
    Args:
        input_channels: Number of input feature channels (e.g., 80 for mel-spec)
        dim: Hidden dimension size
        intermediate_dim: Intermediate dimension for ConvNeXt blocks
        num_layers: Number of ConvNeXt blocks
        n_fft: FFT size for ISTFT
        hop_length: Hop length for ISTFT
    """
    
    def __init__(
        self,
        input_channels: int = 80,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.backbone = VocosBackbone(input_channels, dim, intermediate_dim, num_layers)
        self.head = ISTFTHead(dim, n_fft, hop_length)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, T) acoustic features (e.g., mel-spectrogram)
        Returns:
            audio: (B, audio_len) generated audio waveform
        """
        x = self.backbone(features)
        audio = self.head(x)
        return audio
    
    @torch.inference_mode()
    def generate(self, features: torch.Tensor) -> torch.Tensor:
        """Inference mode generation."""
        return self.forward(features)
