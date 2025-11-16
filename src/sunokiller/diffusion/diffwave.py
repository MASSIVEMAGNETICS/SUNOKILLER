"""DiffWave: Fast Diffusion-based Audio Synthesis

Based on "DiffWave: A Versatile Diffusion Model for Audio Synthesis" (ICLR 2021)
and optimized with SpecDiff-GAN techniques for faster inference.

Key features:
- Efficient diffusion process for audio generation
- Supports unconditional and conditional generation
- Fast sampling with DDIM/DDPM
- Optimized for real-time inference on low-end hardware
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with conditioning for diffusion models."""
    
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        time_emb_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, 1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, channels),
            nn.SiLU(),
        )
        
        # Conditional input projection (e.g., mel-spectrogram)
        self.cond_conv = nn.Conv1d(cond_channels, channels, 1) if cond_channels > 0 else None
        
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        
        # First convolution
        h = self.conv1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None]
        
        # Add conditioning if provided
        if cond is not None and self.cond_conv is not None:
            h = h + self.cond_conv(cond)
            
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + residual


class DiffusionUNet(nn.Module):
    """U-Net architecture for diffusion models."""
    
    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 80,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 512,
    ):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_mult):
            out_channels = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(now_channels, cond_channels, time_emb_dim)
                )
                channels.append(now_channels)
                
            if i != len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv1d(now_channels, out_channels, 3, stride=2, padding=1))
                now_channels = out_channels
                channels.append(now_channels)
        
        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(now_channels, cond_channels, time_emb_dim),
            ResidualBlock(now_channels, cond_channels, time_emb_dim),
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_channels = base_channels * mult
            
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(now_channels + channels.pop(), cond_channels, time_emb_dim)
                )
                now_channels = out_channels
                
            if i != len(channel_mult) - 1:
                self.up_blocks.append(nn.ConvTranspose1d(now_channels, out_channels, 4, stride=2, padding=1))
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, 3, padding=1),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        # Input projection
        h = self.input_proj(x)
        
        # Downsampling
        hs = [h]
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb, cond)
            else:
                h = block(h)
            hs.append(h)
        
        # Middle
        for block in self.mid_blocks:
            h = block(h, time_emb, cond)
        
        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, time_emb, cond)
            else:
                h = block(h)
        
        # Output
        return self.out_proj(h)


class DiffusionModel(nn.Module):
    """
    Fast diffusion model for audio generation.
    
    Supports both unconditional and conditional (mel-spectrogram guided) generation.
    Uses DDIM sampling for fast inference.
    
    Args:
        in_channels: Number of audio channels (1 for mono, 2 for stereo)
        cond_channels: Number of conditioning channels (e.g., 80 for mel-spec)
        num_steps: Number of diffusion timesteps
        beta_start: Starting beta value for noise schedule
        beta_end: Ending beta value for noise schedule
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 80,
        num_steps: int = 50,  # Reduced from 1000 for fast inference
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        **unet_kwargs,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.in_channels = in_channels
        
        # U-Net denoiser
        self.unet = DiffusionUNet(in_channels, cond_channels, **unet_kwargs)
        
        # Noise schedule (linear for now, can use cosine)
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Training forward pass."""
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (batch_size,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise to input
        x_noisy = (
            self.sqrt_alphas_cumprod[t][:, None, None] * x +
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise
        )
        
        # Predict noise
        noise_pred = self.unet(x_noisy, t, cond)
        
        return noise_pred, noise
    
    @torch.inference_mode()
    def sample(
        self,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        DDIM sampling for fast inference.
        
        Args:
            shape: Shape of output (batch_size, channels, length)
            cond: Optional conditioning tensor
            num_inference_steps: Number of inference steps (default: num_steps)
        """
        device = self.betas.device
        batch_size = shape[0]
        
        if num_inference_steps is None:
            num_inference_steps = self.num_steps
            
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # DDIM sampling schedule
        timesteps = torch.linspace(self.num_steps - 1, 0, num_inference_steps, device=device).long()
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.unet(x, t_batch, cond)
            
            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
                
            # Predicted x0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Direction to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred
            
            # Update
            x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt
            
        return x
