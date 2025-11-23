"""Demo script for MusicDiffusionTransformer.

This script demonstrates how to use the MusicDiffusionTransformer model
for text-to-music generation.
"""

import torch
from sunokiller.models import MusicDiffusionTransformer


def main():
    print("=" * 70)
    print("MusicDiffusionTransformer Demo")
    print("=" * 70)
    
    # Create a model with reasonable size for demo
    print("\n1. Creating MusicDiffusionTransformer model...")
    model = MusicDiffusionTransformer(
        dim=512,           # Hidden dimension
        num_layers=6,      # Number of transformer layers
        num_heads=8,       # Number of attention heads
        vocab_size=256,    # Vocabulary size for text
        mel_channels=80,   # Number of mel-spectrogram channels
        max_seq_len=2048   # Maximum sequence length
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {num_params:,} parameters")
    
    # Forward pass example
    print("\n2. Testing forward pass (training mode)...")
    batch_size = 2
    seq_len = 128
    mel_channels = 80
    text_len = 32
    
    # Create dummy inputs
    x = torch.randn(batch_size, seq_len, mel_channels)  # Noisy mel-spec
    t = torch.randint(0, 1000, (batch_size,))           # Timesteps
    text_indices = torch.randint(0, 256, (batch_size, text_len))  # Text tokens
    
    with torch.no_grad():
        output = model(x, t, text_indices)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output.shape}")
    print("   ✓ Forward pass successful!")
    
    # Generation example
    print("\n3. Testing generation (inference mode)...")
    text_prompts = [
        "A funky jazz tune with saxophone",
        "Epic orchestral soundtrack with strings"
    ]
    
    for text in text_prompts:
        print(f"\n   Generating: '{text}'")
        output = model.generate(
            text=text,
            duration_sec=2.0,  # Short duration for demo
            steps=10,          # Few steps for speed
            device="cpu"
        )
        print(f"   Generated shape: {output.shape}")
        print(f"   Duration: ~{output.shape[1] / 86:.1f}s @ 86 fps")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    # Component demonstration
    print("\n4. Testing individual components...")
    
    from sunokiller.models.dit_transformer import (
        RMSNorm, SwiGLU, RotaryEmbedding, 
        TimestepEmbedder, DiTBlock
    )
    
    # RMSNorm
    norm = RMSNorm(dim=512)
    x = torch.randn(2, 10, 512)
    out = norm(x)
    print(f"   RMSNorm:          {x.shape} -> {out.shape} ✓")
    
    # SwiGLU
    swiglu = SwiGLU(dim=512, hidden_dim=2048)
    out = swiglu(x)
    print(f"   SwiGLU:           {x.shape} -> {out.shape} ✓")
    
    # RotaryEmbedding
    rope = RotaryEmbedding(dim=64, max_seq_len=2048)
    x_heads = torch.randn(2, 10, 8, 64)
    cos, sin = rope(x_heads)
    print(f"   RotaryEmbedding:  seq_len={x_heads.shape[1]} -> cos/sin shape={cos.shape} ✓")
    
    # TimestepEmbedder
    time_emb = TimestepEmbedder(hidden_size=512)
    t = torch.randint(0, 1000, (4,))
    emb = time_emb(t)
    print(f"   TimestepEmbedder: timesteps={t.shape} -> {emb.shape} ✓")
    
    # DiTBlock
    block = DiTBlock(dim=512, num_heads=8)
    x = torch.randn(2, 64, 512)
    t_emb = torch.randn(2, 512)
    context = torch.randn(2, 32, 512)
    cos, sin = rope(x, seq_len=64)
    out = block(x, t_emb, context, cos, sin)
    print(f"   DiTBlock:         {x.shape} -> {out.shape} ✓")
    
    print("\n" + "=" * 70)
    print("All components tested successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
