"""Tests for MusicDiffusionTransformer model."""

import torch


def test_music_diffusion_transformer_import():
    """Test that MusicDiffusionTransformer can be imported."""
    from sunokiller.models import MusicDiffusionTransformer
    assert MusicDiffusionTransformer is not None


def test_music_diffusion_transformer_creation():
    """Test that MusicDiffusionTransformer can be instantiated."""
    from sunokiller.models import MusicDiffusionTransformer
    
    model = MusicDiffusionTransformer(
        dim=128,
        num_layers=2,
        num_heads=4,
        vocab_size=256,
        mel_channels=80,
        max_seq_len=512
    )
    
    # Check that model is created
    assert model is not None
    assert model.dim == 128
    assert model.mel_channels == 80
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    assert num_params > 0


def test_music_diffusion_transformer_forward():
    """Test forward pass through the model."""
    from sunokiller.models import MusicDiffusionTransformer
    
    # Create a small model for testing
    model = MusicDiffusionTransformer(
        dim=128,
        num_layers=2,
        num_heads=4,
        mel_channels=80,
        max_seq_len=512
    )
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 64
    mel_channels = 80
    text_len = 16
    
    x = torch.randn(batch_size, seq_len, mel_channels)  # Noisy mel-spec
    t = torch.randint(0, 1000, (batch_size,))  # Timesteps
    text_indices = torch.randint(0, 256, (batch_size, text_len))  # Text tokens
    
    # Forward pass
    with torch.no_grad():
        output = model(x, t, text_indices)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, mel_channels)
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")


def test_core_components():
    """Test individual core components."""
    from sunokiller.models.dit_transformer import RMSNorm, SwiGLU, RotaryEmbedding
    
    # Test RMSNorm
    norm = RMSNorm(dim=128)
    x = torch.randn(2, 10, 128)
    out = norm(x)
    assert out.shape == x.shape
    print("✓ RMSNorm works")
    
    # Test SwiGLU
    swiglu = SwiGLU(dim=128, hidden_dim=512)
    x = torch.randn(2, 10, 128)
    out = swiglu(x)
    assert out.shape == x.shape
    print("✓ SwiGLU works")
    
    # Test RotaryEmbedding
    rope = RotaryEmbedding(dim=64, max_seq_len=512)
    x = torch.randn(2, 10, 4, 64)  # [batch, seq, heads, head_dim]
    cos, sin = rope(x)
    assert cos.shape[0] == 10  # seq_len
    assert cos.shape[1] == 64  # dim
    print("✓ RotaryEmbedding works")


def test_timestep_embedder():
    """Test TimestepEmbedder."""
    from sunokiller.models.dit_transformer import TimestepEmbedder
    
    embedder = TimestepEmbedder(hidden_size=256)
    t = torch.randint(0, 1000, (4,))  # 4 timesteps
    emb = embedder(t)
    
    assert emb.shape == (4, 256)
    print("✓ TimestepEmbedder works")


def test_dit_block():
    """Test DiTBlock."""
    from sunokiller.models.dit_transformer import DiTBlock, RotaryEmbedding
    
    block = DiTBlock(dim=128, num_heads=4)
    
    batch_size = 2
    seq_len = 32
    text_len = 16
    
    x = torch.randn(batch_size, seq_len, 128)
    t_emb = torch.randn(batch_size, 128)
    context = torch.randn(batch_size, text_len, 128)
    
    # Get RoPE embeddings
    rope = RotaryEmbedding(dim=32, max_seq_len=512)  # 128 / 4 heads = 32
    cos, sin = rope(x, seq_len=seq_len)
    
    out = block(x, t_emb, context, cos, sin)
    assert out.shape == x.shape
    print("✓ DiTBlock works")


def test_generation():
    """Test generation method."""
    from sunokiller.models import MusicDiffusionTransformer
    
    # Create a very small model for quick testing
    model = MusicDiffusionTransformer(
        dim=64,
        num_layers=1,
        num_heads=2,
        mel_channels=80,
        max_seq_len=256
    )
    
    # Generate a very short clip with few steps
    output = model.generate(
        text="test music",
        duration_sec=0.5,  # Very short
        steps=5,  # Few steps for speed
        device="cpu"
    )
    
    # Check output shape (0.5s * 86 frames/s = ~43 frames)
    expected_len = int(0.5 * 86)
    assert output.shape[0] == 1  # batch size
    assert output.shape[1] == expected_len
    assert output.shape[2] == 80  # mel channels
    
    print(f"✓ Generation works: {output.shape}")


def test_texttomusic_alias():
    """Test that TextToMusicModel is an alias for MusicDiffusionTransformer."""
    from sunokiller.models import TextToMusicModel, MusicDiffusionTransformer
    
    assert TextToMusicModel is MusicDiffusionTransformer
    print("✓ TextToMusicModel alias works")


def test_model_import_from_main():
    """Test that model can be imported from main sunokiller module."""
    from sunokiller import MusicDiffusionTransformer, TextToMusicModel
    
    assert MusicDiffusionTransformer is not None
    assert TextToMusicModel is not None
    print("✓ Can import from main module")


if __name__ == "__main__":
    print("=" * 60)
    print("MusicDiffusionTransformer Tests")
    print("=" * 60)
    
    test_music_diffusion_transformer_import()
    test_music_diffusion_transformer_creation()
    test_music_diffusion_transformer_forward()
    test_core_components()
    test_timestep_embedder()
    test_dit_block()
    test_generation()
    test_texttomusic_alias()
    test_model_import_from_main()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
