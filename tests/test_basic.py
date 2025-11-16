"""Simple test to validate the SUNOKILLER package structure."""

import sys
import torch

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from sunokiller import AudioSynthesizer
        print("✓ AudioSynthesizer imported")
    except Exception as e:
        print(f"✗ Failed to import AudioSynthesizer: {e}")
        return False
    
    try:
        from sunokiller.models import VocosVocoder, DiffusionModel, TextToMusicModel
        print("✓ Models imported")
    except Exception as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    try:
        from sunokiller.quantization import quantize_model
        print("✓ Quantization imported")
    except Exception as e:
        print(f"✗ Failed to import quantization: {e}")
        return False
    
    try:
        from sunokiller.utils import get_device
        print("✓ Utils imported")
    except Exception as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    return True


def test_model_creation():
    """Test that models can be instantiated."""
    print("\nTesting model creation...")
    
    try:
        from sunokiller.vocoders import VocosVocoder
        vocoder = VocosVocoder(input_channels=80, dim=128, num_layers=2)
        print(f"✓ VocosVocoder created: {sum(p.numel() for p in vocoder.parameters())} parameters")
    except Exception as e:
        print(f"✗ Failed to create VocosVocoder: {e}")
        return False
    
    try:
        from sunokiller.diffusion import DiffusionModel
        diffusion = DiffusionModel(num_steps=10)
        print(f"✓ DiffusionModel created: {sum(p.numel() for p in diffusion.parameters())} parameters")
    except Exception as e:
        print(f"✗ Failed to create DiffusionModel: {e}")
        return False
    
    try:
        from sunokiller.models import TextToMusicModel
        # Use smaller config for testing, avoid T5 loading
        text_model = TextToMusicModel(
            dim=128, 
            num_layers=2, 
            num_heads=4,
            text_encoder_name="none"  # Skip T5 loading
        )
        print(f"✓ TextToMusicModel created: {sum(p.numel() for p in text_model.parameters())} parameters")
    except Exception as e:
        print(f"✗ Failed to create TextToMusicModel: {e}")
        return False
    
    return True


def test_forward_pass():
    """Test simple forward passes."""
    print("\nTesting forward passes...")
    
    try:
        from sunokiller.vocoders import VocosVocoder
        
        vocoder = VocosVocoder(input_channels=80, dim=128, num_layers=2)
        
        # Create dummy mel-spectrogram input
        mel_spec = torch.randn(1, 80, 32)  # (batch, channels, time)
        
        # Forward pass
        with torch.no_grad():
            audio = vocoder(mel_spec)
        
        print(f"✓ Vocos forward pass: input {mel_spec.shape} -> output {audio.shape}")
    except Exception as e:
        print(f"✗ Vocos forward pass failed: {e}")
        return False
    
    try:
        from sunokiller.diffusion import DiffusionModel
        
        diffusion = DiffusionModel(num_steps=10, base_channels=32)
        
        # Create dummy audio input - testing without conditioning first
        audio = torch.randn(1, 1, 256)
        
        # Forward pass (training mode) - unconditional
        with torch.no_grad():
            noise_pred, noise = diffusion(audio, cond=None)
        
        print(f"✓ Diffusion forward pass: input {audio.shape} -> noise prediction {noise_pred.shape}")
    except Exception as e:
        print(f"✗ Diffusion forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_synthesizer():
    """Test AudioSynthesizer initialization."""
    print("\nTesting AudioSynthesizer...")
    
    try:
        from sunokiller import AudioSynthesizer
        from sunokiller.models import VocosVocoder, DiffusionModel, TextToMusicModel
        
        # Create small models for testing
        vocoder = VocosVocoder(dim=128, num_layers=2)
        diffusion = DiffusionModel(num_steps=5, base_channels=32)
        text_model = TextToMusicModel(
            dim=128, 
            num_layers=2, 
            num_heads=4,
            text_encoder_name="none"  # Skip T5 loading
        )
        
        synth = AudioSynthesizer(
            text_to_music_model=text_model,
            diffusion_model=diffusion,
            vocoder=vocoder,
            device="cpu",
        )
        
        print(f"✓ AudioSynthesizer created successfully")
    except Exception as e:
        print(f"✗ AudioSynthesizer creation failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SUNOKILLER Package Validation")
    print("=" * 60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_model_creation():
        all_passed = False
    
    if not test_forward_pass():
        all_passed = False
    
    if not test_synthesizer():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
