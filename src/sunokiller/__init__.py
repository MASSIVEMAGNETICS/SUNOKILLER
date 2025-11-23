"""SUNOKILLER - Advanced Audio Synthesis System

A state-of-the-art audio synthesis system using cutting-edge techniques:
- Vocos: Fourier-based neural vocoder for 10x speedup
- DiffWave/SpecDiff-GAN: Fast diffusion models
- Transformer-based text-to-music generation
- Singing voice synthesis
- INT8/FP16 quantization for low-end hardware
- Voice cloning from audio samples
- Real-time streaming generation
- Pre-trained model weights
- Training scripts and datasets
- Web UI and mobile deployment support

Based on latest research from 2024-2025.
"""

__version__ = "0.1.0"
__author__ = "MASSIVEMAGNETICS"

from .models import VocosVocoder, DiffusionModel, TextToMusicModel, MusicDiffusionTransformer
from .synthesis import AudioSynthesizer
from .quantization import quantize_model
from .pretrained import (
    load_pretrained_weights,
    create_model_from_pretrained,
    list_available_models,
)
from .voice_cloning import VoiceCloner, extract_voice_from_file
from .streaming import StreamingGenerator, create_streaming_synthesizer

__all__ = [
    # Core models
    "VocosVocoder",
    "DiffusionModel", 
    "TextToMusicModel",
    "MusicDiffusionTransformer",
    
    # Synthesis
    "AudioSynthesizer",
    
    # Optimization
    "quantize_model",
    
    # Pre-trained models
    "load_pretrained_weights",
    "create_model_from_pretrained",
    "list_available_models",
    
    # Voice cloning
    "VoiceCloner",
    "extract_voice_from_file",
    
    # Streaming
    "StreamingGenerator",
    "create_streaming_synthesizer",
]
