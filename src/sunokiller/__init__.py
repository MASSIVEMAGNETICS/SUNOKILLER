"""SUNOKILLER - Advanced Audio Synthesis System

A state-of-the-art audio synthesis system using cutting-edge techniques:
- Vocos: Fourier-based neural vocoder for 10x speedup
- DiffWave/SpecDiff-GAN: Fast diffusion models
- Transformer-based text-to-music generation
- Singing voice synthesis
- INT8/FP16 quantization for low-end hardware

Based on latest research from 2024-2025.
"""

__version__ = "0.1.0"
__author__ = "MASSIVEMAGNETICS"

from .models import VocosVocoder, DiffusionModel, TextToMusicModel
from .synthesis import AudioSynthesizer
from .quantization import quantize_model

__all__ = [
    "VocosVocoder",
    "DiffusionModel", 
    "TextToMusicModel",
    "AudioSynthesizer",
    "quantize_model",
]
