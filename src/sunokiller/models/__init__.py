"""Models module - Core neural network architectures for SUNOKILLER."""

from ..vocoders.vocos import VocosVocoder
from ..diffusion.diffwave import DiffusionModel
from .dit_transformer import MusicDiffusionTransformer

# Alias for compatibility
TextToMusicModel = MusicDiffusionTransformer

__all__ = [
    "VocosVocoder",
    "DiffusionModel",
    "MusicDiffusionTransformer",
    "TextToMusicModel",
]
