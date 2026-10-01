"""SUNOKILLER - Advanced Audio Synthesis System.

The package root intentionally avoids importing heavyweight neural/model
modules at import time. Lightweight subsystems such as ``sunokiller.runtime``
must remain usable without installing the full ML dependency stack.

Heavy public API symbols are loaded lazily on first attribute access so the
historical root-level API remains available when its optional dependencies are
installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

__version__ = "0.1.0"
__author__ = "MASSIVEMAGNETICS"


_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Core models
    "VocosVocoder": (".models", "VocosVocoder"),
    "DiffusionModel": (".models", "DiffusionModel"),
    "TextToMusicModel": (".models", "TextToMusicModel"),
    "MusicDiffusionTransformer": (".models", "MusicDiffusionTransformer"),
    # Synthesis
    "AudioSynthesizer": (".synthesis", "AudioSynthesizer"),
    # Optimization
    "quantize_model": (".quantization", "quantize_model"),
    # Pre-trained models
    "load_pretrained_weights": (".pretrained", "load_pretrained_weights"),
    "create_model_from_pretrained": (".pretrained", "create_model_from_pretrained"),
    "list_available_models": (".pretrained", "list_available_models"),
    # Voice cloning
    "VoiceCloner": (".voice_cloning", "VoiceCloner"),
    "extract_voice_from_file": (".voice_cloning", "extract_voice_from_file"),
    # Streaming
    "StreamingGenerator": (".streaming", "StreamingGenerator"),
    "create_streaming_synthesizer": (".streaming", "create_streaming_synthesizer"),
}


__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Load heavyweight root exports only when the caller actually asks for one."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

    module_name, attribute_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
