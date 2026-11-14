"""Pre-trained Model Weights Management

Utilities for downloading, loading, and managing pre-trained model weights.
Supports multiple model versions and automatic downloading from HuggingFace Hub
or other sources.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json
import tempfile
from urllib.request import urlretrieve
from tqdm import tqdm


# Model registry - maps model names to download URLs and configs
MODEL_REGISTRY = {
    "vocos-24khz": {
        "url": "https://huggingface.co/MASSIVEMAGNETICS/sunokiller-vocos-24khz/resolve/main/model.pt",
        "config_url": "https://huggingface.co/MASSIVEMAGNETICS/sunokiller-vocos-24khz/resolve/main/config.json",
        "description": "Vocos vocoder trained at 24kHz",
        "sample_rate": 24000,
    },
    "diffusion-base": {
        "url": "https://huggingface.co/MASSIVEMAGNETICS/sunokiller-diffusion-base/resolve/main/model.pt",
        "config_url": "https://huggingface.co/MASSIVEMAGNETICS/sunokiller-diffusion-base/resolve/main/config.json",
        "description": "Base diffusion model for audio refinement",
        "num_steps": 50,
    },
    "text-to-music-base": {
        "url": "https://huggingface.co/MASSIVEMAGNETICS/sunokiller-text-to-music-base/resolve/main/model.pt",
        "config_url": "https://huggingface.co/MASSIVEMAGNETICS/sunokiller-text-to-music-base/resolve/main/config.json",
        "description": "Base text-to-music transformer model",
        "dim": 768,
        "num_layers": 12,
    },
    "text-to-music-large": {
        "url": "https://huggingface.co/MASSIVEMAGNETICS/sunokiller-text-to-music-large/resolve/main/model.pt",
        "config_url": "https://huggingface.co/MASSIVEMAGNETICS/sunokiller-text-to-music-large/resolve/main/config.json",
        "description": "Large text-to-music transformer model for best quality",
        "dim": 1024,
        "num_layers": 24,
    },
}


def get_cache_dir() -> Path:
    """Get the cache directory for storing downloaded models."""
    cache_dir = os.environ.get(
        "SUNOKILLER_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "sunokiller")
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir)


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _is_nonempty_file(path: Path) -> bool:
    """Return whether path is a regular, non-empty cached artifact."""
    try:
        path = Path(path)
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _write_json_atomically(path: Path, data: Dict[str, Any]) -> None:
    """Write fallback configuration without exposing a partial cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """Download to a same-directory temporary file, then atomically publish it.

    A failed or empty transfer never replaces an existing cache entry and
    never leaves a partial file at the canonical cache path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".part",
        dir=str(output_path.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=description) as t:
            urlretrieve(url, temp_path, reporthook=t.update_to)
        if not _is_nonempty_file(temp_path):
            raise OSError(f"Downloaded file is empty or missing: {url}")
        os.replace(temp_path, output_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

def list_available_models() -> Dict[str, Dict[str, Any]]:
    """List all available pre-trained models."""
    return MODEL_REGISTRY.copy()


def download_pretrained_model(
    model_name: str,
    force_download: bool = False,
) -> Dict[str, Path]:
    """
    Download a pre-trained model and its configuration.
    
    Args:
        model_name: Name of the model from MODEL_REGISTRY
        force_download: Force re-download even if cached
        
    Returns:
        Dictionary with paths to model and config files
        
    Example:
        >>> paths = download_pretrained_model("vocos-24khz")
        >>> model = torch.load(paths["model"])
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {available}"
        )
    
    model_info = MODEL_REGISTRY[model_name]
    cache_dir = get_cache_dir()
    model_dir = cache_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "model.pt"
    config_path = model_dir / "config.json"
    
    # Download model weights if needed
    if not _is_nonempty_file(model_path) or force_download:
        print(f"Downloading {model_name} model weights...")
        try:
            download_file(
                model_info["url"],
                model_path,
                description=f"Downloading {model_name}"
            )
        except Exception as e:
            print(f"Warning: Could not download model weights: {e}")
            print("Models will be initialized with random weights.")
            # Leave the canonical cache path absent so a later call can retry.
    
    # Download config if needed
    if "config_url" in model_info and (not _is_nonempty_file(config_path) or force_download):
        print(f"Downloading {model_name} config...")
        try:
            download_file(
                model_info["config_url"],
                config_path,
                description=f"Downloading {model_name} config"
            )
        except Exception as e:
            print(f"Warning: Could not download config: {e}")
            # Preserve a previously valid cache during failed forced refreshes.
            if not _is_nonempty_file(config_path):
                config = {k: v for k, v in model_info.items() if k not in ["url", "config_url"]}
                _write_json_atomically(config_path, config)
    
    return {
        "model": model_path,
        "config": config_path,
    }


def load_pretrained_weights(
    model: torch.nn.Module,
    model_name: str,
    strict: bool = True,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load pre-trained weights into a model.
    
    Args:
        model: Model instance to load weights into
        model_name: Name of pre-trained model
        strict: Whether to strictly enforce that keys match
        device: Device to load model onto
        
    Returns:
        Model with loaded weights
        
    Example:
        >>> from sunokiller.models import VocosVocoder
        >>> vocoder = VocosVocoder()
        >>> vocoder = load_pretrained_weights(vocoder, "vocos-24khz")
    """
    paths = download_pretrained_model(model_name)
    
    # A failed first download leaves no canonical file; retain the existing
    # explicit random-initialization fallback without poisoning future retries.
    if not _is_nonempty_file(paths["model"]):
        print(f"Warning: No pre-trained weights available for {model_name}")
        print("Using randomly initialized weights.")
        return model
    
    # Load weights
    try:
        state_dict = torch.load(paths["model"], map_location=device)
        
        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        model.load_state_dict(state_dict, strict=strict)
        print(f"Successfully loaded pre-trained weights for {model_name}")
    except Exception as e:
        print(f"Warning: Failed to load weights: {e}")
        print("Using randomly initialized weights.")
    
    return model


def load_config(model_name: str) -> Dict[str, Any]:
    """
    Load configuration for a pre-trained model.
    
    Args:
        model_name: Name of pre-trained model
        
    Returns:
        Configuration dictionary
    """
    paths = download_pretrained_model(model_name)
    
    if paths["config"].exists():
        with open(paths["config"]) as f:
            return json.load(f)
    else:
        # Return defaults from registry
        return {k: v for k, v in MODEL_REGISTRY[model_name].items() 
                if k not in ["url", "config_url"]}


def create_model_from_pretrained(
    model_name: str,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Create a model and load pre-trained weights.
    
    Args:
        model_name: Name of pre-trained model
        device: Device to load model onto
        
    Returns:
        Model with pre-trained weights
        
    Example:
        >>> vocoder = create_model_from_pretrained("vocos-24khz")
    """
    from .models import VocosVocoder, DiffusionModel, TextToMusicModel
    
    config = load_config(model_name)
    
    # Determine model type from name
    if "vocos" in model_name.lower():
        model = VocosVocoder(**{k: v for k, v in config.items() 
                               if k not in ["description", "sample_rate", "url", "config_url"]})
    elif "diffusion" in model_name.lower():
        model = DiffusionModel(**{k: v for k, v in config.items() 
                                 if k not in ["description", "url", "config_url"]})
    elif "text-to-music" in model_name.lower():
        model = TextToMusicModel(**{k: v for k, v in config.items() 
                                   if k not in ["description", "url", "config_url"]})
    else:
        raise ValueError(f"Cannot determine model type from name: {model_name}")
    
    # Load pre-trained weights
    model = load_pretrained_weights(model, model_name, device=device)
    model = model.to(device)
    model.eval()
    
    return model
