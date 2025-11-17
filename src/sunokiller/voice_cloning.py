"""Voice Cloning Module

Implements voice cloning from reference audio samples.
Supports:
- Few-shot voice cloning (1-5 samples)
- Speaker embedding extraction
- Style transfer for singing voice
"""

import torch
import torch.nn as nn
import torchaudio
from typing import List, Optional, Union
import numpy as np
from pathlib import Path


class SpeakerEncoder(nn.Module):
    """Speaker encoder for extracting voice embeddings."""
    
    def __init__(
        self,
        mel_bins: int = 80,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        
        # LSTM for processing mel-spectrograms
        self.lstm = nn.LSTM(
            mel_bins,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        # Projection to embedding space
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel-spectrogram.
        
        Args:
            mel_spec: (batch, mel_bins, time) mel-spectrogram
            
        Returns:
            embedding: (batch, embedding_dim) speaker embedding
        """
        # Transpose for LSTM: (batch, time, mel_bins)
        x = mel_spec.transpose(1, 2)
        
        # Process with LSTM
        output, (hidden, _) = self.lstm(x)
        
        # Use last hidden state from both directions
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Project to embedding
        embedding = self.projection(combined)
        
        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding


class VoiceCloner:
    """
    Voice cloning system for generating audio in a target voice.
    
    Supports few-shot learning from reference audio samples.
    
    Example:
        >>> cloner = VoiceCloner()
        >>> # Extract voice embedding from reference
        >>> embedding = cloner.extract_voice_embedding("reference.wav")
        >>> # Generate audio with that voice
        >>> audio = cloner.clone_voice(
        ...     text="Hello world",
        ...     reference_embedding=embedding,
        ... )
    """
    
    def __init__(
        self,
        speaker_encoder: Optional[SpeakerEncoder] = None,
        device: str = "cpu",
    ):
        """
        Initialize voice cloner.
        
        Args:
            speaker_encoder: Pre-trained speaker encoder (optional)
            device: Device to run on
        """
        self.device = torch.device(device)
        
        if speaker_encoder is None:
            speaker_encoder = SpeakerEncoder()
        
        self.speaker_encoder = speaker_encoder.to(self.device)
        self.speaker_encoder.eval()
        
        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
        )
    
    def load_audio(self, audio_path: str, sample_rate: int = 24000) -> torch.Tensor:
        """Load and preprocess audio file."""
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return audio
    
    @torch.inference_mode()
    def extract_voice_embedding(
        self,
        audio_path: Union[str, List[str]],
        sample_rate: int = 24000,
    ) -> torch.Tensor:
        """
        Extract voice embedding from reference audio.
        
        Args:
            audio_path: Path to audio file or list of paths for multi-sample
            sample_rate: Audio sample rate
            
        Returns:
            embedding: (embedding_dim,) voice embedding vector
        """
        if isinstance(audio_path, str):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path
        
        embeddings = []
        
        for path in audio_paths:
            # Load audio
            audio = self.load_audio(path, sample_rate)
            audio = audio.to(self.device)
            
            # Compute mel-spectrogram
            mel = self.mel_transform(audio)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            
            # Extract embedding
            embedding = self.speaker_encoder(mel)
            embeddings.append(embedding)
        
        # Average embeddings if multiple samples
        if len(embeddings) > 1:
            embedding = torch.stack(embeddings).mean(dim=0)
            # Re-normalize
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        else:
            embedding = embeddings[0]
        
        return embedding.squeeze(0)  # Remove batch dimension
    
    def clone_voice(
        self,
        text: str,
        reference_embedding: torch.Tensor,
        synthesizer=None,
        duration: float = 10.0,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """
        Generate audio in the reference voice.
        
        Args:
            text: Text to synthesize
            reference_embedding: Voice embedding from reference audio
            synthesizer: AudioSynthesizer instance (optional)
            duration: Duration in seconds
            sample_rate: Output sample rate
            
        Returns:
            Generated audio as numpy array
        """
        if synthesizer is None:
            from ..synthesis import AudioSynthesizer
            synthesizer = AudioSynthesizer(device=self.device)
        
        # TODO: Integrate speaker embedding into synthesis pipeline
        # For now, use standard synthesis with voice style hints
        
        # Add voice characteristics to prompt
        enhanced_text = f"{text} [voice_style: custom]"
        
        # Generate with synthesizer
        audio = synthesizer.generate_singing_voice(
            lyrics=text,
            voice_style="custom",
            duration=duration,
        )
        
        return audio
    
    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> float:
        """
        Compute similarity between two voice embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Cosine similarity
        similarity = torch.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0),
        )
        return similarity.item()


def extract_voice_from_file(
    audio_path: Union[str, List[str]],
    output_path: Optional[str] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Utility function to extract and save voice embedding.
    
    Args:
        audio_path: Path(s) to reference audio
        output_path: Optional path to save embedding
        device: Device to use
        
    Returns:
        Voice embedding tensor
    """
    cloner = VoiceCloner(device=device)
    embedding = cloner.extract_voice_embedding(audio_path)
    
    if output_path:
        torch.save({"embedding": embedding}, output_path)
        print(f"Saved voice embedding to {output_path}")
    
    return embedding


def clone_voice_from_file(
    text: str,
    reference_audio: Union[str, torch.Tensor],
    output_path: str,
    device: str = "cpu",
):
    """
    End-to-end voice cloning from file.
    
    Args:
        text: Text to synthesize
        reference_audio: Path to reference audio or pre-computed embedding
        output_path: Where to save generated audio
        device: Device to use
    """
    cloner = VoiceCloner(device=device)
    
    # Extract or load embedding
    if isinstance(reference_audio, str):
        if reference_audio.endswith('.pt'):
            # Load pre-computed embedding
            checkpoint = torch.load(reference_audio)
            embedding = checkpoint["embedding"]
        else:
            # Extract from audio
            embedding = cloner.extract_voice_embedding(reference_audio)
    else:
        embedding = reference_audio
    
    # Generate audio
    audio = cloner.clone_voice(text, embedding)
    
    # Save
    import soundfile as sf
    sf.write(output_path, audio, 24000)
    print(f"Saved cloned voice to {output_path}")
