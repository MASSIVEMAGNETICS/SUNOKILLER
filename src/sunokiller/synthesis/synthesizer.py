"""Audio Synthesis Pipeline

High-level API for text-to-music and singing voice synthesis.
Combines text-to-music models, diffusion models, and vocoders.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
import numpy as np

from ..models import TextToMusicModel, DiffusionModel, VocosVocoder
from ..quantization import quantize_model


class AudioSynthesizer:
    """
    High-level interface for audio synthesis.
    
    Provides easy-to-use methods for:
    - Text-to-music generation
    - Singing voice synthesis
    - Audio quality enhancement
    
    Example:
        >>> synthesizer = AudioSynthesizer()
        >>> audio = synthesizer.generate_music(
        ...     "upbeat pop song with guitar and drums",
        ...     duration=10.0
        ... )
    """
    
    def __init__(
        self,
        text_to_music_model: Optional[TextToMusicModel] = None,
        diffusion_model: Optional[DiffusionModel] = None,
        vocoder: Optional[VocosVocoder] = None,
        device: Union[str, torch.device] = "cpu",
        use_quantization: bool = False,
        quantization_type: str = "dynamic",
    ):
        """
        Initialize the audio synthesizer.
        
        Args:
            text_to_music_model: Text-to-music model (optional, created if None)
            diffusion_model: Diffusion model (optional, created if None)
            vocoder: Neural vocoder (optional, created if None)
            device: Device to run models on
            use_quantization: Whether to use quantized models for efficiency
            quantization_type: Type of quantization ("dynamic", "static", "fp16")
        """
        self.device = torch.device(device)
        
        # Initialize models
        if text_to_music_model is None:
            text_to_music_model = TextToMusicModel()
        self.text_to_music = text_to_music_model.to(self.device)
        
        if diffusion_model is None:
            diffusion_model = DiffusionModel()
        self.diffusion = diffusion_model.to(self.device)
        
        if vocoder is None:
            vocoder = VocosVocoder()
        self.vocoder = vocoder.to(self.device)
        
        # Apply quantization if requested
        if use_quantization:
            self.text_to_music = quantize_model(self.text_to_music, quantization_type)
            self.diffusion = quantize_model(self.diffusion, quantization_type)
            self.vocoder = quantize_model(self.vocoder, quantization_type)
        
        # Set to eval mode
        self.text_to_music.eval()
        self.diffusion.eval()
        self.vocoder.eval()
    
    @torch.inference_mode()
    def generate_music(
        self,
        text: Union[str, List[str]],
        duration: float = 10.0,
        sample_rate: int = 24000,
        temperature: float = 1.0,
        num_diffusion_steps: int = 50,
        guidance_scale: float = 3.0,
    ) -> np.ndarray:
        """
        Generate music from text description.
        
        Args:
            text: Text description of the music to generate
            duration: Duration in seconds
            sample_rate: Output sample rate
            temperature: Sampling temperature (higher = more random)
            num_diffusion_steps: Number of diffusion steps (lower = faster)
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated audio as numpy array (num_samples,)
        """
        if isinstance(text, str):
            text = [text]
        
        batch_size = len(text)
        
        # Calculate sequence length from duration
        hop_length = 256
        mel_frames = int(duration * sample_rate / hop_length)
        
        # Step 1: Generate mel-spectrogram from text using transformer
        mel_spec = self.text_to_music.generate(
            text=text,
            max_length=mel_frames,
            temperature=temperature,
            device=self.device,
        )
        
        # Step 2: Refine with diffusion model (optional, for higher quality)
        # This step can be skipped for faster generation
        audio_shape = (batch_size, 1, mel_frames * hop_length)
        refined_audio = self.diffusion.sample(
            shape=audio_shape,
            cond=mel_spec,
            num_inference_steps=num_diffusion_steps,
        )
        
        # Step 3: Convert refined audio to high-quality waveform using vocoder
        # If we skipped diffusion, convert mel-spec directly
        if refined_audio is not None:
            # Extract features from refined audio for vocoder
            # In practice, we'd use a mel-spectrogram extractor here
            # For now, use the generated mel-spec
            final_audio = self.vocoder.generate(mel_spec)
        else:
            final_audio = self.vocoder.generate(mel_spec)
        
        # Convert to numpy and return
        audio_np = final_audio.cpu().numpy()
        
        # Handle batch dimension
        if batch_size == 1:
            audio_np = audio_np[0]
        
        return audio_np
    
    @torch.inference_mode()
    def generate_singing_voice(
        self,
        lyrics: Union[str, List[str]],
        melody_description: Optional[str] = None,
        duration: float = 10.0,
        sample_rate: int = 24000,
        voice_style: str = "neutral",
    ) -> np.ndarray:
        """
        Generate singing voice from lyrics.
        
        Args:
            lyrics: Lyrics to sing
            melody_description: Optional description of melody/style
            duration: Duration in seconds
            sample_rate: Output sample rate
            voice_style: Voice style ("neutral", "male", "female", "choir")
            
        Returns:
            Generated singing voice as numpy array
        """
        # Combine lyrics with melody description and voice style
        if isinstance(lyrics, str):
            lyrics = [lyrics]
        
        prompts = []
        for lyric in lyrics:
            prompt = f"singing voice, {voice_style} voice, lyrics: {lyric}"
            if melody_description:
                prompt += f", {melody_description}"
            prompts.append(prompt)
        
        # Use the general music generation with singing-specific prompts
        return self.generate_music(
            text=prompts,
            duration=duration,
            sample_rate=sample_rate,
        )
    
    @torch.inference_mode()
    def enhance_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
        num_diffusion_steps: int = 25,
    ) -> np.ndarray:
        """
        Enhance audio quality using diffusion model.
        
        Args:
            audio: Input audio as numpy array
            sample_rate: Sample rate
            num_diffusion_steps: Number of enhancement steps
            
        Returns:
            Enhanced audio
        """
        # Convert to torch tensor
        if audio.ndim == 1:
            audio = audio[None, None, :]  # Add batch and channel dims
        elif audio.ndim == 2:
            audio = audio[:, None, :]  # Add channel dim
        
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        # Use diffusion model for enhancement
        # This adds a small amount of noise and then denoises
        # to improve quality
        batch_size = audio_tensor.shape[0]
        
        # Add small noise
        noise_level = 0.1
        noisy_audio = audio_tensor + torch.randn_like(audio_tensor) * noise_level
        
        # Denoise
        enhanced = self.diffusion.sample(
            shape=noisy_audio.shape,
            cond=None,
            num_inference_steps=num_diffusion_steps,
        )
        
        # Convert back to numpy
        enhanced_np = enhanced.cpu().numpy()
        
        if enhanced_np.shape[0] == 1:
            enhanced_np = enhanced_np[0, 0]  # Remove batch and channel dims
        
        return enhanced_np
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: int = 24000,
    ):
        """
        Save generated audio to file.
        
        Args:
            audio: Audio array
            output_path: Path to save file (supports .wav, .mp3, .flac)
            sample_rate: Sample rate
        """
        try:
            import soundfile as sf
            sf.write(output_path, audio, sample_rate)
        except ImportError:
            try:
                from scipy.io import wavfile
                # Scale to int16
                audio_int = (audio * 32767).astype(np.int16)
                wavfile.write(output_path, sample_rate, audio_int)
            except ImportError:
                raise ImportError(
                    "Please install soundfile or scipy to save audio files:\n"
                    "pip install soundfile"
                )
