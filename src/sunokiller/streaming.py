"""Real-time Streaming Audio Generation

Enables low-latency streaming generation for interactive applications.
Uses chunked generation and optimized inference.
"""

import torch
import numpy as np
from typing import Optional, Iterator, Callable
import queue
import threading
from dataclasses import dataclass


@dataclass
class StreamingConfig:
    """Configuration for streaming generation."""
    chunk_size: int = 2048  # Samples per chunk
    overlap: int = 256  # Overlap between chunks for smoothing
    buffer_size: int = 4  # Number of chunks to buffer
    sample_rate: int = 24000
    

class StreamingGenerator:
    """
    Real-time streaming audio generator.
    
    Generates audio in small chunks for low-latency playback.
    
    Example:
        >>> generator = StreamingGenerator(synthesizer)
        >>> for chunk in generator.stream_music("upbeat pop song"):
        ...     play_audio(chunk)  # Play immediately
    """
    
    def __init__(
        self,
        synthesizer,
        config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize streaming generator.
        
        Args:
            synthesizer: AudioSynthesizer instance
            config: Streaming configuration
        """
        self.synthesizer = synthesizer
        self.config = config or StreamingConfig()
        
        # Audio buffer for smooth transitions
        self.buffer = queue.Queue(maxsize=self.config.buffer_size)
        
    def stream_music(
        self,
        text: str,
        duration: Optional[float] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> Iterator[np.ndarray]:
        """
        Stream music generation chunk by chunk.
        
        Args:
            text: Text description of music
            duration: Total duration (None for continuous)
            callback: Optional callback for each chunk
            
        Yields:
            Audio chunks as numpy arrays
        """
        # Start generation in background thread
        generation_thread = threading.Thread(
            target=self._generate_chunks,
            args=(text, duration),
            daemon=True,
        )
        generation_thread.start()
        
        # Yield chunks as they become available
        while True:
            try:
                chunk = self.buffer.get(timeout=1.0)
                
                if chunk is None:  # End signal
                    break
                
                if callback:
                    callback(chunk)
                
                yield chunk
                
            except queue.Empty:
                if not generation_thread.is_alive():
                    break
        
        generation_thread.join()
    
    def _generate_chunks(self, text: str, duration: Optional[float]):
        """Background thread for chunk generation."""
        # Calculate total samples
        if duration:
            total_samples = int(duration * self.config.sample_rate)
        else:
            total_samples = None
        
        # Generate mel-spectrogram frames incrementally
        # This is a simplified version - in practice, you'd use
        # streaming transformer generation
        
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        
        # For demonstration, generate full audio then chunk it
        # In production, use true streaming generation
        audio = self.synthesizer.generate_music(
            text=text,
            duration=duration or 10.0,
        )
        
        # Split into overlapping chunks
        pos = 0
        while pos < len(audio):
            # Get chunk with overlap
            end = min(pos + chunk_size, len(audio))
            chunk = audio[pos:end]
            
            # Apply fade in/out for smooth transitions
            if pos > 0:  # Not first chunk
                fade_in = np.linspace(0, 1, overlap)
                chunk[:overlap] *= fade_in
            
            if end < len(audio):  # Not last chunk
                fade_out = np.linspace(1, 0, overlap)
                chunk[-overlap:] *= fade_out
            
            # Put in buffer
            try:
                self.buffer.put(chunk, timeout=1.0)
            except queue.Full:
                # Wait for space in buffer
                continue
            
            # Move to next chunk (with overlap)
            pos += chunk_size - overlap
        
        # Signal end of generation
        self.buffer.put(None)
    
    def stream_singing(
        self,
        lyrics: str,
        voice_style: str = "neutral",
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> Iterator[np.ndarray]:
        """
        Stream singing voice generation.
        
        Args:
            lyrics: Lyrics to sing
            voice_style: Voice style
            callback: Optional callback for each chunk
            
        Yields:
            Audio chunks
        """
        # Similar to stream_music but for singing
        for chunk in self.stream_music(
            text=f"singing voice, {voice_style}, lyrics: {lyrics}",
            callback=callback,
        ):
            yield chunk


class RealTimeProcessor:
    """
    Real-time audio processor for live applications.
    
    Processes audio with minimal latency using circular buffers
    and chunked processing.
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_size: int = 1024,
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        
        # Circular buffer for input
        self.input_buffer = np.zeros(chunk_size * 4)
        self.buffer_pos = 0
        
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process a chunk of audio in real-time.
        
        Args:
            audio_chunk: Input audio chunk
            
        Returns:
            Processed audio chunk
        """
        # Add to buffer
        chunk_len = len(audio_chunk)
        
        # Circular buffer implementation
        if self.buffer_pos + chunk_len > len(self.input_buffer):
            # Wrap around
            first_part = len(self.input_buffer) - self.buffer_pos
            self.input_buffer[self.buffer_pos:] = audio_chunk[:first_part]
            self.input_buffer[:chunk_len - first_part] = audio_chunk[first_part:]
            self.buffer_pos = chunk_len - first_part
        else:
            self.input_buffer[self.buffer_pos:self.buffer_pos + chunk_len] = audio_chunk
            self.buffer_pos += chunk_len
        
        # Process (placeholder - add your processing here)
        processed = audio_chunk.copy()
        
        return processed
    
    def stream_process(
        self,
        input_stream: Iterator[np.ndarray],
    ) -> Iterator[np.ndarray]:
        """
        Stream process audio chunks.
        
        Args:
            input_stream: Iterator yielding audio chunks
            
        Yields:
            Processed audio chunks
        """
        for chunk in input_stream:
            processed = self.process_chunk(chunk)
            yield processed


def create_streaming_synthesizer(
    device: str = "cpu",
    use_quantization: bool = True,
) -> StreamingGenerator:
    """
    Create a streaming synthesizer optimized for low latency.
    
    Args:
        device: Device to use
        use_quantization: Use quantized models for speed
        
    Returns:
        StreamingGenerator instance
    """
    from ..synthesis import AudioSynthesizer
    
    # Create synthesizer with optimizations
    synthesizer = AudioSynthesizer(
        device=device,
        use_quantization=use_quantization,
        quantization_type="fp16" if device == "cuda" else "dynamic",
    )
    
    # Configure for low latency
    config = StreamingConfig(
        chunk_size=1024,  # Smaller chunks for lower latency
        overlap=128,
        buffer_size=2,  # Smaller buffer
    )
    
    return StreamingGenerator(synthesizer, config)


# Example usage functions

def example_streaming_to_file(output_path: str = "streamed_output.wav"):
    """Example: Stream generation and save to file."""
    import soundfile as sf
    
    generator = create_streaming_synthesizer()
    
    chunks = []
    for chunk in generator.stream_music("upbeat electronic music", duration=10.0):
        chunks.append(chunk)
        print(f"Generated chunk of {len(chunk)} samples")
    
    # Combine and save
    full_audio = np.concatenate(chunks)
    sf.write(output_path, full_audio, 24000)
    print(f"Saved to {output_path}")


def example_streaming_to_audio_device():
    """Example: Stream directly to audio output device."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Install sounddevice for real-time playback: pip install sounddevice")
        return
    
    generator = create_streaming_synthesizer()
    
    def play_chunk(chunk):
        """Play audio chunk immediately."""
        sd.play(chunk, 24000)
        sd.wait()
    
    # Stream with playback callback
    for chunk in generator.stream_music(
        "calm piano melody",
        duration=10.0,
        callback=play_chunk,
    ):
        pass  # Callback handles playback
