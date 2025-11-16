"""
Basic usage example for SUNOKILLER.

This script demonstrates how to generate music from text descriptions.
"""

from sunokiller import AudioSynthesizer
from sunokiller.utils import get_device

def main():
    print("SUNOKILLER - Basic Usage Example")
    print("=" * 50)
    
    # Get the best available device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Initialize the synthesizer
    print("\nInitializing synthesizer...")
    synthesizer = AudioSynthesizer(
        device=device,
        use_quantization=True,  # Enable for faster inference
        quantization_type="fp16",  # Good balance of speed and quality
    )
    
    # Example 1: Generate upbeat pop music
    print("\n" + "=" * 50)
    print("Example 1: Generating upbeat pop music...")
    print("=" * 50)
    
    audio = synthesizer.generate_music(
        text="upbeat pop song with electric guitar, drums, and synthesizer",
        duration=5.0,
        temperature=1.0,
    )
    
    output_path = "example_pop.wav"
    synthesizer.save_audio(audio, output_path)
    print(f"✓ Saved to: {output_path}")
    
    # Example 2: Generate classical piano
    print("\n" + "=" * 50)
    print("Example 2: Generating classical piano...")
    print("=" * 50)
    
    audio = synthesizer.generate_music(
        text="gentle classical piano piece, romantic, slow tempo",
        duration=5.0,
        temperature=0.8,  # Lower temperature for more consistent output
    )
    
    output_path = "example_piano.wav"
    synthesizer.save_audio(audio, output_path)
    print(f"✓ Saved to: {output_path}")
    
    # Example 3: Generate electronic dance music
    print("\n" + "=" * 50)
    print("Example 3: Generating electronic dance music...")
    print("=" * 50)
    
    audio = synthesizer.generate_music(
        text="energetic electronic dance music with heavy bass and synthesizers",
        duration=5.0,
        temperature=1.2,  # Higher temperature for more variation
    )
    
    output_path = "example_edm.wav"
    synthesizer.save_audio(audio, output_path)
    print(f"✓ Saved to: {output_path}")
    
    print("\n" + "=" * 50)
    print("All examples generated successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
