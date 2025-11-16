"""
Singing voice synthesis example.

Demonstrates how to generate singing voices with different styles.
"""

from sunokiller import AudioSynthesizer
from sunokiller.utils import get_device

def main():
    print("SUNOKILLER - Singing Voice Synthesis Example")
    print("=" * 50)
    
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Initialize synthesizer
    print("\nInitializing synthesizer...")
    synthesizer = AudioSynthesizer(
        device=device,
        use_quantization=True,
        quantization_type="fp16",
    )
    
    # Example 1: Female voice singing a lullaby
    print("\n" + "=" * 50)
    print("Example 1: Female voice - Lullaby")
    print("=" * 50)
    
    audio = synthesizer.generate_singing_voice(
        lyrics="Twinkle twinkle little star, how I wonder what you are",
        voice_style="female",
        melody_description="gentle lullaby, soft and calming",
        duration=8.0,
    )
    
    synthesizer.save_audio(audio, "singing_lullaby.wav")
    print("✓ Saved to: singing_lullaby.wav")
    
    # Example 2: Male voice singing pop
    print("\n" + "=" * 50)
    print("Example 2: Male voice - Pop song")
    print("=" * 50)
    
    audio = synthesizer.generate_singing_voice(
        lyrics="I wanna dance with somebody, I wanna feel the heat with somebody",
        voice_style="male",
        melody_description="upbeat pop, energetic",
        duration=10.0,
    )
    
    synthesizer.save_audio(audio, "singing_pop.wav")
    print("✓ Saved to: singing_pop.wav")
    
    # Example 3: Choir singing
    print("\n" + "=" * 50)
    print("Example 3: Choir - Hymn")
    print("=" * 50)
    
    audio = synthesizer.generate_singing_voice(
        lyrics="Amazing grace, how sweet the sound",
        voice_style="choir",
        melody_description="traditional hymn, harmonious",
        duration=8.0,
    )
    
    synthesizer.save_audio(audio, "singing_choir.wav")
    print("✓ Saved to: singing_choir.wav")
    
    print("\n" + "=" * 50)
    print("All singing examples generated successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
