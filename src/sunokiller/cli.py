"""Command-line interface for SUNOKILLER."""

import argparse
import sys
import time
from pathlib import Path

from .synthesis import AudioSynthesizer
from .utils import get_device, format_time


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SUNOKILLER - Advanced Audio Synthesis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate music from text
  sunokiller generate "upbeat pop song with guitar" -o output.wav
  
  # Generate singing voice
  sunokiller sing "Happy birthday to you" -o birthday.wav
  
  # Enhance audio quality
  sunokiller enhance input.wav -o enhanced.wav
  
  # Use quantized models for faster inference
  sunokiller generate "jazz piano" -o jazz.wav --quantize
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate music command
    generate_parser = subparsers.add_parser("generate", help="Generate music from text")
    generate_parser.add_argument("text", type=str, help="Text description of music")
    generate_parser.add_argument("-o", "--output", type=str, required=True, help="Output audio file")
    generate_parser.add_argument("-d", "--duration", type=float, default=10.0, help="Duration in seconds")
    generate_parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sampling temperature")
    generate_parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    generate_parser.add_argument("--sample-rate", type=int, default=24000, help="Sample rate")
    
    # Singing voice command
    sing_parser = subparsers.add_parser("sing", help="Generate singing voice")
    sing_parser.add_argument("lyrics", type=str, help="Lyrics to sing")
    sing_parser.add_argument("-o", "--output", type=str, required=True, help="Output audio file")
    sing_parser.add_argument("-d", "--duration", type=float, default=10.0, help="Duration in seconds")
    sing_parser.add_argument("-s", "--style", type=str, default="neutral", 
                            choices=["neutral", "male", "female", "choir"],
                            help="Voice style")
    sing_parser.add_argument("-m", "--melody", type=str, help="Melody description")
    
    # Enhance audio command
    enhance_parser = subparsers.add_parser("enhance", help="Enhance audio quality")
    enhance_parser.add_argument("input", type=str, help="Input audio file")
    enhance_parser.add_argument("-o", "--output", type=str, required=True, help="Output audio file")
    enhance_parser.add_argument("--steps", type=int, default=25, help="Number of enhancement steps")
    
    # Common arguments
    for p in [generate_parser, sing_parser, enhance_parser]:
        p.add_argument("--device", type=str, default="auto", 
                      choices=["auto", "cpu", "cuda", "mps"],
                      help="Device to use")
        p.add_argument("--quantize", action="store_true",
                      help="Use quantized models for faster inference")
        p.add_argument("--quantize-type", type=str, default="dynamic",
                      choices=["dynamic", "static", "fp16"],
                      help="Quantization type")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = args.device
    
    print(f"SUNOKILLER - Advanced Audio Synthesis System")
    print(f"Using device: {device}")
    
    # Initialize synthesizer
    print("Loading models...")
    start_time = time.time()
    
    synthesizer = AudioSynthesizer(
        device=device,
        use_quantization=args.quantize,
        quantization_type=args.quantize_type if args.quantize else "dynamic",
    )
    
    load_time = time.time() - start_time
    print(f"Models loaded in {format_time(load_time)}")
    
    # Execute command
    if args.command == "generate":
        print(f"\nGenerating music: '{args.text}'")
        print(f"Duration: {args.duration}s")
        
        start_time = time.time()
        audio = synthesizer.generate_music(
            text=args.text,
            duration=args.duration,
            sample_rate=args.sample_rate,
            temperature=args.temperature,
            num_diffusion_steps=args.steps,
        )
        gen_time = time.time() - start_time
        
        # Save output
        synthesizer.save_audio(audio, args.output, args.sample_rate)
        print(f"\nGeneration complete in {format_time(gen_time)}")
        print(f"Output saved to: {args.output}")
        
    elif args.command == "sing":
        print(f"\nGenerating singing voice")
        print(f"Lyrics: '{args.lyrics}'")
        print(f"Style: {args.style}")
        
        start_time = time.time()
        audio = synthesizer.generate_singing_voice(
            lyrics=args.lyrics,
            melody_description=args.melody,
            duration=args.duration,
            voice_style=args.style,
        )
        gen_time = time.time() - start_time
        
        # Save output
        synthesizer.save_audio(audio, args.output)
        print(f"\nGeneration complete in {format_time(gen_time)}")
        print(f"Output saved to: {args.output}")
        
    elif args.command == "enhance":
        print(f"\nEnhancing audio: {args.input}")
        
        # Load input audio
        from .utils import load_audio
        audio, sr = load_audio(args.input)
        audio_np = audio.numpy()[0]  # Convert to numpy
        
        start_time = time.time()
        enhanced = synthesizer.enhance_audio(
            audio=audio_np,
            sample_rate=sr,
            num_diffusion_steps=args.steps,
        )
        enhance_time = time.time() - start_time
        
        # Save output
        synthesizer.save_audio(enhanced, args.output, sr)
        print(f"\nEnhancement complete in {format_time(enhance_time)}")
        print(f"Output saved to: {args.output}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
