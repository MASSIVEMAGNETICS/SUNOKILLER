"""Convert SUNOKILLER models to CoreML for iOS deployment.

This script converts PyTorch models to CoreML format optimized for
iOS devices, including iPhone and iPad.
"""

import argparse
import torch
from pathlib import Path
import sys

try:
    import coremltools as ct
except ImportError:
    print("Error: coremltools not installed")
    print("Install with: pip install coremltools")
    sys.exit(1)

from sunokiller.models import VocosVocoder, DiffusionModel, TextToMusicModel


def convert_vocos_to_coreml(
    output_path: str,
    dim: int = 512,
    num_layers: int = 8,
    quantize: bool = True,
):
    """Convert Vocos vocoder to CoreML."""
    print("Converting Vocos vocoder to CoreML...")
    
    # Create model
    model = VocosVocoder(
        input_channels=80,
        dim=dim,
        num_layers=num_layers,
    )
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 80, 256)
    
    # Trace model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="mel_spectrogram", shape=(1, 80, ct.RangeDim(1, 1024)))],
        outputs=[ct.TensorType(name="audio")],
        compute_precision=ct.precision.FLOAT16 if quantize else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS15,
    )
    
    # Set metadata
    mlmodel.short_description = "Vocos neural vocoder for high-quality audio synthesis"
    mlmodel.author = "MASSIVEMAGNETICS"
    mlmodel.license = "MIT"
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    
    print(f"Saved CoreML model to {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def convert_diffusion_to_coreml(
    output_path: str,
    num_steps: int = 50,
    quantize: bool = True,
):
    """Convert Diffusion model to CoreML."""
    print("Converting Diffusion model to CoreML...")
    print("Warning: Diffusion models may be too large for mobile deployment")
    print("Consider using a smaller variant or skipping diffusion on mobile")
    
    # Diffusion model is complex and may not convert well
    # This is a placeholder for future implementation
    print("Diffusion model conversion not yet implemented for CoreML")
    print("Use Vocos-only pipeline for mobile deployment")


def convert_text_to_music_to_coreml(
    output_path: str,
    dim: int = 384,
    num_layers: int = 6,
    quantize: bool = True,
):
    """Convert Text-to-Music model to CoreML."""
    print("Converting Text-to-Music model to CoreML...")
    print("Note: Using smaller model for mobile deployment")
    
    # Create smaller model for mobile
    model = TextToMusicModel(
        dim=dim,
        num_layers=num_layers,
        num_heads=6,
        text_encoder_name="none",  # Use simple embedding for mobile
    )
    model.eval()
    
    # For text encoding, we'll export just the generation part
    # Text encoding can be done separately
    print("Text-to-Music model conversion requires custom implementation")
    print("Consider using cloud-based text encoding with on-device synthesis")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SUNOKILLER models to CoreML for iOS"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["vocos", "diffusion", "text-to-music"],
        help="Type of model to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mobile/ios/models",
        help="Output directory for CoreML models",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=384,
        help="Model dimension (smaller for mobile)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of layers (fewer for mobile)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable FP16 quantization",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model_type == "vocos":
        output_path = output_dir / "Vocos.mlpackage"
        convert_vocos_to_coreml(
            str(output_path),
            dim=args.dim,
            num_layers=args.num_layers,
            quantize=not args.no_quantize,
        )
    elif args.model_type == "diffusion":
        output_path = output_dir / "Diffusion.mlpackage"
        convert_diffusion_to_coreml(
            str(output_path),
            quantize=not args.no_quantize,
        )
    elif args.model_type == "text-to-music":
        output_path = output_dir / "TextToMusic.mlpackage"
        convert_text_to_music_to_coreml(
            str(output_path),
            dim=args.dim,
            num_layers=args.num_layers,
            quantize=not args.no_quantize,
        )
    
    print("\nConversion complete!")
    print("\nNext steps:")
    print("1. Add the .mlpackage to your Xcode project")
    print("2. Use SUNOKILLERWrapper.swift for easy integration")
    print("3. See mobile/ios/Example/ for a complete example")


if __name__ == "__main__":
    main()
