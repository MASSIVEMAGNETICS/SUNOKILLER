"""Convert SUNOKILLER models to ONNX for Android deployment.

This script converts PyTorch models to ONNX format optimized for
Android devices using ONNX Runtime.
"""

import argparse
import torch
from pathlib import Path
import sys

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("Error: onnx or onnxruntime not installed")
    print("Install with: pip install onnx onnxruntime")
    sys.exit(1)

from sunokiller.models import VocosVocoder, DiffusionModel, TextToMusicModel


def convert_vocos_to_onnx(
    output_path: str,
    dim: int = 512,
    num_layers: int = 8,
    quantize: bool = True,
):
    """Convert Vocos vocoder to ONNX."""
    print("Converting Vocos vocoder to ONNX...")
    
    # Create model
    model = VocosVocoder(
        input_channels=80,
        dim=dim,
        num_layers=num_layers,
    )
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 80, 256)
    
    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["mel_spectrogram"],
        output_names=["audio"],
        dynamic_axes={
            "mel_spectrogram": {2: "time"},
            "audio": {1: "samples"},
        },
    )
    
    print(f"Saved ONNX model to {output_path}")
    
    # Quantize if requested
    if quantize:
        quantized_path = output_path.parent / (output_path.stem + "_quantized.onnx")
        print(f"Quantizing model...")
        
        quantize_dynamic(
            str(output_path),
            str(quantized_path),
            weight_type=QuantType.QUInt8,
        )
        
        print(f"Saved quantized model to {quantized_path}")
        print(f"Original size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"Quantized size: {quantized_path.stat().st_size / 1024 / 1024:.2f} MB")


def convert_text_to_music_to_onnx(
    output_path: str,
    dim: int = 384,
    num_layers: int = 6,
    quantize: bool = True,
):
    """Convert Text-to-Music model to ONNX."""
    print("Converting Text-to-Music model to ONNX...")
    print("Note: Using smaller model for mobile deployment")
    
    # Create smaller model
    model = TextToMusicModel(
        dim=dim,
        num_layers=num_layers,
        num_heads=6,
        text_encoder_name="none",
    )
    model.eval()
    
    # For mobile, we export the generation part
    # Text encoding should be done separately or on server
    
    print("Text-to-Music model ONNX export requires custom implementation")
    print("Consider hybrid approach: server-side text encoding + on-device synthesis")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SUNOKILLER models to ONNX for Android"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["vocos", "text-to-music"],
        help="Type of model to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mobile/android/app/src/main/assets",
        help="Output directory for ONNX models",
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
        help="Disable quantization",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model_type == "vocos":
        output_path = output_dir / "vocos.onnx"
        convert_vocos_to_onnx(
            str(output_path),
            dim=args.dim,
            num_layers=args.num_layers,
            quantize=not args.no_quantize,
        )
    elif args.model_type == "text-to-music":
        output_path = output_dir / "text_to_music.onnx"
        convert_text_to_music_to_onnx(
            str(output_path),
            dim=args.dim,
            num_layers=args.num_layers,
            quantize=not args.no_quantize,
        )
    
    print("\nConversion complete!")
    print("\nNext steps:")
    print("1. Add ONNX Runtime to your build.gradle")
    print("2. Copy models to app/src/main/assets/")
    print("3. Use SUNOKILLERWrapper.kt for easy integration")
    print("4. See mobile/android/app/ for a complete example")


if __name__ == "__main__":
    main()
