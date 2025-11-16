"""
Model quantization example.

Demonstrates how to quantize models for efficient inference on low-end hardware.
"""

import torch
from sunokiller.models import VocosVocoder, DiffusionModel
from sunokiller.quantization import (
    quantize_model,
    export_to_onnx,
    get_model_size,
    print_quantization_summary,
)

def main():
    print("SUNOKILLER - Model Quantization Example")
    print("=" * 50)
    
    # Create a sample model (Vocos vocoder)
    print("\nCreating Vocos vocoder model...")
    model = VocosVocoder(
        input_channels=80,
        dim=512,
        num_layers=8,
    )
    
    print(f"Original model size: {get_model_size(model):.2f} MB")
    
    # Example 1: FP16 Quantization
    print("\n" + "=" * 50)
    print("Example 1: FP16 Quantization (Half Precision)")
    print("=" * 50)
    
    model_fp16 = quantize_model(
        model,
        quantization_type="fp16",
        output_path="models/vocos_fp16.pth",
    )
    
    print_quantization_summary(model, model_fp16)
    print("✓ FP16 model saved to: models/vocos_fp16.pth")
    
    # Example 2: Dynamic INT8 Quantization
    print("\n" + "=" * 50)
    print("Example 2: Dynamic INT8 Quantization")
    print("=" * 50)
    
    model_int8 = quantize_model(
        model,
        quantization_type="dynamic",
        output_path="models/vocos_int8.pth",
    )
    
    print_quantization_summary(model, model_int8)
    print("✓ INT8 model saved to: models/vocos_int8.pth")
    
    # Example 3: Export to ONNX
    print("\n" + "=" * 50)
    print("Example 3: Export to ONNX")
    print("=" * 50)
    
    try:
        onnx_path = export_to_onnx(
            model,
            output_path="models/vocos.onnx",
            input_shapes={"features": (1, 80, 256)},
            dynamic_axes={"features": {2: "time"}},
        )
        print(f"✓ ONNX model exported to: {onnx_path}")
        
        # Quantize ONNX model
        from sunokiller.quantization import quantize_onnx_model
        
        quantized_onnx = quantize_onnx_model(
            onnx_path,
            "models/vocos_quantized.onnx",
            quantization_type="dynamic",
        )
        print(f"✓ Quantized ONNX model saved to: {quantized_onnx}")
        
    except ImportError:
        print("⚠ ONNX not available. Install with: pip install onnx onnxruntime")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    
    # Create dummy input
    dummy_input = torch.randn(1, 80, 256)
    
    # Benchmark original model
    import time
    
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(dummy_input)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = model(dummy_input)
        original_time = (time.time() - start) / 10
    
    print(f"\nOriginal model:")
    print(f"  - Size: {get_model_size(model):.2f} MB")
    print(f"  - Inference time: {original_time*1000:.2f} ms")
    
    print(f"\nFP16 model:")
    print(f"  - Size: {get_model_size(model_fp16):.2f} MB")
    print(f"  - Expected speedup: 1.5-2x on GPU")
    
    print(f"\nINT8 model:")
    print(f"  - Size: {get_model_size(model_int8):.2f} MB")
    print(f"  - Expected speedup: 2-4x on CPU")
    
    print("\n" + "=" * 50)
    print("Quantization examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
