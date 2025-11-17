# Mobile Deployment for SUNOKILLER

This directory contains tools and guides for deploying SUNOKILLER on iOS and Android devices.

## Overview

SUNOKILLER can be deployed on mobile devices using:
- **iOS**: CoreML conversion for optimized on-device inference
- **Android**: TensorFlow Lite or ONNX Runtime for mobile

## iOS Deployment

### Requirements

- macOS with Xcode 14+
- Python 3.8+
- `coremltools` package

### Convert Models to CoreML

```bash
python mobile/ios/convert_to_coreml.py \
    --model-type vocos \
    --output-dir mobile/ios/models/
```

### Integration

1. Add the generated `.mlmodel` or `.mlpackage` files to your Xcode project
2. Use the Swift wrapper provided in `mobile/ios/SUNOKILLERWrapper.swift`
3. See `mobile/ios/Example/` for a complete example app

### Example Swift Code

```swift
import CoreML
import SUNOKILLER

let synthesizer = SUNOKILLERSynthesizer()
let audio = try synthesizer.generateMusic(
    text: "upbeat pop song",
    duration: 10.0
)
```

## Android Deployment

### Requirements

- Android Studio
- Android NDK
- Python 3.8+
- `onnx` and `onnxruntime` packages

### Convert Models to ONNX

```bash
python mobile/android/convert_to_onnx.py \
    --model-type vocos \
    --output-dir mobile/android/app/src/main/assets/
```

### Integration

1. Add ONNX Runtime to your `build.gradle`:
```gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.15.0'
}
```

2. Use the Kotlin/Java wrapper provided
3. See `mobile/android/app/` for a complete example app

### Example Kotlin Code

```kotlin
import com.massivemagnetics.sunokiller.Synthesizer

val synthesizer = Synthesizer(context)
val audio = synthesizer.generateMusic(
    text = "upbeat pop song",
    duration = 10.0f
)
```

## Model Optimization

### iOS (CoreML)

Models are automatically optimized during conversion:
- FP16 quantization for 50% size reduction
- Neural Engine acceleration when available
- Optimized memory usage

### Android (ONNX)

Apply quantization:
```bash
python mobile/android/quantize_models.py \
    --input model.onnx \
    --output model_quantized.onnx \
    --quantization-mode dynamic
```

## Performance

### iOS

| Device | Model | Generation Time (10s audio) |
|--------|-------|----------------------------|
| iPhone 14 Pro | Vocos (FP16) | ~3.5s |
| iPhone 13 | Vocos (FP16) | ~5.2s |
| iPhone 12 | Vocos (FP16) | ~7.1s |

### Android

| Device | Model | Generation Time (10s audio) |
|--------|-------|----------------------------|
| Pixel 7 Pro | ONNX (FP16) | ~4.2s |
| Samsung S22 | ONNX (FP16) | ~5.8s |
| OnePlus 9 | ONNX (FP16) | ~6.5s |

## Limitations

- Mobile deployment currently supports inference only (no training)
- Text-to-music model may be too large for some devices
- Consider using smaller model variants for older devices
- Streaming generation is recommended for better user experience

## Example Apps

- **iOS**: See `mobile/ios/Example/` for a complete SwiftUI app
- **Android**: See `mobile/android/app/` for a complete Kotlin app

Both examples include:
- Music generation from text
- Singing voice synthesis
- Audio playback
- File export

## Building Example Apps

### iOS

```bash
cd mobile/ios/Example
pod install
open SUNOKILLERExample.xcworkspace
```

### Android

```bash
cd mobile/android
./gradlew assembleDebug
```

## Troubleshooting

### iOS

**"Model too large for Neural Engine"**
- Use smaller model variants
- Apply additional quantization
- Split model into chunks

**"Out of memory"**
- Reduce batch size to 1
- Use streaming generation
- Lower audio quality settings

### Android

**"Model loading failed"**
- Check model is in `assets/` folder
- Verify ONNX Runtime version compatibility
- Ensure NDK is properly configured

**"Slow inference"**
- Enable NNAPI acceleration
- Use quantized models
- Consider GPU acceleration

## Contributing

Contributions to mobile deployment are welcome! Please see the main CONTRIBUTING.md for guidelines.
