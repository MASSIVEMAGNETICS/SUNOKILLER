# SUNOKILLER Complete Feature Guide

This document provides a comprehensive guide to all features implemented in SUNOKILLER.

## Table of Contents

1. [Core Features](#core-features)
2. [Pre-trained Models](#pre-trained-models)
3. [Training](#training)
4. [Voice Cloning](#voice-cloning)
5. [Real-time Streaming](#real-time-streaming)
6. [Web UI](#web-ui)
7. [Mobile Deployment](#mobile-deployment)
8. [VST/AU Plugin](#vstau-plugin)

## Core Features

### Text-to-Music Generation

Generate music from text descriptions:

```python
from sunokiller import AudioSynthesizer

synthesizer = AudioSynthesizer(device="cuda", use_quantization=True)

audio = synthesizer.generate_music(
    text="upbeat pop song with electric guitar and drums",
    duration=10.0,
    temperature=1.0,
)

synthesizer.save_audio(audio, "output.wav")
```

### Singing Voice Synthesis

Generate singing voices from lyrics:

```python
audio = synthesizer.generate_singing_voice(
    lyrics="Happy birthday to you, happy birthday to you",
    voice_style="female",
    melody_description="gentle and warm",
    duration=10.0,
)
```

### Audio Enhancement

Improve audio quality using diffusion models:

```python
enhanced = synthesizer.enhance_audio(
    audio=audio_input,
    sample_rate=24000,
    num_diffusion_steps=25,
)
```

## Pre-trained Models

### Loading Pre-trained Weights

```python
from sunokiller.pretrained import create_model_from_pretrained

# Load a pre-trained vocoder
vocoder = create_model_from_pretrained("vocos-24khz", device="cuda")

# List available models
from sunokiller.pretrained import list_available_models
models = list_available_models()
for name, info in models.items():
    print(f"{name}: {info['description']}")
```

### Available Models

- `vocos-24khz`: Vocos vocoder trained at 24kHz
- `diffusion-base`: Base diffusion model
- `text-to-music-base`: Base text-to-music transformer
- `text-to-music-large`: Large model for best quality

## Training

### Training Vocos Vocoder

```bash
python training/train_vocos.py \
    --data-dir /path/to/audio \
    --output-dir outputs/vocos \
    --batch-size 16 \
    --num-epochs 100 \
    --use-wandb
```

### Training Text-to-Music Model

Prepare your data:
```json
{
  "song1.wav": "upbeat pop song with guitar",
  "song2.wav": "calm piano melody"
}
```

Train:
```bash
python training/train_text_to_music.py \
    --data-dir /path/to/dataset \
    --output-dir outputs/text_to_music \
    --batch-size 8 \
    --num-epochs 100
```

See `training/README.md` for detailed training instructions.

## Voice Cloning

### Extract Voice Embedding

```python
from sunokiller.voice_cloning import VoiceCloner

cloner = VoiceCloner(device="cuda")

# Extract from single reference
embedding = cloner.extract_voice_embedding("reference.wav")

# Extract from multiple samples (better quality)
embedding = cloner.extract_voice_embedding([
    "ref1.wav",
    "ref2.wav",
    "ref3.wav",
])

# Save for later use
import torch
torch.save({"embedding": embedding}, "voice_embedding.pt")
```

### Clone Voice

```python
# Load embedding
checkpoint = torch.load("voice_embedding.pt")
embedding = checkpoint["embedding"]

# Generate with cloned voice
audio = cloner.clone_voice(
    text="Hello, this is a cloned voice",
    reference_embedding=embedding,
    duration=5.0,
)
```

### Command-Line Voice Cloning

```bash
# Extract voice
python -c "
from sunokiller.voice_cloning import extract_voice_from_file
extract_voice_from_file('reference.wav', 'voice.pt')
"

# Clone voice
python -c "
from sunokiller.voice_cloning import clone_voice_from_file
clone_voice_from_file('Hello world', 'voice.pt', 'output.wav')
"
```

## Real-time Streaming

### Stream Music Generation

```python
from sunokiller.streaming import create_streaming_synthesizer

generator = create_streaming_synthesizer(device="cuda")

# Stream chunks as they're generated
for chunk in generator.stream_music("upbeat electronic music", duration=10.0):
    # Process or play chunk immediately
    play_audio(chunk)
```

### Stream to Audio Device

```python
import sounddevice as sd

def play_chunk(chunk):
    sd.play(chunk, 24000)
    sd.wait()

for chunk in generator.stream_music(
    "calm piano melody",
    callback=play_chunk,
):
    pass  # Callback handles playback
```

### Custom Streaming Configuration

```python
from sunokiller.streaming import StreamingGenerator, StreamingConfig

config = StreamingConfig(
    chunk_size=1024,  # Smaller = lower latency
    overlap=128,
    buffer_size=2,
)

generator = StreamingGenerator(synthesizer, config)
```

## Web UI

### Start Web Server

```bash
python web_ui/app.py --host 0.0.0.0 --port 5000
```

Then open http://localhost:5000 in your browser.

### Features

- Generate music from text
- Singing voice synthesis
- Audio enhancement
- Download generated audio
- Adjust parameters in real-time

### API Endpoints

```python
import requests

# Generate music
response = requests.post("http://localhost:5000/generate", json={
    "text": "upbeat pop song",
    "duration": 10.0,
    "temperature": 1.0,
    "mode": "music",
})

file_id = response.json()["file_id"]

# Download
audio_url = f"http://localhost:5000/download/{file_id}"
```

## Mobile Deployment

### iOS (CoreML)

Convert models:
```bash
python mobile/ios/convert_to_coreml.py \
    --model-type vocos \
    --output-dir mobile/ios/models/
```

Integrate in Swift:
```swift
import SUNOKILLER

let synthesizer = SUNOKILLERSynthesizer()
let audio = try synthesizer.generateMusic(
    text: "upbeat pop song",
    duration: 10.0
)
```

### Android (ONNX)

Convert models:
```bash
python mobile/android/convert_to_onnx.py \
    --model-type vocos \
    --output-dir mobile/android/app/src/main/assets/
```

Integrate in Kotlin:
```kotlin
import com.massivemagnetics.sunokiller.Synthesizer

val synthesizer = Synthesizer(context)
val audio = synthesizer.generateMusic(
    text = "upbeat pop song",
    duration = 10.0f
)
```

See `mobile/README.md` for detailed instructions.

## VST/AU Plugin

### Build Plugin

```bash
cd plugins/vst
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Install

The plugin will be installed to:
- **VST3**: `~/.vst3/SUNOKILLER.vst3`
- **AU**: `/Library/Audio/Plug-Ins/Components/SUNOKILLER.component`

### Use in DAW

1. Scan for new plugins in your DAW
2. Load SUNOKILLER as an instrument
3. Enter text prompt
4. Generate music

See `plugins/vst/README.md` for detailed plugin documentation.

## Command-Line Interface

### Generate Music

```bash
sunokiller generate "upbeat pop song with guitar" -o output.wav
```

### Singing Voice

```bash
sunokiller sing "Happy birthday to you" -o birthday.wav -s female
```

### Enhance Audio

```bash
sunokiller enhance input.wav -o enhanced.wav
```

### Advanced Options

```bash
sunokiller generate "jazz piano" \
    -o jazz.wav \
    --duration 15 \
    --temperature 1.2 \
    --steps 50 \
    --quantize \
    --device cuda
```

## Best Practices

### Performance

1. **Use GPU**: 10-20x faster than CPU
2. **Quantization**: Enable for 2-4x speedup with minimal quality loss
3. **Batch Processing**: Generate multiple samples at once
4. **Streaming**: Use for real-time applications

### Quality

1. **Temperature**: Lower (0.7-0.9) for coherent, higher (1.1-1.5) for creative
2. **Duration**: Shorter clips tend to be more coherent
3. **Prompts**: Be specific and descriptive
4. **Diffusion Steps**: More steps = better quality but slower

### Memory

1. Use quantized models on limited RAM
2. Clear GPU cache between generations
3. Stream for long audio generation
4. Use mobile-optimized models on devices

## Troubleshooting

### Common Issues

**Out of Memory**
```python
# Use smaller models
synthesizer = AudioSynthesizer(use_quantization=True)

# Clear cache
import torch
torch.cuda.empty_cache()
```

**Slow Generation**
```python
# Enable quantization
synthesizer = AudioSynthesizer(
    use_quantization=True,
    quantization_type="fp16",
)

# Reduce diffusion steps
audio = synthesizer.generate_music(text="...", num_diffusion_steps=25)
```

**Import Errors**
```bash
# Reinstall package
pip install -e .

# Check dependencies
pip install -r requirements.txt
```

## Examples

See `examples/` directory for:
- Complete code examples
- Jupyter notebooks
- Sample outputs
- Advanced use cases

## Support

- GitHub Issues: https://github.com/MASSIVEMAGNETICS/SUNOKILLER/issues
- Documentation: See individual module READMEs
- Community: [Discord/Forum links]

## License

MIT License - see LICENSE file for details.
