# SUNOKILLER Quick Start Guide

Get up and running with SUNOKILLER in 5 minutes!

## Installation

### Prerequisites
- Python 3.8 or later
- 4GB RAM minimum (8GB recommended)

### Install from GitHub

```bash
# Clone the repository
git clone https://github.com/MASSIVEMAGNETICS/SUNOKILLER.git
cd SUNOKILLER

# Install dependencies
pip install -r requirements.txt

# Install SUNOKILLER
pip install -e .
```

### Verify Installation

```bash
# Test the installation
python tests/test_basic.py
```

You should see:
```
✓ All tests passed!
```

## Basic Usage

### 1. Generate Music from Text (CLI)

```bash
sunokiller generate "upbeat pop song with electric guitar" -o music.wav
```

Options:
- `-d, --duration`: Duration in seconds (default: 10)
- `-t, --temperature`: Creativity level 0.5-2.0 (default: 1.0)
- `--quantize`: Enable for faster inference on CPU

Example with options:
```bash
sunokiller generate "relaxing jazz piano" -o jazz.wav -d 15 --temperature 0.8 --quantize
```

### 2. Generate Singing Voice (CLI)

```bash
sunokiller sing "Happy birthday to you" -o birthday.wav -s female
```

Voice styles:
- `neutral`: Neutral voice (default)
- `male`: Male voice
- `female`: Female voice
- `choir`: Choir/ensemble

Example:
```bash
sunokiller sing "Twinkle twinkle little star" -o lullaby.wav -s female -d 8
```

### 3. Python API

```python
from sunokiller import AudioSynthesizer

# Initialize (first time may take a moment)
synth = AudioSynthesizer(
    device="cpu",          # or "cuda" for GPU
    use_quantization=True, # Recommended for CPU
)

# Generate music
audio = synth.generate_music(
    text="epic orchestral soundtrack",
    duration=10.0,
    temperature=1.0,
)

# Save to file
synth.save_audio(audio, "output.wav")
```

## Common Use Cases

### Music for Videos

```python
# Generate background music
audio = synth.generate_music(
    text="calm ambient background music, slow tempo",
    duration=60.0,  # 1 minute
    temperature=0.7,  # Lower for consistency
)
synth.save_audio(audio, "video_bg.wav")
```

### Game Soundtracks

```python
# Battle music
battle = synth.generate_music(
    text="intense battle music with heavy drums and brass",
    duration=30.0,
    temperature=1.2,  # Higher for variety
)

# Exploration music
explore = synth.generate_music(
    text="peaceful exploration music with strings and woodwinds",
    duration=45.0,
    temperature=0.9,
)
```

### Podcast Intros

```python
# Energetic intro
intro = synth.generate_music(
    text="upbeat podcast intro with synthesizers",
    duration=10.0,
)
synth.save_audio(intro, "podcast_intro.wav")
```

### Singing Birthday Cards

```bash
sunokiller sing "Happy birthday dear Sarah, happy birthday to you" \
  -o sarah_birthday.wav \
  -s female \
  -m "cheerful and celebratory" \
  -d 8
```

## Tips for Best Results

### 1. Writing Good Prompts

**Good prompts are**:
- Specific: "jazz piano with walking bassline" vs "jazz music"
- Include instruments: "with guitar, drums, and synthesizer"
- Mention style/mood: "upbeat", "melancholic", "energetic"
- Include tempo: "fast", "slow", "moderate tempo"

**Examples**:

✅ Good:
```
"upbeat pop song with electric guitar, drums, and synthesizer, 120 BPM"
"melancholic piano ballad, slow tempo, emotional"
"energetic electronic dance music with heavy bass"
```

❌ Too vague:
```
"music"
"song"
"pop"
```

### 2. Temperature Settings

- **0.5-0.7**: More consistent, predictable output
  - Use for: Background music, loops, professional use
  
- **0.8-1.0**: Balanced creativity and consistency
  - Use for: General music generation, demos
  
- **1.1-2.0**: More creative, varied output
  - Use for: Experimental music, exploration

### 3. Optimizing Performance

**For CPU (Low-End Hardware)**:
```python
synth = AudioSynthesizer(
    device="cpu",
    use_quantization=True,     # Essential for CPU
    quantization_type="fp16",  # Best balance
)
```

**For GPU**:
```python
synth = AudioSynthesizer(
    device="cuda",
    use_quantization=False,  # GPU can handle full precision
)
```

**For Apple Silicon (M1/M2)**:
```python
synth = AudioSynthesizer(
    device="mps",
    use_quantization=True,
    quantization_type="fp16",
)
```

### 4. Duration Guidelines

- **Short (5-10s)**: Fastest generation, good for testing
- **Medium (10-30s)**: Balance of quality and speed
- **Long (30-60s)**: Longer context, more coherent

**Note**: Longer durations may have consistency issues. For very long music, generate in segments.

## Troubleshooting

### Out of Memory Error

```python
# Use low-end configuration
from sunokiller.models import VocosVocoder, DiffusionModel, TextToMusicModel

# Smaller models
vocoder = VocosVocoder(dim=256, num_layers=6)
diffusion = DiffusionModel(num_steps=25, base_channels=64)
text_model = TextToMusicModel(dim=512, num_layers=6, text_encoder_name="none")

synth = AudioSynthesizer(
    text_to_music_model=text_model,
    diffusion_model=diffusion,
    vocoder=vocoder,
    device="cpu",
    use_quantization=True,
)
```

### Slow Generation

1. **Enable quantization**:
   ```bash
   sunokiller generate "text" -o out.wav --quantize
   ```

2. **Reduce diffusion steps**:
   ```bash
   sunokiller generate "text" -o out.wav --steps 25
   ```

3. **Use GPU if available**:
   ```bash
   sunokiller generate "text" -o out.wav --device cuda
   ```

### Poor Quality Output

1. **Increase diffusion steps**:
   ```bash
   sunokiller generate "text" -o out.wav --steps 100
   ```

2. **Disable quantization** (if you have enough memory):
   ```python
   synth = AudioSynthesizer(device="cuda", use_quantization=False)
   ```

3. **Adjust temperature** (0.7-1.0 for best quality):
   ```bash
   sunokiller generate "text" -o out.wav --temperature 0.8
   ```

## Next Steps

### Learn More
- Read the [Architecture Documentation](docs/ARCHITECTURE.md)
- Check out [API Reference](docs/API.md)
- Explore [Example Scripts](examples/)

### Advanced Usage
- Fine-tune on your own music dataset
- Export models to ONNX for deployment
- Integrate into your application

### Get Help
- GitHub Issues: [Report bugs or ask questions](https://github.com/MASSIVEMAGNETICS/SUNOKILLER/issues)
- Documentation: [Full docs](docs/)

## Example Workflows

### Complete Song Production

```python
from sunokiller import AudioSynthesizer

synth = AudioSynthesizer(device="cpu", use_quantization=True)

# 1. Generate instrumental
instrumental = synth.generate_music(
    text="pop song with guitar, drums, bass, and piano",
    duration=30.0,
)
synth.save_audio(instrumental, "instrumental.wav")

# 2. Generate vocals
vocals = synth.generate_singing_voice(
    lyrics="This is my song, singing all day long",
    voice_style="female",
    melody_description="catchy pop melody",
    duration=30.0,
)
synth.save_audio(vocals, "vocals.wav")

# 3. Mix in your DAW or using audio libraries
```

### Batch Generation

```python
prompts = [
    "upbeat electronic",
    "calm ambient",
    "energetic rock",
    "smooth jazz",
]

for i, prompt in enumerate(prompts):
    audio = synth.generate_music(prompt, duration=15.0)
    synth.save_audio(audio, f"track_{i}.wav")
    print(f"Generated track {i+1}/4")
```

### Variations of Same Prompt

```python
prompt = "epic cinematic orchestral music"

for temp in [0.7, 1.0, 1.3]:
    audio = synth.generate_music(
        prompt,
        duration=10.0,
        temperature=temp,
    )
    synth.save_audio(audio, f"epic_temp_{temp}.wav")
```

---

**Congratulations!** You're now ready to create amazing AI-generated music with SUNOKILLER! 🎵

For more advanced features and detailed documentation, visit the [docs/](docs/) directory.
