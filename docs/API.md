# SUNOKILLER API Reference

## AudioSynthesizer

Main interface for audio synthesis.

### Class: `AudioSynthesizer`

```python
class AudioSynthesizer(
    text_to_music_model: Optional[TextToMusicModel] = None,
    diffusion_model: Optional[DiffusionModel] = None,
    vocoder: Optional[VocosVocoder] = None,
    device: Union[str, torch.device] = "cpu",
    use_quantization: bool = False,
    quantization_type: str = "dynamic",
)
```

**Parameters**:
- `text_to_music_model`: Pre-initialized text-to-music model (optional)
- `diffusion_model`: Pre-initialized diffusion model (optional)
- `vocoder`: Pre-initialized vocoder model (optional)
- `device`: Device to run on ("cpu", "cuda", "mps", or torch.device)
- `use_quantization`: Enable model quantization for efficiency
- `quantization_type`: Type of quantization ("dynamic", "static", "fp16")

**Example**:
```python
from sunokiller import AudioSynthesizer

synth = AudioSynthesizer(
    device="cuda",
    use_quantization=True,
    quantization_type="fp16"
)
```

### Method: `generate_music`

```python
def generate_music(
    text: Union[str, List[str]],
    duration: float = 10.0,
    sample_rate: int = 24000,
    temperature: float = 1.0,
    num_diffusion_steps: int = 50,
    guidance_scale: float = 3.0,
) -> np.ndarray
```

Generate music from text description.

**Parameters**:
- `text`: Text description or list of descriptions
- `duration`: Duration in seconds
- `sample_rate`: Output sample rate in Hz
- `temperature`: Sampling temperature (0.5-2.0, default 1.0)
  - Lower = more consistent, higher = more creative
- `num_diffusion_steps`: Number of diffusion steps (10-100, default 50)
  - Lower = faster, higher = better quality
- `guidance_scale`: Classifier-free guidance scale (1.0-10.0, default 3.0)
  - Higher = stronger adherence to text

**Returns**: Audio array (num_samples,) or (batch_size, num_samples)

**Example**:
```python
audio = synth.generate_music(
    text="epic orchestral soundtrack",
    duration=15.0,
    temperature=1.2,
)
```

### Method: `generate_singing_voice`

```python
def generate_singing_voice(
    lyrics: Union[str, List[str]],
    melody_description: Optional[str] = None,
    duration: float = 10.0,
    sample_rate: int = 24000,
    voice_style: str = "neutral",
) -> np.ndarray
```

Generate singing voice from lyrics.

**Parameters**:
- `lyrics`: Lyrics to sing
- `melody_description`: Optional melody/style description
- `duration`: Duration in seconds
- `sample_rate`: Output sample rate
- `voice_style`: Voice style ("neutral", "male", "female", "choir")

**Returns**: Audio array

**Example**:
```python
audio = synth.generate_singing_voice(
    lyrics="Happy birthday to you",
    voice_style="female",
    melody_description="gentle and slow",
)
```

### Method: `enhance_audio`

```python
def enhance_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    num_diffusion_steps: int = 25,
) -> np.ndarray
```

Enhance audio quality using diffusion.

**Parameters**:
- `audio`: Input audio array
- `sample_rate`: Sample rate
- `num_diffusion_steps`: Enhancement steps (default 25)

**Returns**: Enhanced audio array

### Method: `save_audio`

```python
def save_audio(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 24000,
)
```

Save audio to file.

**Parameters**:
- `audio`: Audio array
- `output_path`: Output file path (.wav, .mp3, .flac)
- `sample_rate`: Sample rate

## Models

### Class: `VocosVocoder`

Fourier-based neural vocoder.

```python
class VocosVocoder(
    input_channels: int = 80,
    dim: int = 512,
    intermediate_dim: int = 1536,
    num_layers: int = 8,
    n_fft: int = 1024,
    hop_length: int = 256,
)
```

**Example**:
```python
from sunokiller.models import VocosVocoder

vocoder = VocosVocoder(dim=512, num_layers=8)
audio = vocoder(mel_spectrogram)
```

### Class: `DiffusionModel`

Fast diffusion model for audio.

```python
class DiffusionModel(
    in_channels: int = 1,
    cond_channels: int = 80,
    num_steps: int = 50,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
)
```

**Example**:
```python
from sunokiller.models import DiffusionModel

diffusion = DiffusionModel(num_steps=50)
audio = diffusion.sample(
    shape=(1, 1, 24000),
    cond=mel_spectrogram,
)
```

### Class: `TextToMusicModel`

Transformer-based text-to-music model.

```python
class TextToMusicModel(
    vocab_size: int = 2048,
    dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    output_dim: int = 80,
)
```

**Example**:
```python
from sunokiller.models import TextToMusicModel

model = TextToMusicModel(dim=768, num_layers=12)
mel_spec = model.generate(
    text=["pop music"],
    max_length=1024,
)
```

## Quantization

### Function: `quantize_model`

```python
def quantize_model(
    model: nn.Module,
    quantization_type: Literal["dynamic", "static", "fp16"] = "dynamic",
    calibration_data: Optional[torch.Tensor] = None,
    output_path: Optional[str] = None,
) -> nn.Module
```

Quantize a model for efficient inference.

**Example**:
```python
from sunokiller.quantization import quantize_model

quantized = quantize_model(
    model,
    quantization_type="fp16",
    output_path="model_fp16.pth"
)
```

### Function: `export_to_onnx`

```python
def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shapes: dict,
    opset_version: int = 14,
    dynamic_axes: Optional[dict] = None,
) -> str
```

Export model to ONNX format.

**Example**:
```python
from sunokiller.quantization import export_to_onnx

export_to_onnx(
    model,
    "model.onnx",
    input_shapes={"features": (1, 80, 256)},
    dynamic_axes={"features": {2: "time"}},
)
```

## Utilities

### Function: `get_device`

```python
def get_device(prefer_cuda: bool = True) -> torch.device
```

Get the best available device.

### Function: `audio_to_mel_spectrogram`

```python
def audio_to_mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    sample_rate: int = 24000,
) -> torch.Tensor
```

Convert audio to mel-spectrogram.

### Function: `normalize_audio`

```python
def normalize_audio(
    audio: np.ndarray,
    target_level: float = -20.0,
) -> np.ndarray
```

Normalize audio to target dB level.

## CLI Commands

### Generate Music

```bash
sunokiller generate "text description" -o output.wav [options]
```

**Options**:
- `-d, --duration SECONDS`: Duration in seconds (default: 10.0)
- `-t, --temperature FLOAT`: Sampling temperature (default: 1.0)
- `--steps INT`: Number of diffusion steps (default: 50)
- `--sample-rate INT`: Sample rate (default: 24000)
- `--device {auto,cpu,cuda,mps}`: Device to use
- `--quantize`: Enable quantization
- `--quantize-type {dynamic,static,fp16}`: Quantization type

### Generate Singing

```bash
sunokiller sing "lyrics" -o output.wav [options]
```

**Options**:
- `-d, --duration SECONDS`: Duration in seconds
- `-s, --style {neutral,male,female,choir}`: Voice style
- `-m, --melody TEXT`: Melody description

### Enhance Audio

```bash
sunokiller enhance input.wav -o output.wav [options]
```

**Options**:
- `--steps INT`: Enhancement steps (default: 25)
