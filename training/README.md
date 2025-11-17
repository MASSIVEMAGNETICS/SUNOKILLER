# Training Scripts

This directory contains training scripts for all SUNOKILLER models.

## Available Training Scripts

### 1. Vocos Vocoder Training

Train the Vocos neural vocoder on audio datasets:

```bash
python training/train_vocos.py \
    --data-dir /path/to/audio/files \
    --output-dir outputs/vocos \
    --batch-size 16 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --use-wandb
```

**Data Format**: Directory containing audio files (`.wav`, `.mp3`, `.flac`, `.ogg`)

### 2. Text-to-Music Model Training

Train the transformer-based text-to-music model:

```bash
python training/train_text_to_music.py \
    --data-dir /path/to/dataset \
    --output-dir outputs/text_to_music \
    --batch-size 8 \
    --num-epochs 100 \
    --text-encoder t5-base \
    --use-wandb
```

**Data Format**: Directory with audio files and a `metadata.json` file:

```json
{
  "song1.wav": "upbeat pop song with guitar and drums",
  "song2.wav": "calm piano melody with strings",
  "song3.wav": "energetic rock music with electric guitar"
}
```

### 3. Diffusion Model Training

Train the diffusion model for audio refinement:

```bash
python training/train_diffusion.py \
    --data-dir /path/to/audio \
    --output-dir outputs/diffusion \
    --batch-size 16 \
    --num-steps 50 \
    --use-wandb
```

## Training Parameters

### Common Parameters

- `--data-dir`: Path to training data
- `--output-dir`: Where to save checkpoints and final model
- `--batch-size`: Batch size for training
- `--num-epochs`: Number of training epochs
- `--learning-rate`: Learning rate (default: 1e-4)
- `--device`: Device to use (auto/cpu/cuda/mps)
- `--use-wandb`: Enable Weights & Biases logging
- `--num-workers`: Number of data loading workers

### Model-Specific Parameters

**Vocos**:
- `--dim`: Model dimension (default: 512)
- `--num-layers`: Number of ConvNeXt layers (default: 8)
- `--stft-loss-weight`: Weight for STFT loss (default: 1.0)

**Text-to-Music**:
- `--dim`: Transformer dimension (default: 768)
- `--num-layers`: Number of transformer layers (default: 12)
- `--num-heads`: Number of attention heads (default: 12)
- `--text-encoder`: Text encoder model (t5-small/t5-base/none)

## Dataset Preparation

### Audio Datasets

For vocoder and diffusion training, organize your audio files in a directory:

```
data/
├── audio/
│   ├── song1.wav
│   ├── song2.wav
│   └── ...
```

Supported formats: WAV, MP3, FLAC, OGG

### Text-Audio Pairs

For text-to-music training, you need paired data:

1. Place audio files in a directory
2. Create a `metadata.json` file with text descriptions:

```json
{
  "audio1.wav": "description of the music",
  "audio2.wav": "another description",
  ...
}
```

Example metadata:

```json
{
  "pop_song_1.wav": "upbeat pop song with catchy melody, electric guitar, drums, and bass",
  "jazz_1.wav": "smooth jazz with saxophone solo, piano accompaniment, and soft drums",
  "classical_1.wav": "orchestral piece with strings, woodwinds, and dramatic crescendos"
}
```

## Recommended Datasets

### Public Datasets

1. **FMA (Free Music Archive)**: Large collection of music
   - https://github.com/mdeff/fma

2. **MusicCaps**: Music with text descriptions
   - https://www.kaggle.com/datasets/googleai/musiccaps

3. **MUSDB18**: Multi-track music dataset
   - https://sigsep.github.io/datasets/musdb.html

4. **AudioSet**: Large-scale audio dataset
   - https://research.google.com/audioset/

### Data Requirements

- **Minimum**: 10 hours of audio for basic training
- **Recommended**: 100+ hours for good quality
- **Production**: 1000+ hours for best results

## Multi-GPU Training

Use PyTorch's DistributedDataParallel:

```bash
torchrun --nproc_per_node=4 training/train_vocos.py \
    --data-dir /path/to/data \
    --batch-size 16
```

## Monitoring Training

### Weights & Biases

Enable W&B logging with `--use-wandb`:

```bash
wandb login
python training/train_vocos.py --use-wandb ...
```

### TensorBoard

Coming soon!

## Fine-tuning Pre-trained Models

Load pre-trained weights and continue training:

```python
from sunokiller.pretrained import load_pretrained_weights
from sunokiller.models import VocosVocoder

model = VocosVocoder()
model = load_pretrained_weights(model, "vocos-24khz")

# Continue training...
```

## Tips for Best Results

1. **Data Quality**: Use high-quality audio (24kHz or 48kHz sample rate)
2. **Batch Size**: Larger batches generally give better results
3. **Learning Rate**: Start with 1e-4 and adjust if needed
4. **Gradient Clipping**: Prevents exploding gradients (use 1.0)
5. **Mixed Precision**: Use AMP for faster training on GPUs
6. **Checkpointing**: Save frequently to resume from failures

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Reduce model size (`--dim`, `--num-layers`)
- Use gradient accumulation

### Poor Quality

- Increase training duration
- Use more data
- Adjust loss weights
- Try different learning rates

### Slow Training

- Increase `--num-workers`
- Use GPU if available
- Enable mixed precision training
- Reduce model size for experimentation
