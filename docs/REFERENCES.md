# Research Papers and References

This document lists the key research papers and technologies that SUNOKILLER builds upon.

## Core Technologies

### Vocos - Neural Vocoder (2024)

**Paper**: "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders"

- **Authors**: Hubert Siuzdak et al.
- **Conference**: ICLR 2024
- **Link**: https://arxiv.org/abs/2306.00814
- **GitHub**: https://github.com/charactr-platform/vocos

**Key Contributions**:
- Direct Fourier coefficient generation
- 10x faster than WaveNet
- ConvNeXt backbone for efficient processing
- Minimal quality degradation

**Implementation in SUNOKILLER**:
- `src/sunokiller/vocoders/vocos.py`
- Modified for 24kHz audio synthesis
- Optimized for low-latency inference

---

### DiffWave - Diffusion Model (2021)

**Paper**: "DiffWave: A Versatile Diffusion Model for Audio Synthesis"

- **Authors**: Zhifeng Kong et al.
- **Conference**: ICLR 2021
- **Link**: https://arxiv.org/abs/2009.09761
- **GitHub**: https://github.com/lmnt-com/diffwave

**Key Contributions**:
- Non-autoregressive diffusion for audio
- Fast parallel generation
- Conditional and unconditional synthesis

**Implementation in SUNOKILLER**:
- `src/sunokiller/diffusion/diffwave.py`
- DDIM sampling for reduced steps (50 vs 1000)
- Mel-spectrogram conditioning

---

### SpecDiff-GAN (2024)

**Paper**: "SpecDiff-GAN: A Spectrally-Shaped Noise Diffusion GAN for Speech and Music Synthesis"

- **Authors**: Teysir Baoueb et al.
- **Organization**: MERL
- **Link**: https://www.merl.com/publications/docs/TR2024-013.pdf

**Key Contributions**:
- Hybrid diffusion-GAN architecture
- Improved stability and quality
- Faster training and inference

**Influence on SUNOKILLER**:
- Architecture inspiration for diffusion model
- Noise scheduling strategies
- Training stability techniques

---

### SongGen - Text-to-Song (2025)

**Paper**: "SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation"

- **Authors**: Zhenhua Liu et al.
- **Date**: February 2025
- **Link**: https://arxiv.org/abs/2502.13128
- **Project**: https://liuzh-19.github.io/SongGen/

**Key Contributions**:
- Single-stage transformer for vocals + accompaniment
- Voice cloning capabilities
- Mixed and dual-track generation

**Implementation in SUNOKILLER**:
- `src/sunokiller/models/text_to_music.py`
- Transformer architecture with cross-attention
- T5 text encoder integration

---

### MusicGen (2023)

**Paper**: "Simple and Controllable Music Generation"

- **Authors**: Jade Copet et al. (Meta AI)
- **Conference**: NeurIPS 2023
- **Link**: https://arxiv.org/abs/2306.05284
- **Demo**: https://musicgen.com/

**Key Contributions**:
- Single language model for music generation
- Text and melody conditioning
- High-quality stereo output

**Influence on SUNOKILLER**:
- Text conditioning strategies
- Multi-modal input handling
- Architecture design principles

---

### Stable Audio 2.0 (2024)

**Paper**: "Fast Timing-Conditioned Latent Audio Diffusion"

- **Authors**: Zach Evans et al. (Stability AI)
- **Link**: https://arxiv.org/abs/2402.04825
- **Website**: https://stability.ai/stable-audio

**Key Contributions**:
- Latent diffusion for audio
- 3-minute generation capability
- Audio-to-audio transformation

**Influence on SUNOKILLER**:
- Latent space processing ideas
- Long-form generation strategies
- Quality vs speed trade-offs

---

### AudioLDM (2023)

**Paper**: "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models"

- **Authors**: Haohe Liu et al.
- **Link**: https://arxiv.org/abs/2301.12503
- **GitHub**: https://github.com/haoheliu/AudioLDM

**Key Contributions**:
- CLAP embeddings for text-audio alignment
- Latent diffusion in VAE space
- Versatile audio generation

**Influence on SUNOKILLER**:
- Text encoding strategies
- Conditional generation approaches

---

## Optimization Techniques

### Model Quantization

**Paper**: "What Is int8 Quantization and Why Is It Popular for Deep Neural Networks"

- **Organization**: MathWorks
- **Link**: https://www.mathworks.com/company/technical-articles/what-is-int8-quantization-and-why-is-it-popular-for-deep-neural-networks.html

**Implementation**:
- `src/sunokiller/quantization/quantize.py`
- PyTorch dynamic/static quantization
- ONNX quantization support

---

### ONNX Runtime

**Documentation**: https://onnxruntime.ai/docs/

**Features Used**:
- Model export and conversion
- INT8 quantization
- Cross-platform inference

---

### TensorRT

**Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/

**Features**:
- GPU optimization
- FP16/INT8 inference
- Layer fusion

---

## Related Research

### iSTFTNet (2023)

**Paper**: "iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder"

- **Link**: https://arxiv.org/abs/2203.02395
- **Influence**: ISTFT head design in Vocos implementation

### ConvNeXt (2022)

**Paper**: "A ConvNet for the 2020s"

- **Link**: https://arxiv.org/abs/2201.03545
- **Usage**: Backbone architecture in Vocos and ResBlocks

### Rotary Position Embeddings (2021)

**Paper**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"

- **Link**: https://arxiv.org/abs/2104.09864
- **Usage**: Position encoding in text-to-music transformer

### T5 Text Encoder (2019)

**Paper**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"

- **Authors**: Colin Raffel et al. (Google)
- **Link**: https://arxiv.org/abs/1910.10683
- **Usage**: Text encoding in text-to-music model

---

## Datasets (Reference)

While SUNOKILLER doesn't include datasets, here are common datasets for training:

- **MusicCaps**: 5.5K music-text pairs
- **AudioSet**: 2M audio clips with labels
- **FMA**: Free Music Archive (100K+ tracks)
- **LibriTTS**: Speech dataset for voice synthesis
- **LJSpeech**: Single speaker speech dataset

---

## Additional Resources

### Surveys and Reviews

1. "AI-Enabled Text-to-Music Generation: A Comprehensive Review"
   - MDPI Electronics, 2024
   - https://www.mdpi.com/2079-9292/14/6/1197

2. "A Survey on Deep Learning for Music Generation"
   - ACM Computing Surveys, 2024

### Tutorials and Blogs

1. Hugging Face Diffusers: https://huggingface.co/docs/diffusers/
2. PyTorch Audio: https://pytorch.org/audio/
3. AudioLDM Tutorial: https://audioldm.github.io/

---

## Citation

If you use SUNOKILLER in your research, please cite:

```bibtex
@software{sunokiller2025,
  title={SUNOKILLER: Advanced Audio Synthesis System},
  author={MASSIVEMAGNETICS},
  year={2025},
  url={https://github.com/MASSIVEMAGNETICS/SUNOKILLER}
}
```

And please cite the original papers that this work builds upon, particularly:

- Vocos (Siuzdak et al., ICLR 2024)
- DiffWave (Kong et al., ICLR 2021)
- SongGen (Liu et al., 2025)
- MusicGen (Copet et al., NeurIPS 2023)
