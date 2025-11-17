# VST/AU Plugin for SUNOKILLER

This directory contains the VST3 and Audio Unit (AU) plugin implementations for using SUNOKILLER in Digital Audio Workstations (DAWs).

## Overview

The SUNOKILLER plugin allows you to:
- Generate music directly in your DAW
- Use text prompts to create musical ideas
- Synthesize singing voices
- Enhance existing audio tracks
- Real-time parameter control

## Supported Formats

- **VST3**: Compatible with most DAWs (Ableton, FL Studio, Cubase, etc.)
- **AU**: macOS DAWs (Logic Pro, GarageBand, etc.)
- **AAX**: Pro Tools (coming soon)

## Building the Plugin

### Requirements

- CMake 3.15+
- C++17 compatible compiler
- JUCE framework (included as submodule)
- Python 3.8+ (for model conversion)
- PyTorch LibTorch (for inference)

### Build Instructions

```bash
# Clone with submodules
git submodule update --init --recursive

# Create build directory
mkdir -p plugins/vst/build
cd plugins/vst/build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Install (optional)
cmake --install .
```

### Platform-Specific Notes

**macOS**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
```

**Windows**:
```bash
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**Linux**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

## Plugin Features

### Parameters

- **Text Prompt**: Enter music description
- **Duration**: Length of generated audio (1-30s)
- **Temperature**: Creativity control (0.5-1.5)
- **Voice Style**: For singing synthesis (Male/Female/Neutral/Choir)
- **Quality**: Speed vs quality tradeoff (Fast/Balanced/High)

### MIDI Control

- Note On: Trigger generation
- CC1 (Mod Wheel): Control temperature
- CC7 (Volume): Output level
- CC74: Duration control

### Automation

All parameters can be automated in your DAW for dynamic control.

## Usage

### In DAW

1. Load SUNOKILLER as an instrument or effect
2. Enter text prompt in the UI
3. Click "Generate" or send MIDI note
4. Audio will be generated and output to track

### Workflow Examples

**Idea Generation**:
```
1. Open SUNOKILLER on a new MIDI track
2. Type: "upbeat funk bassline"
3. Generate and listen
4. Drag to arrangement if you like it
```

**Vocal Creation**:
```
1. Add SUNOKILLER to vocal track
2. Type lyrics and select voice style
3. Generate singing voice
4. Process with effects
```

**Audio Enhancement**:
```
1. Insert SUNOKILLER as effect on audio track
2. Enable "Enhance Mode"
3. Playback will be enhanced in real-time
```

## Performance

### Latency

- **Streaming Mode**: ~100ms initial latency, real-time after
- **Batch Mode**: Full generation before playback (lower CPU)

### CPU Usage

- **GPU Acceleration**: Recommended for real-time use
- **CPU Only**: Works but may have higher latency
- **Quantized Models**: 2-3x faster on CPU

### Memory

- ~500MB RAM for base models
- ~1.5GB with all features enabled
- Models loaded on-demand to save memory

## Configuration

### Model Settings

Edit `SUNOKILLERPlugin/Config.json`:

```json
{
  "model_path": "/path/to/models",
  "use_gpu": true,
  "use_quantization": true,
  "buffer_size": 2048,
  "sample_rate": 48000
}
```

### Presets

Presets are stored in:
- **macOS**: `~/Library/Audio/Presets/MASSIVEMAGNETICS/SUNOKILLER/`
- **Windows**: `%APPDATA%/MASSIVEMAGNETICS/SUNOKILLER/Presets/`
- **Linux**: `~/.config/MASSIVEMAGNETICS/SUNOKILLER/Presets/`

## Troubleshooting

### Plugin Not Showing Up

- **VST3**: Check `~/.vst3/` or `C:\Program Files\Common Files\VST3\`
- **AU**: Check `/Library/Audio/Plug-Ins/Components/`
- Rescan plugins in your DAW

### High CPU Usage

- Enable GPU acceleration
- Use quantized models
- Increase buffer size
- Disable real-time mode

### Crashes

- Update to latest version
- Check system requirements
- Verify model files are present
- Check DAW compatibility

## Development

### Project Structure

```
plugins/vst/
├── CMakeLists.txt          # Build configuration
├── Source/
│   ├── PluginProcessor.cpp # Audio processing
│   ├── PluginEditor.cpp    # GUI
│   └── ModelWrapper.cpp    # ML model interface
├── Resources/              # UI assets
└── Models/                 # Converted models
```

### Adding Features

1. Fork the repository
2. Create feature branch
3. Implement in `Source/`
4. Test with multiple DAWs
5. Submit pull request

### Testing

```bash
# Run plugin validator (VST3)
pluginval --strictness-level 5 --validate path/to/SUNOKILLER.vst3

# Test in DAW
# - Ableton Live
# - FL Studio  
# - Logic Pro
# - Reaper
```

## Known Limitations

- Real-time generation requires GPU
- Maximum 30s generation per trigger
- Some DAWs may have compatibility issues
- AAX format not yet supported

## Roadmap

- [ ] AAX format support
- [ ] Standalone application
- [ ] Cloud model library
- [ ] Collaborative features
- [ ] MIDI file export
- [ ] Advanced automation

## Support

For plugin-specific issues:
- Check [Issues](https://github.com/MASSIVEMAGNETICS/SUNOKILLER/issues)
- Join [Discord community]
- Email: support@massivemagnetics.com

## License

VST is a trademark of Steinberg Media Technologies GmbH.
Audio Unit is a trademark of Apple Inc.

SUNOKILLER plugin is licensed under MIT License.
