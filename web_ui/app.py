"""Web UI for SUNOKILLER

Flask-based web interface for easy music generation.
"""

from flask import Flask, render_template, request, send_file, jsonify
import os
from pathlib import Path
import tempfile
import uuid
from datetime import datetime

from sunokiller import AudioSynthesizer
from sunokiller.utils import get_device


app = Flask(__name__)

# Configuration
OUTPUT_DIR = Path(tempfile.gettempdir()) / "sunokiller_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize synthesizer (lazy loading)
synthesizer = None

def get_synthesizer():
    """Lazy load synthesizer."""
    global synthesizer
    if synthesizer is None:
        device = get_device()
        print(f"Initializing synthesizer on {device}...")
        synthesizer = AudioSynthesizer(
            device=device,
            use_quantization=True,  # Use quantization for faster web inference
        )
    return synthesizer


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate music from text."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        duration = float(data.get('duration', 10.0))
        temperature = float(data.get('temperature', 1.0))
        mode = data.get('mode', 'music')  # music or singing
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{file_id}.wav"
        
        # Get synthesizer
        synth = get_synthesizer()
        
        # Generate audio
        if mode == 'music':
            audio = synth.generate_music(
                text=text,
                duration=duration,
                temperature=temperature,
            )
        else:  # singing
            voice_style = data.get('voice_style', 'neutral')
            audio = synth.generate_singing_voice(
                lyrics=text,
                duration=duration,
                voice_style=voice_style,
            )
        
        # Save audio
        synth.save_audio(audio, str(output_path))
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'message': 'Audio generated successfully!',
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<file_id>')
def download(file_id):
    """Download generated audio."""
    file_path = OUTPUT_DIR / f"{file_id}.wav"
    
    if not file_path.exists():
        return "File not found", 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=f"sunokiller_{file_id}.wav",
    )


@app.route('/enhance', methods=['POST'])
def enhance():
    """Enhance uploaded audio."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save uploaded file temporarily
        temp_input = OUTPUT_DIR / f"input_{uuid.uuid4()}.wav"
        audio_file.save(temp_input)
        
        # Load and enhance
        from sunokiller.utils import load_audio
        audio, sr = load_audio(str(temp_input))
        audio_np = audio.numpy()[0]
        
        synth = get_synthesizer()
        enhanced = synth.enhance_audio(audio_np, sample_rate=sr)
        
        # Save enhanced audio
        file_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{file_id}.wav"
        synth.save_audio(enhanced, str(output_path), sr)
        
        # Clean up input
        temp_input.unlink()
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'message': 'Audio enhanced successfully!',
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
    })


def main():
    """Run the web server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SUNOKILLER Web UI")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SUNOKILLER Web UI")
    print("=" * 60)
    print(f"Starting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == '__main__':
    main()
