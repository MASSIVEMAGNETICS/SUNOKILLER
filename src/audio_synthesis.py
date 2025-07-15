import subprocess
import tempfile
from pydub import AudioSegment

def abc_to_mp3(abc_string):
  """
  Converts an ABC notation string to an MP3 file.
  """
  with tempfile.NamedTemporaryFile(mode='w', suffix='.abc', delete=False) as abc_file:
    abc_file.write(abc_string)
    abc_filepath = abc_file.name

  midi_filepath = abc_filepath.replace('.abc', '.mid')
  wav_filepath = abc_filepath.replace('.abc', '.wav')
  mp3_filepath = abc_filepath.replace('.abc', '.mp3')

  # Convert ABC to MIDI
  subprocess.run(['abc2midi', abc_filepath, '-o', midi_filepath])

  # Convert MIDI to WAV
  subprocess.run(['fluidsynth', '-ni', '/usr/share/sounds/sf2/FluidR3_GM.sf2', midi_filepath, '-F', wav_filepath, '-r', '44100'])

  # Convert WAV to MP3
  audio = AudioSegment.from_wav(wav_filepath)
  audio.export(mp3_filepath, format='mp3')

  return mp3_filepath
