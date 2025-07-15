from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.text_processing import clean_text
from src.music_generation import generate_music
from transformers import AutoProcessor, AutoModel

app = FastAPI()

processor = AutoProcessor.from_pretrained("declare-lab/tango2")
model = AutoModel.from_pretrained("declare-lab/tango2")

@app.post("/generate-song/")
async def generate_song(text: str):
  """
  Generates a song from a text prompt and returns it as a JSON response.
  """
  cleaned_text = clean_text(text)
  audio_data = generate_music(cleaned_text, model, processor)

  return JSONResponse(content={"audio_data": audio_data})
