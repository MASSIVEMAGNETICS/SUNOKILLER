import torch
from transformers import AutoProcessor, AutoModel

def generate_music(text, model, processor):
  """
  Generates audio from a text prompt.
  """
  inputs = processor(text=text, return_tensors="pt")

  speech_values = model.generate(**inputs, do_sample=True)

  return speech_values.cpu().numpy().squeeze().tolist()
