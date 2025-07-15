import re
from g2p_en import G2p

def clean_text(text):
  """
  Cleans and normalizes text by converting it to lowercase and removing punctuation.
  """
  text = text.lower()
  text = re.sub(r'[^\w\s]', '', text)
  return text

def text_to_phonemes(text):
  """
  Converts text to a sequence of phonemes.
  """
  g2p = G2p()
  phonemes = g2p(text)
  return phonemes
