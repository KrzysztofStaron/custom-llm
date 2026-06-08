import json
from pathlib import Path


class CharTokenizer:
  def __init__(self, vocab):
    self.vocab = vocab
    self.vocab_size = len(vocab)

  def encode(self, str):
    output = []
    for char in str:
      output.append(self.vocab.index(char))

    return output

  def decode(self, int_arr):
    output = ""
    for i in int_arr:
      output += self.vocab[i]
    
    return output


class BPETokenizer:
  def __init__(self, version):
    tokenizer_path = Path(__file__).resolve().parent / "data" / f"tokenizer_{version}.json"
    if not tokenizer_path.exists():
      if version == "bpe":
        tokenizer_path = Path(__file__).resolve().parent / "data" / "tokenizer_bpe.json"
      else:
        raise FileNotFoundError(f"Tokenizer file not found for version '{version}'.")

    tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    self.merges = [tuple(merge) for merge in tokenizer_data["merges"]]
    self.tokens = {
      int(token_id): bytes(token_bytes)
      for token_id, token_bytes in tokenizer_data["tokens"].items()
    }
    self.vocab_size = len(self.tokens)

  def encode(self, text):
    token_ids = list(text.encode("utf-8"))
    for left, right, new_id in self.merges:
      merged = []
      i = 0
      while i < len(token_ids):
        if i < len(token_ids) - 1 and token_ids[i] == left and token_ids[i + 1] == right:
          merged.append(new_id)
          i += 2
        else:
          merged.append(token_ids[i])
          i += 1
      token_ids = merged

    return token_ids

  def decode(self, int_arr):
    output_bytes = bytearray()
    for token_id in int_arr:
      output_bytes.extend(self.tokens[token_id])

    return bytes(output_bytes).decode("utf-8", errors="replace")


