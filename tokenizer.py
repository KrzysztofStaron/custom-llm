import json
from array import array
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
    self.version = version
    self.data_dir = Path(__file__).resolve().parent / "data"
    tokenizer_path = Path(__file__).resolve().parent / "data" / f"tokenizer_{version}.json"
    if not tokenizer_path.exists():
      if version == "bpe":
        tokenizer_path = self.data_dir / "tokenizer_bpe.json"
      else:
        raise FileNotFoundError(f"Tokenizer file not found for version '{version}'.")

    tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    self.merges = [tuple(merge) for merge in tokenizer_data["merges"]]
    self.tokens = {
      int(token_id): bytes(token_bytes)
      for token_id, token_bytes in tokenizer_data["tokens"].items()
    }
    self.vocab_size = len(self.tokens)

  def load_dataset_tokens_if_version_matches(self):
    tokens_path = self.data_dir / f"dataset_tokens_{self.version}.bin"
    meta_path = tokens_path.with_suffix(tokens_path.suffix + ".meta.json")
    if not tokens_path.exists() or not meta_path.exists():
      return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("version") != self.version:
      return None

    token_ids = array("I")
    bytes_count = tokens_path.stat().st_size
    if bytes_count % token_ids.itemsize != 0:
      return None

    with tokens_path.open("rb") as token_file:
      token_ids.fromfile(token_file, bytes_count // token_ids.itemsize)

    return token_ids.tolist()

  def encode(self, text):
    if isinstance(text, str):
      raw_bytes = text.encode("utf-8")
    else:
      raw_bytes = text

    return encode_bytes(raw_bytes, self.merges)

  def decode(self, int_arr):
    output_bytes = bytearray()
    for token_id in int_arr:
      output_bytes.extend(self.tokens[token_id])

    return bytes(output_bytes).decode("utf-8", errors="replace")


def merge_pair_array(token_ids: array, left: int, right: int, new_id: int) -> array:
  merged = array("I")
  i = 0
  n = len(token_ids)
  while i < n:
    if i < n - 1 and token_ids[i] == left and token_ids[i + 1] == right:
      merged.append(new_id)
      i += 2
    else:
      merged.append(token_ids[i])
      i += 1
  return merged


def encode_bytes(raw_bytes: bytes, merges: list[tuple[int, int, int]]) -> list[int]:
  token_ids = array("I", list(raw_bytes))
  for left, right, new_id in merges:
    if len(token_ids) < 2:
      break
    token_ids = merge_pair_array(token_ids, left, right, new_id)
  return token_ids.tolist()

