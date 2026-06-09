import json
from array import array
from pathlib import Path

import numpy as np


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


def _merge_np(arr: np.ndarray, left: int, right: int, new_id: int) -> np.ndarray:
  """Replace every non-overlapping (left, right) adjacent pair with new_id, vectorized."""
  if arr.size < 2:
    return arr
  idx = np.flatnonzero((arr[:-1] == left) & (arr[1:] == right))
  if idx.size == 0:
    return arr
  if left == right:
    # Overlapping matches (e.g. "aa" in "aaa") must be resolved greedily left-to-right:
    # a kept match at i consumes i+1, so a match starting at i+1 is dropped.
    keep = np.empty(idx.size, dtype=bool)
    keep[0] = True
    last = idx[0]
    for k in range(1, idx.size):
      if idx[k] == last + 1:
        keep[k] = False
      else:
        keep[k] = True
        last = idx[k]
    idx = idx[keep]
  out = arr.copy()
  out[idx] = new_id
  mask = np.ones(arr.size, dtype=bool)
  mask[idx + 1] = False
  return out[mask]


def encode_bytes_np(raw_bytes: bytes, merges: list[tuple[int, int, int]]) -> np.ndarray:
  """Byte-level BPE encode, returning a uint32 numpy array of token ids."""
  arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.uint32)
  for left, right, new_id in merges:
    if arr.size < 2:
      break
    arr = _merge_np(arr, left, right, new_id)
  return arr


def encode_bytes(raw_bytes: bytes, merges: list[tuple[int, int, int]]) -> list[int]:
  return encode_bytes_np(raw_bytes, merges).tolist()

