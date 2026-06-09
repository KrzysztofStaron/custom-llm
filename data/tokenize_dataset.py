from __future__ import annotations

import argparse
import json
import time
from array import array
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer import BPETokenizer


DEFAULT_INPUT_PATH = Path("data/dataset.txt")
DEFAULT_VERSION = "byte-bpe-v2"
DEFAULT_PROGRESS_BAR_WIDTH = 32
DEFAULT_PROGRESS_EVERY = 10


def render_progress(current: int, total: int, width: int) -> str:
  if total <= 0:
    return "[" + ("-" * width) + "] 0.0%"

  ratio = current / total
  filled = int(ratio * width)
  bar = "#" * filled + "-" * (width - filled)
  return f"[{bar}] {ratio * 100:5.1f}% ({current}/{total})"


def merge_pair(token_ids: list[int], left: int, right: int, new_id: int) -> list[int]:
  merged: list[int] = []
  i = 0
  while i < len(token_ids):
    if i < len(token_ids) - 1 and token_ids[i] == left and token_ids[i + 1] == right:
      merged.append(new_id)
      i += 2
    else:
      merged.append(token_ids[i])
      i += 1
  return merged


def encode_with_progress(tokenizer: BPETokenizer, text: str, progress_every: int, bar_width: int) -> list[int]:
  token_ids = list(text.encode("utf-8"))
  total_merges = len(tokenizer.merges)

  print(f"Initial token count (bytes): {len(token_ids):,}")
  print("Applying BPE merges...")
  start = time.time()

  for idx, (left, right, new_id) in enumerate(tokenizer.merges, start=1):
    token_ids = merge_pair(token_ids, left, right, new_id)

    should_log = (idx == total_merges) or (idx % progress_every == 0)
    if should_log:
      elapsed = time.time() - start
      progress = render_progress(idx, total_merges, bar_width)
      print(f"\r{progress} | elapsed {elapsed:6.1f}s | tokens {len(token_ids):,}", end="", flush=True)

  print()
  return token_ids


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Tokenize dataset with trained byte-level BPE and show progress.")
  parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input dataset text file path.")
  parser.add_argument("--version", type=str, default=DEFAULT_VERSION, help="Tokenizer version, e.g. byte-bpe-v2.")
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output .bin path for token ids (uint32). Defaults to data/dataset_tokens_<version>.bin",
  )
  parser.add_argument(
    "--progress-every",
    type=int,
    default=DEFAULT_PROGRESS_EVERY,
    help="How often (in merge steps) to refresh progress output.",
  )
  parser.add_argument(
    "--bar-width",
    type=int,
    default=DEFAULT_PROGRESS_BAR_WIDTH,
    help="Progress bar width in characters.",
  )
  return parser


def main() -> None:
  args = build_parser().parse_args()

  if args.progress_every <= 0:
    raise ValueError("--progress-every must be > 0")
  if args.bar_width <= 0:
    raise ValueError("--bar-width must be > 0")

  input_path: Path = args.input
  output_path: Path = args.output or Path(f"data/dataset_tokens_{args.version}.bin")

  print(f"Loading dataset from {input_path}...")
  text = input_path.read_text(encoding="utf-8", errors="ignore")
  print(f"Loaded {len(text):,} characters")

  tokenizer = BPETokenizer(args.version)
  token_ids = encode_with_progress(tokenizer, text, args.progress_every, args.bar_width)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  ids_array = array("I", token_ids)
  with output_path.open("wb") as out_file:
    ids_array.tofile(out_file)

  meta = {
    "version": args.version,
    "input_path": str(input_path),
    "output_path": str(output_path),
    "token_count": len(token_ids),
    "vocab_size": tokenizer.vocab_size,
  }
  meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
  meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

  print(f"Saved token ids to {output_path} ({len(token_ids):,} tokens)")
  print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
  main()
