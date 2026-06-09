from __future__ import annotations

import argparse
import json
import os
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from settings import load_settings, update_settings
from tokenizer import BPETokenizer, encode_bytes


DEFAULT_INPUT_PATH = Path("data/dataset.txt")
DEFAULT_VERSION = load_settings()["tokenizer"]["version"]
DEFAULT_PROGRESS_BAR_WIDTH = 32
DEFAULT_CHUNK_SIZE = 2_000_000
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) - 1)


def render_progress(current: int, total: int, width: int) -> str:
  if total <= 0:
    return "[" + ("-" * width) + "] 0.0%"

  ratio = current / total
  filled = int(ratio * width)
  bar = "#" * filled + "-" * (width - filled)
  return f"[{bar}] {ratio * 100:5.1f}% ({current}/{total})"


_MERGE_CACHE: list[tuple[int, int, int]] | None = None


def _init_worker(merges: list[tuple[int, int, int]]) -> None:
  global _MERGE_CACHE
  _MERGE_CACHE = merges


def _encode_chunk(chunk: bytes) -> list[int]:
  return encode_bytes(chunk, _MERGE_CACHE)


def encode_parallel(
  raw_bytes: bytes,
  merges: list[tuple[int, int, int]],
  chunk_size: int,
  workers: int,
) -> list[int]:
  chunks = [raw_bytes[i:i + chunk_size] for i in range(0, len(raw_bytes), chunk_size)]
  total_chunks = len(chunks)

  if workers <= 1 or total_chunks == 1:
    print(f"Encoding {len(raw_bytes):,} bytes in 1 chunk...")
    return encode_bytes(raw_bytes, merges)

  print(f"Encoding {len(raw_bytes):,} bytes across {total_chunks} chunks with {workers} workers...")
  start = time.time()
  results: list[list[int] | None] = [None] * total_chunks
  completed = 0

  with ProcessPoolExecutor(
    max_workers=workers,
    initializer=_init_worker,
    initargs=(merges,),
  ) as executor:
    futures = {
      executor.submit(_encode_chunk, chunk): index
      for index, chunk in enumerate(chunks)
    }

    for future in as_completed(futures):
      index = futures[future]
      results[index] = future.result()
      completed += 1
      elapsed = time.time() - start
      progress = render_progress(completed, total_chunks, DEFAULT_PROGRESS_BAR_WIDTH)
      print(f"\r{progress} | elapsed {elapsed:6.1f}s", end="", flush=True)

  print()
  token_ids: list[int] = []
  for chunk_tokens in results:
    token_ids.extend(chunk_tokens)
  return token_ids


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Tokenize dataset with trained byte-level BPE and show progress.")
  parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input dataset text file path.")
  parser.add_argument("--version", type=str, default=DEFAULT_VERSION, help="Tokenizer version, e.g. byte-bpe-v3.")
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output .bin path for token ids (uint32). Defaults to data/dataset_tokens_<version>.bin",
  )
  parser.add_argument(
    "--chunk-size",
    type=int,
    default=DEFAULT_CHUNK_SIZE,
    help="Bytes per encoding chunk for parallel processing.",
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=DEFAULT_WORKERS,
    help="Parallel worker processes. Use 1 to disable multiprocessing.",
  )
  parser.add_argument(
    "--force",
    action="store_true",
    help="Re-tokenize even if output already exists.",
  )
  return parser


def main() -> None:
  args = build_parser().parse_args()

  if args.chunk_size <= 0:
    raise ValueError("--chunk-size must be > 0")
  if args.workers <= 0:
    raise ValueError("--workers must be > 0")

  input_path: Path = args.input
  output_path: Path = args.output or Path(f"data/dataset_tokens_{args.version}.bin")
  meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")

  if output_path.exists() and meta_path.exists() and not args.force:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("version") == args.version:
      print(f"Output already exists: {output_path} ({meta.get('token_count', '?'):,} tokens)")
      print("Use --force to re-tokenize.")
      return

  print(f"Loading dataset bytes from {input_path}...")
  raw_bytes = input_path.read_bytes()
  print(f"Loaded {len(raw_bytes):,} bytes")

  tokenizer = BPETokenizer(args.version)
  start = time.time()
  token_ids = encode_parallel(raw_bytes, tokenizer.merges, args.chunk_size, args.workers)
  elapsed = time.time() - start

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
    "chunk_size": args.chunk_size,
    "workers": args.workers,
  }
  meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

  bytes_per_token = len(raw_bytes) / len(token_ids) if token_ids else 0.0
  print(f"Saved token ids to {output_path} ({len(token_ids):,} tokens)")
  print(f"Saved metadata to {meta_path}")
  print(f"Done in {elapsed:.1f}s ({len(token_ids) / elapsed:,.0f} tokens/s)")

  update_settings({
    "dataset": {
      "token_count": len(token_ids),
      "tokens_bin_path": str(output_path),
      "byte_count": len(raw_bytes),
    },
    "tokenizer": {
      "chars_per_token": bytes_per_token,
      "compression_bytes_per_token": bytes_per_token,
    },
  })


if __name__ == "__main__":
  main()
