from __future__ import annotations

import json
from pathlib import Path

TOKENIZER_VERSION = "byte-bpe-v2"
BASE_VOCAB_SIZE = 256
DEFAULT_INPUT_PATH = Path("data/dataset.txt")
DEFAULT_TARGET_VOCAB_SIZE = 1024
DEFAULT_MAX_BYTES = 2_000_000
MIN_PAIR_FREQUENCY = 2
MERGE_LOG_EVERY = 100
MERGE_PREVIEW_BYTES = 24


def count_pairs(token_ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    i = 0
    left, right = pair
    while i < len(token_ids):
        if i < len(token_ids) - 1 and token_ids[i] == left and token_ids[i + 1] == right:
            merged.append(new_id)
            i += 2
        else:
            merged.append(token_ids[i])
            i += 1
    return merged


def train_byte_bpe(raw_bytes: bytes, target_vocab_size: int) -> tuple[list[tuple[int, int, int]], dict[int, bytes], list[int]]:
    if target_vocab_size <= BASE_VOCAB_SIZE:
        raise ValueError(f"target_vocab_size must be greater than {BASE_VOCAB_SIZE} for byte-level BPE.")

    token_ids = list(raw_bytes)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BASE_VOCAB_SIZE)}
    merges: list[tuple[int, int, int]] = []
    next_id = BASE_VOCAB_SIZE

    print(f"Starting BPE training with {len(token_ids):,} initial tokens...")
    while next_id < target_vocab_size:
        pair_counts = count_pairs(token_ids)
        if not pair_counts:
            print("No more pairs available. Stopping early.")
            break

        best_pair, best_count = max(pair_counts.items(), key=lambda item: item[1])
        if best_count < MIN_PAIR_FREQUENCY:
            print("Most frequent pair appears only once. Stopping early.")
            break

        left, right = best_pair
        vocab[next_id] = vocab[left] + vocab[right]
        token_ids = merge_pair(token_ids, best_pair, next_id)
        merges.append((left, right, next_id))

        if len(merges) <= 10 or len(merges) % MERGE_LOG_EVERY == 0:
            merged_preview = vocab[next_id][:MERGE_PREVIEW_BYTES]
            print(
                f"[merge {len(merges):>5}] "
                f"pair=({left},{right}) freq={best_count:>8} "
                f"-> id={next_id}, bytes={list(merged_preview)}"
            )

        next_id += 1

    print(f"Finished training: vocab_size={len(vocab):,}, merges={len(merges):,}")
    return merges, vocab, token_ids


def save_tokenizer(path: Path, merges: list[tuple[int, int, int]], vocab: dict[int, bytes]) -> None:
    payload = {
        "version": TOKENIZER_VERSION,
        "base_vocab_size": BASE_VOCAB_SIZE,
        "vocab_size": len(vocab),
        "merges": [[a, b, new_id] for (a, b, new_id) in merges],
        "tokens": {str(token_id): list(token_bytes) for token_id, token_bytes in vocab.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Saved tokenizer to {path}")


def main() -> None:
    # Read input and config
    input_path = DEFAULT_INPUT_PATH
    vocab_size = DEFAULT_TARGET_VOCAB_SIZE
    max_bytes = DEFAULT_MAX_BYTES

    raw = input_path.read_bytes()
    if max_bytes > 0:
        raw = raw[: max_bytes]

    print(f"Loaded {len(raw):,} bytes from {input_path}")
    merges, vocab, final_ids = train_byte_bpe(raw, vocab_size)

    # Determine output path with tokenizer version in filename
    output_dir = input_path.parent
    output_path = output_dir / f"tokenizer_{TOKENIZER_VERSION}.json"

    save_tokenizer(output_path, merges, vocab)

    original_len = len(raw)
    tokenized_len = len(final_ids)
    ratio = original_len / tokenized_len if tokenized_len else 0.0
    print(f"Compression: {original_len:,} bytes -> {tokenized_len:,} tokens (x{ratio:.3f})")


if __name__ == "__main__":
    main()
