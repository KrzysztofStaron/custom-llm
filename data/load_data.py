import math
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clean import has_unwanted_chars, strip_markdown
from settings import compute_model_param_count, get_target_tokens, load_settings, update_settings

DATASET_PATH = ROOT / "data/dataset.txt"
PARQUET_TOTAL_SHARDS = 104
ESTIMATED_TOKENS_PER_SHARD = 125_000_000
TARGET_TOKENS_MULTIPLIER = 200

settings = load_settings()
PARAM_COUNT = compute_model_param_count(settings)
TARGET_TOKENS = get_target_tokens(settings, TARGET_TOKENS_MULTIPLIER)
CHARS_PER_TOKEN = settings["tokenizer"]["chars_per_token"]
SHARD_COUNT = min(
    PARQUET_TOTAL_SHARDS,
    max(1, math.ceil(TARGET_TOKENS / ESTIMATED_TOKENS_PER_SHARD)),
)
PARQUET_SHARD_URLS = [
    f"hf://datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2/train-{shard:05d}-of-00104.parquet"
    for shard in range(SHARD_COUNT)
]

print(
    f"Target tokens: {TARGET_TOKENS:,.0f} "
    f"({PARAM_COUNT:,} params x {TARGET_TOKENS_MULTIPLIER})"
)
print(f"Loading {SHARD_COUNT} parquet shard(s)...")

ds = load_dataset(
    "parquet",
    data_files=PARQUET_SHARD_URLS,
    split="train",
)


def yield_rows(ds, token_limit):
    acc = 0
    skipped = 0
    kept = 0
    for row in ds:
        text = strip_markdown(row["text"]) + "\n"
        if has_unwanted_chars(text):
            skipped += 1
            continue
        yield text
        kept += 1
        acc += len(text) / CHARS_PER_TOKEN
        if acc >= token_limit:
            break
    print(f"Skipped {skipped} rows with unwanted characters")
    print(f"Processed {acc:,.0f} estimated tokens from {kept:,} rows (chars/token={CHARS_PER_TOKEN:.3f})")


DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
with DATASET_PATH.open("w", encoding="utf-8") as f:
    for text in yield_rows(ds, TARGET_TOKENS):
        f.write(text)

with DATASET_PATH.open("r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
print("".join(chars))

update_settings({
    "dataset": {
        "char_count": len(text),
        "estimated_tokens": len(text) / CHARS_PER_TOKEN,
        "target_tokens": TARGET_TOKENS,
        "model_param_count": PARAM_COUNT,
    },
})
