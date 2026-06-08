from functools import reduce
from datasets import load_dataset

TARGET_TOKENS = 20e6

ds = load_dataset(
    "parquet",
    data_files="hf://datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2/train-00000-of-00104.parquet",
    split="train",
)


def yield_rows(ds, token_limit):
    acc = 0
    for row in ds:
        yield row["text"]
        acc += row["token_length"]
        if acc >= token_limit:
            break

with open("dataset.txt", "w") as f:
    for text in yield_rows(ds, TARGET_TOKENS):
        f.write(text)