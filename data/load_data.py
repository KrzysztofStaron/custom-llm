from datasets import load_dataset

from clean import has_unwanted_chars, strip_markdown

TARGET_TOKENS = 20e6

ds = load_dataset(
    "parquet",
    data_files="hf://datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2/train-00000-of-00104.parquet",
    split="train",
)


def yield_rows(ds, token_limit):
    acc = 0
    skipped = 0
    for row in ds:
        text = strip_markdown(row["text"]) + "\n"
        if has_unwanted_chars(text):
            skipped += 1
            continue
        yield text
        acc += row["token_length"]
        if acc >= token_limit:
            break
    print(f"Skipped {skipped} rows with unwanted characters")
    print(f"Processed {acc} tokens from {acc//row['token_length'] if row['token_length'] else 0} rows (not skipped)")

with open("dataset.txt", "w", encoding="utf-8") as f:
    for text in yield_rows(ds, TARGET_TOKENS):
        f.write(text)

# Load the dataset and show the unique characters
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
print("".join(chars))