"""Quick throughput benchmark for train.py settings across batch sizes."""

import time

import torch
import torch.nn as nn
from torch.nn import functional as F

import train as train_mod


def pick_device() -> str:
  if torch.cuda.is_available():
    return "cuda"
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def sync(device: str) -> None:
  if device == "cuda":
    torch.cuda.synchronize()
  elif device == "mps":
    torch.mps.synchronize()


def measure_step(model, x, y, optimizer, device: str, warmup: int = 5, steps: int = 20) -> float:
  model.train()
  for _ in range(warmup):
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
  sync(device)

  start = time.perf_counter()
  for _ in range(steps):
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
  sync(device)
  elapsed = time.perf_counter() - start
  return elapsed / steps


def main() -> None:
  device = pick_device()
  train_mod.DEVICE = device
  context_length = train_mod.CONTEXT_LENGTH
  vocab_size = train_mod.vocab_size

  print(f"Device: {device}")
  print(f"Model params: {sum(p.numel() for p in train_mod.BigramLanguageModel().parameters()):,}")
  print(f"Context: {context_length}, vocab: {vocab_size}")
  print()

  batch_sizes = [16, 32, 64, 128, 256]
  if device == "cpu":
    batch_sizes = [16, 32, 64, 128]

  baseline = None
  for batch_size in batch_sizes:
    try:
      model = train_mod.BigramLanguageModel().to(device)
      optimizer = torch.optim.AdamW(model.parameters(), lr=train_mod.LEARNING_RATE)
      x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
      y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

      step_s = measure_step(model, x, y, optimizer, device)
      tokens_per_s = (batch_size * context_length) / step_s

      if baseline is None:
        baseline = tokens_per_s

      rel = tokens_per_s / baseline
      print(
        f"batch={batch_size:>3} | {step_s * 1000:6.1f} ms/step | "
        f"{tokens_per_s:>8,.0f} tok/s | {rel:4.2f}x vs batch=16"
      )

      del model, optimizer, x, y
      if device == "cuda":
        torch.cuda.empty_cache()
    except RuntimeError as err:
      print(f"batch={batch_size:>3} | OOM or failed: {err}")
      break

  print()
  print("Notes:")
  print("- Higher tok/s at larger batch usually means you have headroom.")
  print("- If ms/step grows linearly but tok/s plateaus, you're likely compute-bound.")
  print("- On Mac, enable MPS in train.py if you are still on CPU.")


if __name__ == "__main__":
  main()
