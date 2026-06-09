from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer


def checkpoint_path_for_step(checkpoints_dir: Path, step: int) -> Path:
  return checkpoints_dir / f"checkpoint_{step}.pt"


def latest_checkpoint_path(checkpoints_dir: Path) -> Path | None:
  if not checkpoints_dir.exists():
    return None

  checkpoints = list(checkpoints_dir.glob("checkpoint_*.pt"))
  if not checkpoints:
    return None

  return max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))


def save_checkpoint(
  *,
  model_dir: Path,
  checkpoints_dir: Path,
  weights_path: Path,
  model: Module,
  optimizer: Optimizer,
  step: int,
  tokenizer_version: str,
) -> None:
  model_dir.mkdir(parents=True, exist_ok=True)
  checkpoints_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_path = checkpoint_path_for_step(checkpoints_dir, step)
  checkpoint = {
    "step": step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "tokenizer_version": tokenizer_version,
  }
  torch.save(checkpoint, checkpoint_path)
  torch.save(model.state_dict(), weights_path)
  print(f"Saved checkpoint at step {step} to {checkpoint_path}")


def load_checkpoint(
  *,
  checkpoints_dir: Path,
  weights_path: Path,
  model: Module,
  optimizer: Optimizer,
  device: str,
) -> int:
  latest_checkpoint = latest_checkpoint_path(checkpoints_dir)
  if latest_checkpoint is not None:
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"] + 1
    print(f"Resumed from {latest_checkpoint} at step {checkpoint['step']}")
    return step

  if weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from {weights_path}")
    return 0

  return 0


def save_sample(
  *,
  samples_dir: Path,
  model: Module,
  step: int,
  device: str,
  sample_tokens: int,
  tokenizer,
) -> None:
  model.eval()
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  generated_tokens = model.generate(context, max_new_tokens=sample_tokens)[0].tolist()
  generated_text = tokenizer.decode(generated_tokens)

  samples_dir.mkdir(parents=True, exist_ok=True)
  sample_path = samples_dir / f"sample_step_{step}.txt"
  sample_path.write_text(generated_text, encoding="utf-8")
  print(f"Saved sample to {sample_path}")
  model.train()
