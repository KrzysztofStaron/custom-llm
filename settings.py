from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import model_config


PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_ROOT / "pipeline_settings.json"

DEFAULTS: dict[str, Any] = {
  "tokenizer": {
    "version": "byte-bpe-v3",
    "target_vocab_size": 2048,
    "chars_per_token": 3.323,
    "compression_bytes_per_token": None,
  },
  "dataset": {
    "token_count": None,
    "byte_count": None,
    "char_count": None,
    "estimated_tokens": None,
    "target_tokens": None,
    "model_param_count": None,
    "tokens_bin_path": None,
  },
}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
  merged = copy.deepcopy(base)
  for key, value in updates.items():
    if isinstance(value, dict) and isinstance(merged.get(key), dict):
      merged[key] = _deep_merge(merged[key], value)
    else:
      merged[key] = value
  return merged


def load_settings() -> dict[str, Any]:
  if not SETTINGS_PATH.exists():
    return copy.deepcopy(DEFAULTS)

  loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
  return _deep_merge(DEFAULTS, loaded)


def save_settings(settings: dict[str, Any]) -> None:
  SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
  print(f"Updated settings at {SETTINGS_PATH}")


def update_settings(updates: dict[str, Any]) -> dict[str, Any]:
  settings = load_settings()
  settings = _deep_merge(settings, updates)
  save_settings(settings)
  return settings


def compute_model_param_count(settings: dict[str, Any]) -> int:
  vocab_size = settings["tokenizer"]["target_vocab_size"]
  context_length = model_config.CONTEXT_LENGTH
  n_embd = model_config.N_EMBD
  n_head = model_config.N_HEAD
  n_layer = model_config.N_LAYER
  head_size = n_embd // n_head

  token_embedding = vocab_size * n_embd
  pos_embedding = context_length * n_embd
  lm_head = n_embd * vocab_size + vocab_size
  final_layer_norm = 2 * n_embd

  head_params = 3 * n_embd * head_size
  attention_params = n_head * head_params + n_embd * n_embd + n_embd
  feedforward_params = n_embd * (4 * n_embd) + (4 * n_embd) + (4 * n_embd) * n_embd + n_embd
  block_layer_norm_params = 2 * (2 * n_embd)
  block_params = attention_params + feedforward_params + block_layer_norm_params

  return token_embedding + pos_embedding + lm_head + final_layer_norm + n_layer * block_params


def get_target_tokens(settings: dict[str, Any], multiplier: float) -> float:
  return compute_model_param_count(settings) * multiplier
