import torch
from pathlib import Path
from tokenizer import BPETokenizer
from torch.nn import functional as F
import torch.nn as nn

DATA_PATH = Path("data/dataset.txt")
MODEL_NAME = "mimi-256-11"
SPLIT_PERCENT = 0.9
TARGET_DATASET_TOKENS = 26_000_190
CONTEXT_LENGTH = 256
BATCHE_SIZE = 128
TOKENS_PER_STEP = BATCHE_SIZE * CONTEXT_LENGTH
MAX_ITER = int(TARGET_DATASET_TOKENS * SPLIT_PERCENT) // TOKENS_PER_STEP
DEVICE = (
  'cuda' if torch.cuda.is_available()
  else 'mps' if torch.backends.mps.is_available()
  else 'cpu'
)
EVAL_ITERS = 75
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 10000
SAMPLE_TOKENS = 1000
LEARNING_RATE = 1e-3
MODEL_DIR = Path(MODEL_NAME)
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"
WEIGHTS_PATH = MODEL_DIR / "model_weights.pt"
SAMPLES_DIR = MODEL_DIR / "samples"

N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
TOKENIZER_VERSION = "byte-bpe-v3"


def checkpoint_path_for_step(step: int) -> Path:
  return CHECKPOINTS_DIR / f"checkpoint_{step}.pt"


def latest_checkpoint_path() -> Path | None:
  if not CHECKPOINTS_DIR.exists():
    return None

  checkpoints = list(CHECKPOINTS_DIR.glob("checkpoint_*.pt"))
  if not checkpoints:
    return None

  return max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))

class Head(nn.Module):
  def __init__(self, head_size: int):
    super().__init__()
    self.key = nn.Linear(N_EMBD, head_size, bias=False)
    self.query = nn.Linear(N_EMBD, head_size, bias=False)
    self.value = nn.Linear(N_EMBD, head_size, bias=False)
    self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)))
    self.dropout = nn.Dropout(DROPOUT)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads: int, head_size: int):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(N_EMBD, N_EMBD)
    self.dropout = nn.Dropout(DROPOUT)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.dropout(self.proj(out))
      return out
  
class FeedForward(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, n_embd * 4),
      nn.GELU(),
      nn.Linear(n_embd * 4, n_embd),
      nn.Dropout(DROPOUT),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = LayerNorm1d(n_embd)
    self.ln2 = LayerNorm1d(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
    

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
    self.pos_embedding_table = nn.Embedding(CONTEXT_LENGTH, N_EMBD)

    
    self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
    self.ln_f = LayerNorm1d(N_EMBD)

    self.lm_head = nn.Linear(N_EMBD, vocab_size) # (b, t, VOCAB_SIZE)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_embd = self.token_embedding_table(idx)
    pos_emb = self.pos_embedding_table(torch.arange(T, device=DEVICE))

    x = tok_embd + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -CONTEXT_LENGTH:] # crop the context
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
  
    return idx
  
class LayerNorm1d(nn.Module):
  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(dim))
    self.beta = nn.Parameter(torch.zeros(dim))

  def forward(self, x):
    # Support both (B, C) and (B, T, C)
    if x.dim() == 2:
      mean = x.mean(-1, keepdim=True)
      var = x.var(-1, unbiased=False, keepdim=True)
    else:
      mean = x.mean(-1, keepdim=True)
      var = x.var(-1, unbiased=False, keepdim=True)
    xhat = (x - mean) / torch.sqrt(var + self.eps)
    return self.gamma * xhat + self.beta

tokenizer = BPETokenizer(TOKENIZER_VERSION)
vocab_size = tokenizer.vocab_size
train_data = None
val_data = None


def init_data() -> None:
  global train_data, val_data

  text = DATA_PATH.read_text(encoding="utf-8", errors="ignore")
  cached_token_ids = tokenizer.load_dataset_tokens_if_version_matches()
  if cached_token_ids is not None:
    print(f"Loaded cached dataset token ids ({len(cached_token_ids):,})")
    data = torch.tensor(cached_token_ids)
  else:
    data = torch.tensor(tokenizer.encode(text))

  n = int(SPLIT_PERCENT * len(data))
  train_data = data[:n]
  val_data = data[n:]

def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - CONTEXT_LENGTH, (BATCHE_SIZE,))
  x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in ix])
  y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in ix])
  x, y = x.to(DEVICE), y.to(DEVICE)
  return x, y

@torch.no_grad()
def estimate_loss():
  losses = {}
  m.eval()
  for split in ["train", "val"]:
    batch_losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
      xb, yb = get_batch(split)
      _, loss = m(xb, yb)
      batch_losses[k] = loss.item()
    losses[split] = batch_losses.mean().item()
  m.train()
  return losses


def save_checkpoint(model, optimizer, step: int) -> None:
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
  checkpoint_path = checkpoint_path_for_step(step)
  checkpoint = {
    "step": step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "tokenizer_version": TOKENIZER_VERSION,
  }
  torch.save(checkpoint, checkpoint_path)
  torch.save(model.state_dict(), WEIGHTS_PATH)
  print(f"Saved checkpoint at step {step} to {checkpoint_path}")


def save_sample(model, step: int) -> None:
  model.eval()
  context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
  generated_tokens = model.generate(context, max_new_tokens=SAMPLE_TOKENS)[0].tolist()
  generated_text = tokenizer.decode(generated_tokens)

  SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
  sample_path = SAMPLES_DIR / f"sample_step_{step}.txt"
  sample_path.write_text(generated_text, encoding="utf-8")
  print(f"Saved sample to {sample_path}")
  model.train()


def load_checkpoint(model, optimizer) -> int:
  latest_checkpoint = latest_checkpoint_path()
  if latest_checkpoint is not None:
    checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"] + 1
    print(f"Resumed from {latest_checkpoint} at step {checkpoint['step']}")
    return step

  if WEIGHTS_PATH.exists():
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print(f"Loaded weights from {WEIGHTS_PATH}")
    return 0

  return 0


if __name__ == "__main__":
  init_data()
  m = BigramLanguageModel().to(DEVICE)
  num_params = sum(p.numel() for p in m.parameters())
  print(f"{num_params/1e6:.3f}M parameters ({num_params:,} total)")
  print(f"Device: {DEVICE}")
  optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)
  step = load_checkpoint(m, optimizer)

  train_tokens = int(TARGET_DATASET_TOKENS * SPLIT_PERCENT)
  print(f"Training for {MAX_ITER:,} steps ({train_tokens:,} train tokens, {TOKENS_PER_STEP:,} tokens/step).")
  print(f"Checkpoint + sample every {CHECKPOINT_INTERVAL:,} steps.")

  try:
    while step < MAX_ITER:
      if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {step}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

      xb, yb = get_batch("train")
      logits, loss = m(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      step += 1

      if step % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(m, optimizer, step)
        save_sample(m, step)
  except KeyboardInterrupt:
    save_checkpoint(m, optimizer, step)
    print(f"\nStopped early at step {step}. Checkpoint saved.")
    raise SystemExit(0)

  save_checkpoint(m, optimizer, step)
  save_sample(m, step)
  print(f"Finished training at step {step}.")


