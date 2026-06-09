import torch
from pathlib import Path
import checkpoints
import model_config
from paths import DATASET_PATH
from settings import load_settings
from tokenizer import BPETokenizer
from torch.nn import functional as F
import torch.nn as nn

DATA_PATH = DATASET_PATH
MODEL_NAME = "mimi-256-11"
SPLIT_PERCENT = 0.9
EPOCHS = 1
CONTEXT_LENGTH = model_config.CONTEXT_LENGTH
BATCHE_SIZE = 128
TOKENS_PER_STEP = BATCHE_SIZE * CONTEXT_LENGTH
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

N_EMBD = model_config.N_EMBD
N_HEAD = model_config.N_HEAD
N_LAYER = model_config.N_LAYER
DROPOUT = 0.1
TOKENIZER_VERSION = load_settings()["tokenizer"]["version"]


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


if __name__ == "__main__":
  init_data()
  m = BigramLanguageModel().to(DEVICE)
  num_params = sum(p.numel() for p in m.parameters())
  print(f"{num_params/1e6:.3f}M parameters ({num_params:,} total)")
  print(f"Device: {DEVICE}")
  optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)
  step = checkpoints.load_checkpoint(
    checkpoints_dir=CHECKPOINTS_DIR,
    weights_path=WEIGHTS_PATH,
    model=m,
    optimizer=optimizer,
    device=DEVICE,
  )

  steps_per_epoch = len(train_data) // TOKENS_PER_STEP
  max_iter = steps_per_epoch * EPOCHS
  print(f"Dataset: {len(train_data) + len(val_data):,} tokens ({len(train_data):,} train / {len(val_data):,} val)")
  print(f"Training for {max_iter:,} steps ({EPOCHS} epochs x {steps_per_epoch:,} steps/epoch, {TOKENS_PER_STEP:,} tokens/step).")
  print(f"Checkpoint + sample every {CHECKPOINT_INTERVAL:,} steps.")

  try:
    while step < max_iter:
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
        checkpoints.save_checkpoint(
          model_dir=MODEL_DIR,
          checkpoints_dir=CHECKPOINTS_DIR,
          weights_path=WEIGHTS_PATH,
          model=m,
          optimizer=optimizer,
          step=step,
          tokenizer_version=TOKENIZER_VERSION,
        )
        checkpoints.save_sample(
          samples_dir=SAMPLES_DIR,
          model=m,
          step=step,
          device=DEVICE,
          sample_tokens=SAMPLE_TOKENS,
          tokenizer=tokenizer,
        )
  except KeyboardInterrupt:
    checkpoints.save_checkpoint(
      model_dir=MODEL_DIR,
      checkpoints_dir=CHECKPOINTS_DIR,
      weights_path=WEIGHTS_PATH,
      model=m,
      optimizer=optimizer,
      step=step,
      tokenizer_version=TOKENIZER_VERSION,
    )
    print(f"\nStopped early at step {step}. Checkpoint saved.")
    raise SystemExit(0)

  checkpoints.save_checkpoint(
    model_dir=MODEL_DIR,
    checkpoints_dir=CHECKPOINTS_DIR,
    weights_path=WEIGHTS_PATH,
    model=m,
    optimizer=optimizer,
    step=step,
    tokenizer_version=TOKENIZER_VERSION,
  )
  checkpoints.save_sample(
    samples_dir=SAMPLES_DIR,
    model=m,
    step=step,
    device=DEVICE,
    sample_tokens=SAMPLE_TOKENS,
    tokenizer=tokenizer,
  )
  print(f"Finished training at step {step}.")


