import torch
from tokenizer import Tokenizer
from torch.nn import functional as F
import torch.nn as nn

text = open("input.txt", "r").read()

SPLIT_PERCENT = 0.9
CONTEXT_LENGTH = 8
BATCHE_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
EVAL_ITERS = 500
MAX_ITER = 5000
LEARNING_RATE = 1e-3
N_EMBD = 32


class Head(nn.Module):
  def __init__(self, head_size: int):
    super().__init__()
    self.key = nn.Linear(N_EMBD, head_size, bias=False)
    self.query = nn.Linear(N_EMBD, head_size, bias=False)
    self.value = nn.Linear(N_EMBD, head_size, bias=False)
    self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    v = self.value(x)
    out = wei @ v
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads: int, head_size: int):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(N_EMBD, N_EMBD)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.proj(out)
      return out
  
class FeedForward(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, n_embd),
      nn.ReLU(),
      nn.Linear(n_embd, n_embd),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)

  def forward(self, x):
    x = x + self.sa(x)
    x = x + self.ffwd(x)
    return x
    

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
    self.pos_embedding_table = nn.Embedding(CONTEXT_LENGTH, N_EMBD)

    
    self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=4) for _ in range(3)])

    self.lm_head = nn.Linear(N_EMBD, vocab_size) # (b, t, VOCAB_SIZE)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_embd = self.token_embedding_table(idx)
    pos_emb = self.pos_embedding_table(torch.arange(T, device=DEVICE))

    x = tok_embd + pos_emb
    x = self.blocks(x)
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
  
torch.manual_seed(1337)

chars = sorted(list(set(text)))
vocab_size = len(chars)


tokenizer = Tokenizer(chars)

data = torch.tensor(tokenizer.encode(text))
print(data.shape, data.dtype)


n = int(SPLIT_PERCENT * len(data))
train_data = data[:n]
val_data = data[n:]
print(train_data.shape, val_data.shape)

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

# generating batches from train data
x, y = get_batch("train")

# Generating training sets from batches
for b in range(BATCHE_SIZE):
  for t in range(CONTEXT_LENGTH):
    context = x[b, :t+1]
    target = y[b, t]



m = BigramLanguageModel().to(DEVICE)
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

# Training loop
for step in range(MAX_ITER):
    if step % EVAL_ITERS == 0:
      losses = estimate_loss()
      print(f"step {step}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item())
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(tokenizer.decode(m.generate(context, max_new_tokens=300)[0].tolist()))

B, T, C = 4, 8, 4
x = torch.randn(B, T, C)

print(x.shape)







