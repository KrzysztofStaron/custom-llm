import torch
from tokenizer import Tokenizer
from torch.nn import functional as F
import torch.nn as nn

text = (
    open("input.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("c.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("c2.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("c3.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("c4.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("closed_factual_questions.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("kys.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("qa.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("books1.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("book2.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("archive/test.txt", "r", encoding="utf-8", errors="ignore").read()
    + open("archive/train.txt", "r", encoding="utf-8", errors="ignore").read()
)

# Remove unwanted characters from the text
UNWANTED_CHARS = "§°ÆÇÉÜàâäæçèéêîóôöùûüœɑɣΔέαβγδεινὶῶ–—‘’“”•™¡£«»¿ÈÓáëíïñõúŒθμοςτ …~¥±²³µ·½ÁÅÍÎÖ×ØÚÞãåìòøĀāăćčĐđėīŁłńŌōśşšūųŻžơưʻʿ̃κСавекостяاحصلنه्กงณตมยรลัาิ่์გდვზიკორსუცძწხჯ჻ḥṃṅṣṭṯảấầắễệịớửỳ‑„†′″⁄₤€₹⅓⅔→−≤☉♭♯〈〉のァアキスットプュリルヴ・動場大戦攻機殻火礮空隊"
text = text.translate({ord(c): None for c in UNWANTED_CHARS})


SPLIT_PERCENT = 0.9
CONTEXT_LENGTH = 64
BATCHE_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
MAX_ITER = 10000
LEARNING_RATE = 1e-3
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1

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

chars = sorted(list(set(text)))
print("".join(chars))
vocab_size = len(chars)

tokenizer = Tokenizer(chars)

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
  m = BigramLanguageModel().to(DEVICE)
  num_params = sum(p.numel() for p in m.parameters())
  print(f"{num_params/1e6:.3f}M parameters ({num_params:,} total)")
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
      
  # save trained weights
  save_path = 'model_weights.pt'
  torch.save(m.state_dict(), save_path)
  print(f"Saved model weights to {save_path}")

  # Generate 1000 tokens just for fun
  m.eval()
  tokenizer = Tokenizer(chars)
  context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # start token (assume 0)
  generated_tokens = m.generate(context, max_new_tokens=1000)[0].tolist()
  generated_text = tokenizer.decode(generated_tokens)
  print("\n--- Generated Sample (1000 tokens) ---\n")
  print(generated_text)
  print("\n--- End of Sample ---\n")


