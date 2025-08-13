import torch
from tokenizer import Tokenizer

text = open("input.txt", "r").read()

SPLIT_PERCENT = 0.9
CONTEXT_LENGTH = 8
BATCHE_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
MAX_ITER = 3000

torch.manual_seed(1337)

chars = sorted(list(set(text)))
vocab_size = len(chars)


tokenizer = Tokenizer(chars)

data = torch.tensor(tokenizer.encode(text))
print(data.shape, data.dtype)
print(data)


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


from bigramLM import BigramLanguageModel

m = BigramLanguageModel(vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

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
print(x)

# v1
xbow = torch.zeros((B, T, C))
for b in range(B):
  for t in range(T):
    xprev = x[b, :t+1]
    xbow[b, t] = torch.mean(xprev, dim=0)
# v2
wei = torch.tril(torch.ones(T, T))
wei = wei/torch.sum(wei, dim=1, keepdim=True)
xbow2 = wei @ x

#v3
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x











