from tokenizer import Tokenizer

text = open("input.txt", "r").read()
SPLIT_PERCENT = 0.9
CONTEXT_LENGTH = 8


chars = sorted(list(set(text)))
vocab_size = len(chars)


tokenizer = Tokenizer(chars)


encoded = tokenizer.encode("hello")
print(encoded)
print(tokenizer.decode(encoded))


import torch

data = torch.tensor(tokenizer.encode(text))
print(data.shape, data.dtype)
print(data)



n = int(SPLIT_PERCENT * len(data))
train_data = data[:n]
val_data = data[n:]
print(train_data.shape, val_data.shape)


BATCHES = 4

def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - CONTEXT_LENGTH, (BATCHES,))
  x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in ix])
  y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in ix])
  return x, y

# generating batches from train data
x, y = get_batch("train")

# Generating training sets from batches
for b in range(BATCHES):
  for t in range(CONTEXT_LENGTH):
    context = x[b, :t+1]
    target = y[b, t]


from bigramLM import BigramLanguageModel

m = BigramLanguageModel(vocab_size)
out = m(x)
print(out.shape)





