import torch
import train as train_mod
from tokenizer import Tokenizer

# Recreate tokenizer/vocab exactly as in training
tokenizer = Tokenizer(train_mod.chars)

# Recreate model and load trained weights
m = train_mod.BigramLanguageModel().to(train_mod.DEVICE)
state_dict = torch.load("model_weights.pt", map_location=train_mod.DEVICE)
m.load_state_dict(state_dict)
m.eval()

while True:
  user_text = input("Enter your text (blank to quit): ")
  if user_text == "":
    break
  filtered_text = "".join(ch for ch in user_text if ch in tokenizer.vocab)
  encoded = tokenizer.encode(filtered_text)
  if len(encoded) == 0:
    encoded = [0]
  generated = encoded.copy()
  with torch.no_grad():
    while True:
      context_tensor = torch.tensor([generated], dtype=torch.long, device=train_mod.DEVICE)
      next_token = m.generate(context_tensor, max_new_tokens=1)[0, -1].item()
      generated.append(next_token)
      if tokenizer.vocab[next_token] == ".":
        break
  print(tokenizer.decode(generated))
