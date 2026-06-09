import torch
import train as train_mod

# Recreate model and load trained weights
m = train_mod.BigramLanguageModel().to(train_mod.DEVICE)
state_dict = torch.load(train_mod.WEIGHTS_PATH, map_location=train_mod.DEVICE)
m.load_state_dict(state_dict)
m.eval()

while True:
  user_text = input("Enter your text (blank to quit): ")
  if user_text == "":
    break
  encoded = train_mod.tokenizer.encode(user_text)
  if len(encoded) == 0:
    encoded = [0]
  context_tensor = torch.tensor([encoded], dtype=torch.long, device=train_mod.DEVICE)
  with torch.no_grad():
    generated = m.generate(context_tensor, max_new_tokens=200)[0].tolist()
  print(train_mod.tokenizer.decode(generated))
