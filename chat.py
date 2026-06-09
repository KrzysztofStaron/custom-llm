import torch
import train as train_mod
import sys

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
        input_tensor = context_tensor.clone()
        sys.stdout.flush()
        for _ in range(train_mod.CONTEXT_LENGTH):
            idx_cond = input_tensor[:, -train_mod.CONTEXT_LENGTH:]  # crop to block size
            logits, _ = m(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            char = train_mod.tokenizer.decode([token_id])
            sys.stdout.write(char)
            sys.stdout.flush()
        # New line
        print()
