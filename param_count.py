import train as train_mod


def main() -> None:
  model = train_mod.BigramLanguageModel()
  num_params = sum(parameter.numel() for parameter in model.parameters())
  trainable_params = sum(
    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
  )

  print(f"Model: {train_mod.MODEL_NAME}")
  print(f"Tokenizer: {train_mod.TOKENIZER_VERSION}")
  print(f"Vocab size: {train_mod.vocab_size:,}")
  print(f"Context length: {train_mod.CONTEXT_LENGTH}")
  print(f"N_EMBD: {train_mod.N_EMBD}, N_HEAD: {train_mod.N_HEAD}, N_LAYER: {train_mod.N_LAYER}")
  print(f"Total parameters: {num_params:,} ({num_params / 1e6:.3f}M)")
  print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.3f}M)")


if __name__ == "__main__":
  main()
