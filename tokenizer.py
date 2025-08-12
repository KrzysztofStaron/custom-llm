class Tokenizer:
  def __init__(self, vocab):
    self.vocab = vocab
    self.vocab_size = len(vocab)

  def encode(self, str):
    output = []
    for char in str:
      output.append(self.chars.index(char))

    return output

  def decode(self, int_arr):
    output = ""
    for i in int_arr:
      output += self.chars[i]
    
    return output


