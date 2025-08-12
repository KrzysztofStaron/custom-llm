from tokenizer import Tokenizer

text = open("input.txt", "r").read()


chars = sorted(list(set(text)))
vocab_size = len(chars)


tokenizer = Tokenizer(chars)


encoded = tokenizer.encode("hello")
print(encoded)
print(tokenizer.decode(encoded))



