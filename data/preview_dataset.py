with open("dataset.txt", "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= 200:
            break
        print(line, end='')
   