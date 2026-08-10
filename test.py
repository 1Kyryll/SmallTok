from src.regex import RegexTokenizer, GPT4_PAT

text = open("data/taylorswift.txt", encoding="utf-8").read()

tokenizer = RegexTokenizer(GPT4_PAT)
tokenizer.train(text, num_merges=256, verbose=True)

ids = tokenizer.encode("hello world")
tokenizer.decode(ids)

tokenizer.save("mytokenizer")               # .model (reloadable) + .vocab (readable)
RegexTokenizer().load("mytokenizer.model")
