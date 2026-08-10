import os
import tempfile

from src.basic import Basic
from src.regex import RegexTokenizer, GPT2_PAT, GPT4_PAT

TRAIN_TEXT = """
import os
import sys


class BufferedWriter:
    def __init__(self, path, buf_size=65536, retries=3):
        self.path = path
        self.buf_size = buf_size
        self.retries = retries
        self.buffer = []

    def write(self, chunk):
        # DON'T forget to flush the buffer!
        if self.buf_size > 1000000:
            raise ValueError("buffer too large")
        self.buffer.append(chunk)
        if len(self.buffer) >= self.buf_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return False
        for attempt in range(self.retries):
            try:
                with open(self.path, "ab") as handle:
                    handle.write(b"".join(self.buffer))
                self.buffer = []
                return True
            except OSError as error:
                print(f"attempt {attempt} failed: {error}", file=sys.stderr)
        return False


def process(paths, buf_size=65536):
    results = {}
    for path in paths:
        if not os.path.exists(path):
            results[path] = None
            continue
        writer = BufferedWriter(path, buf_size=buf_size)
        results[path] = writer.flush()
    return results
"""

SAMPLE = "Hello world! I've got 12345 things. 안녕하세요 👋"


def report(name, tokenizer, text):
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    ratio = len(text.encode("utf-8")) / len(encoded)

    print(f"\n{name}")
    print(f"  merges:      {len(tokenizer.merges)}")
    print(f"  vocab size:  {len(tokenizer.vocab)}")
    print(f"  {len(text.encode('utf-8'))} bytes -> {len(encoded)} tokens ({ratio:.2f}x compression)")
    print(f"  round-trip:  {'OK' if decoded == text else 'FAIL'}")
    return len(encoded)


print("Training on a Python snippet, 64 merges each")

basic = Basic()
basic.train(TRAIN_TEXT, num_merges=64)

gpt2 = RegexTokenizer(GPT2_PAT)
gpt2.train(TRAIN_TEXT, num_merges=64)

gpt4 = RegexTokenizer(GPT4_PAT)
gpt4.train(TRAIN_TEXT, num_merges=64)

print(f"\nSample: {SAMPLE!r}")
report("Basic (no chunking)", basic, SAMPLE)
report("Regex (GPT-2 pattern)", gpt2, SAMPLE)
report("Regex (GPT-4 pattern)", gpt4, SAMPLE)

# on the training text itself the chunking difference is clearest
print(f"\nOn the training text ({len(TRAIN_TEXT.encode('utf-8'))} bytes):")
n2 = len(gpt2.encode(TRAIN_TEXT))
n4 = len(gpt4.encode(TRAIN_TEXT))
print(f"  GPT-2 pattern: {n2} tokens")
print(f"  GPT-4 pattern: {n4} tokens  ({n2 / n4:.2f}x better)")

# a token can end mid-codepoint, so decoding a partial sequence must not crash
partial = gpt4.encode("👋")[:1]
print(f"\nPartial emoji token {partial} decodes to {gpt4.decode(partial)!r} (replacement char, no crash)")

# the real thing: load GPT-2's 50k merges and check we reproduce tiktoken exactly
try:
    import tiktoken

    from src.gpt2 import GPT2Tokenizer

    real = GPT2Tokenizer.from_tiktoken()
    reference = tiktoken.get_encoding("gpt2")
    mine, theirs = real.encode(SAMPLE), reference.encode(SAMPLE)

    print("\nGPT-2 (loaded from tiktoken)")
    print(f"  merges:      {len(real.merges)}")
    print(f"  {len(SAMPLE.encode('utf-8'))} bytes -> {len(mine)} tokens")
    print(f"  matches tiktoken: {'OK' if mine == theirs else 'FAIL'}")
    print(f"  round-trip:       {'OK' if real.decode(mine) == SAMPLE else 'FAIL'}")
except ImportError:
    print("\n(install tiktoken to compare against the real GPT-2 tokenizer)")

# special tokens are spliced in whole, never merged into their neighbours
eot = 256 + len(gpt4.merges)
gpt4.register_special_tokens({"<|endoftext|>": eot})
tagged = "first document<|endoftext|>second document"

print("\nSpecial tokens")
print(f"  allowed_special='all':  {eot in gpt4.encode(tagged, allowed_special='all')} (id {eot} present)")
print(f"  allowed_special='none': {eot in gpt4.encode(tagged, allowed_special='none')} (encoded as plain text)")
try:
    gpt4.encode(tagged)
except ValueError:
    print("  default:                refuses text containing a special token")
print(f"  round-trip:             "
      f"{'OK' if gpt4.decode(gpt4.encode(tagged, allowed_special='all')) == tagged else 'FAIL'}")

# save / load
with tempfile.TemporaryDirectory() as tmp:
    prefix = os.path.join(tmp, "gpt4")
    gpt4.save(prefix)
    reloaded = RegexTokenizer().load(prefix + ".model")

    same = reloaded.encode(TRAIN_TEXT) == gpt4.encode(TRAIN_TEXT)
    print("\nSave / load")
    print(f"  wrote {os.path.basename(prefix)}.model and {os.path.basename(prefix)}.vocab")
    print(f"  merges + specials restored: {reloaded.merges == gpt4.merges and reloaded.special_tokens == gpt4.special_tokens}")
    print(f"  encodes identically:        {'OK' if same else 'FAIL'}")
