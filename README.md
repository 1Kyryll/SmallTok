# SmallTok

A byte-level BPE tokenizer built from scratch, for learning how tokenization
actually works. It trains on your own text, and can also load OpenAI's real
GPT-2 merges and reproduce them exactly.

```bash
python main.py
```

## Usage

```python
from src.regex import RegexTokenizer, GPT4_PAT

tokenizer = RegexTokenizer(GPT4_PAT)
tokenizer.train(text, num_merges=256, verbose=True)

ids = tokenizer.encode("hello world")
tokenizer.decode(ids)

tokenizer.save("mytokenizer")               # .model (reloadable) + .vocab (readable)
RegexTokenizer().load("mytokenizer.model")
```

### Special tokens

```python
tokenizer.register_special_tokens({"<|endoftext|>": 256 + len(tokenizer.merges)})

tokenizer.encode(text, allowed_special="all")   # recognise them
tokenizer.encode(text, allowed_special="none")  # encode as plain text
tokenizer.encode(text)                          # refuse text containing one (default)
```

The default refuses rather than silently encoding, so untrusted input can't
smuggle in a control token — the same default tiktoken uses.

### The real GPT-2

```python
from src.gpt2 import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_tiktoken()   # needs tiktoken
tokenizer.save("gpt2")                      # afterwards it loads without tiktoken
```

`recover_merges` rebuilds the `(id, id) -> id` merge list from tiktoken's flat
`bytes -> rank` table by replaying BPE on each token's own bytes. The result
encodes byte-identically to `tiktoken.get_encoding("gpt2")`.

## Docs

| Doc | What it covers |
| --- | --- |
| [tokenization_basics.md](docs/tokenization_basics.md) | Why a vocabulary is needed, and why it starts from 256 UTF-8 bytes |
| [bpe_algorithm.md](docs/bpe_algorithm.md) | The merge loop itself |
| [chunking.md](docs/chunking.md) | Why the text is split on a regex first, and what it costs |
| [encoding.md](docs/encoding.md) | Replaying merges, decoding, and special tokens |

## Layout

| File | What's in it |
| --- | --- |
| [src/tokenizer.py](src/tokenizer.py) | `Tokenizer` base: the merge loop, vocab, encode/decode, save/load |
| [src/basic.py](src/basic.py) | `Basic` — BPE over the whole text, no chunking |
| [src/regex.py](src/regex.py) | `RegexTokenizer` — splits on a regex first; GPT-2 and GPT-4 patterns |
| [src/gpt2.py](src/gpt2.py) | Loading OpenAI's GPT-2 merges out of tiktoken |
| [SmallTok.ipynb](SmallTok.ipynb) | The notebook this was built up from |

Subclasses only decide **how the text is chunked** before BPE runs — everything
else lives in the base class.

## Why chunking matters

`Basic` lets a merge span any two adjacent bytes, so it will happily learn a
token like `"dog."` — on this repo's corpus, 113 of its 256 learned tokens
cross a word or punctuation boundary. Splitting on a regex first keeps merges
inside a chunk, which costs some raw compression and buys consistency.
Details and numbers in [docs/chunking.md](docs/chunking.md).

## Not implemented

- Training is O(n) per merge over the whole corpus, and `encode` re-scans the
  sequence once per merge. Fine for learning, far too slow for a real corpus.
- No parallelism, no incremental/streaming encode.
