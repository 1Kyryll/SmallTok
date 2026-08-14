# Why Tokenization Exists

A language model cannot read text. It reads a sequence of **integers**, each of
which indexes a row in an embedding table. A tokenizer is the thing that turns
a string into those integers, and back again.

The whole design problem is choosing the vocabulary - the fixed, finite set of
symbols the model is allowed to see.

## The two obvious answers are both bad

**One token per character.** The vocabulary is tiny and nothing is ever
unknown, but sequences get very long. That is expensive: attention cost grows
**quadratically** with sequence length, so doubling the token count roughly
quadruples the work. The model also has to spend capacity relearning that
`c`, `a`, `t` means "cat".

**One token per word.** Sequences get short, but the vocabulary explodes, and
you still lose. Any word not in the vocabulary - a typo, a new product name, a
rare inflection - becomes an out-of-vocabulary token, and the model is simply
blind to it. `running`, `runs` and `ran` also end up as three unrelated symbols
with nothing shared between them.

BPE sits in between: frequent sequences become single tokens, rare ones stay
split into smaller pieces. See [bpe_algorithm.md](bpe_algorithm.md).

## Why the initial alphabet is bytes

This repo starts from **UTF-8 bytes**, so the initial alphabet is exactly 256
symbols, ids `0..255`.

The payoff is that out-of-vocabulary text becomes *impossible*. Every string in
every language, every emoji, every byte of malformed input decomposes into
bytes, and every byte already has an id. There is no `<UNK>` token in this
repo, and there does not need to be one.

The cost is that a "character" is not one symbol. UTF-8 uses 1 byte for ASCII,
2 for Cyrillic and most accented Latin, 3 for CJK, and 4 for emoji. So before
any merges are learned, non-English text starts out several times longer.

Measured with the real GPT-2 tokenizer in this repo:

| Text | Chars | Bytes | Tokens |
| --- | --- | --- | --- |
| `Hello, how are you?` | 19 | 19 | **6** |
| `Привет, как дела?` | 17 | 30 | **18** |
| `안녕하세요, 잘 지내세요?` | 14 | 34 | **31** |
| `👋🏽` | 2 | 8 | **5** |

The same sentence costs 5x more tokens in Korean than in English. Training
merges mostly on English text is what causes that - the merges that would
collapse Korean byte sequences were never learned. It is a real cost: more
tokens means more compute, more money, and less text fitting in the context
window.

That last row is also worth noticing. `👋🏽` is one *visual* emoji but two
codepoints (a wave plus a skin-tone modifier), 8 bytes, and 5 tokens. A single
token can end in the middle of a codepoint, which is why decoding has to
tolerate invalid UTF-8 - see [encoding.md](encoding.md).
