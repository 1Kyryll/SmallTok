# Why the Text Is Split Before BPE

Plain BPE merges *any* two adjacent symbols that happen to be frequent. It has
no idea what a word is. Left alone on ordinary prose, it spends a lot of its
vocabulary on tokens that span boundaries between words and punctuation.

This is measurable. Training `Basic` (no chunking) for 256 merges on
`data/taylorswift.txt`:

- **113 of the 256** learned tokens cross a word or punctuation boundary
- the longest token learned is `". Archived from the original on "` - a full
  phrase, including the leading period and both spaces

The same budget spent by `RegexTokenizer` with the GPT-4 pattern produces
clean, reusable pieces instead:

```plaintext
Basic   longest: ". Archived from the original on "  "from the original "
GPT-4   longest: " Retrieved"  " Billboard"  " September"  " original"
```

## Why cross-boundary tokens are a problem

They make a word's tokenization depend on its **neighbours**. If `"dog."` is
one token but `"dog"` is another, then the model sees `dog` at the end of a
sentence and `dog` mid-sentence as unrelated symbols, and has to learn the same
fact twice. The vocabulary is also just wasted: `". Archived from the original
on "` is one very long token that only ever fires on Wikipedia citation lines.

## The fix

Split the text on a regex first, then run BPE **inside each chunk
independently**. Pair counts are still summed across all chunks, so merges are
still chosen globally - but a merge can never join two symbols that sit on
opposite sides of a split.

That invariant is enforced by a test
(`test_merges_never_cross_chunk_boundaries` in [../test.py](../test.py)), which
checks that every learned token is a substring of some chunk.

## GPT-2 vs GPT-4 patterns

Both patterns are in [../src/regex.py](../src/regex.py). They agree on the
basics - words keep their leading space, so `" world"` is one chunk - but
differ in a few deliberate ways:

| Input | GPT-2 | GPT-4 |
| --- | --- | --- |
| `1234567` | `['1234567']` | `['123', '456', '7']` |
| `IT'S` | `['IT', "'", 'S']` | `['IT', "'S"]` |
| `():\n    ` | `['():', '\n    ']` | `['():\n', '    ']` |

- **Digits** are capped at 3 per chunk by GPT-4, so the tokenizer cannot learn
  a token for one specific long number like `65536`.
- **Contractions** are case-insensitive in GPT-4 (`(?i:...)`); GPT-2 only
  matched lowercase, so `IT'S` fragments.
- **Whitespace** is grouped with the preceding newline by GPT-4, which handles
  runs of indentation better. On Python source this is worth a few percent:
  542 tokens (GPT-2) vs 533 (GPT-4) in `main.py`'s comparison.

## The trade-off: chunking costs raw compression

This is the counter-intuitive part. At an **equal budget of 256 merges** on the
same corpus:

| Tokenizer | Compression |
| --- | --- |
| `Basic` (no chunking) | **2.36x** |
| `RegexTokenizer` (GPT-2 pattern) | 2.21x |
| `RegexTokenizer` (GPT-4 pattern) | 2.13x |

Chunking makes compression **worse**, and that is the intended trade. Those
cross-boundary tokens genuinely do pack prose tighter - `" the "` with both
spaces really is efficient. What you buy by giving them up is consistency: a
word tokenizes the same way regardless of what follows it.

So a lower compression ratio here is not a regression. This is pinned by
`test_chunking_costs_raw_compression` so it stays visible.
