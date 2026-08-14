# Encoding, Decoding, and Why Merge Order Matters

Training and encoding are two different operations, and it is easy to conflate
them.

- **Training** looks at a corpus and *decides* which pairs to merge, in what
  order. It happens once.
- **Encoding** takes that frozen list of merges and *replays* it on new text.
  It learns nothing.

## Encoding is not longest-match

The tempting way to encode is greedily: scan the vocabulary and take the
longest token that fits. **That is wrong**, and it will not reproduce a real
tokenizer.

Encoding has to apply the merges in the exact order they were learned. If
merge #5 built `"th"` and merge #40 built `"the"`, then `"the"` can only exist
because `"th"` was formed first. Applying #40 before #5 would find nothing to
merge.

So `_encode_chunk` repeatedly asks: of all the pairs currently present, which
one has the **lowest merge index**? That is the earliest-learned applicable
merge, and it goes next.

```python
pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
if pair not in self.merges:
    break
```

The `float("inf")` default is the trick that makes this one line: any pair that
was never learned sorts last, so if the winner is still unlearned, no pair is
applicable at all and we are done.

This is why the merge dict's *insertion order is the model*. It is also why
[save/load](../src/tokenizer.py) writes merges in order and rebuilds their ids
from position, rather than storing the ids.

## Decoding is just a lookup

Decoding is far simpler and cannot fail to find a token: every id maps to a
byte string in `vocab`, built by concatenating the two halves of its merge
(see [bpe_algorithm.md](bpe_algorithm.md)). Concatenate the bytes, decode UTF-8.

The one subtlety is that **a token can end mid-codepoint**. Tokens are byte
sequences and know nothing about character boundaries, so a valid id sequence -
say the first token of `👋` alone - decodes to an incomplete UTF-8 character.

That is expected, not a bug, so decoding uses `errors="replace"` and yields
`�` rather than raising. This matters in practice: it is exactly what happens
when a model streams output token by token, which is why you sometimes see a
replacement character flicker before the rest of an emoji arrives.

## Special tokens skip BPE entirely

Tokens like `<|endoftext|>` are not learned and never appear in `merges`. They
live above the BPE vocabulary and are spliced in by id, so their text can never
be merged into a neighbour or split apart.

Because of that, `encode` refuses text containing a special token by default:

```python
tokenizer.encode(text)                          # raises if text contains one
tokenizer.encode(text, allowed_special="all")   # recognise them
tokenizer.encode(text, allowed_special="none")  # encode as ordinary text
```

The strict default is deliberate. If user-supplied text were silently encoded
with a real `<|endoftext|>` id in it, that text could forge a document boundary
or the end of a turn - a prompt injection at the tokenizer level. tiktoken
defaults the same way.
