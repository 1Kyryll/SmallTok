# Byte-Pair Encoding Algorithm 

This whole repo is based on that single algorithm. It was initially developed in 1994 for data compression purposes, yet was adopted later by GPT series and others particularly as a main logic for tokenizers. 

## The Logic Behind It

BPE bridges a gap between character-level and word-level tokenization by iteratively building a vocabulary of common character sequences. 

1. The process begins by representing text as a sequence of **UTF-8 bytes**, which gives the "initial alphabet" of 256 possible symbols. 
2. Later the algorithm scans for the most frequent pairs(`get_stats()` in the repo). 
3. After identifying the most frequent one, the pair is replaced with a single symbol(starting from 256). 
4. The process repeats until a merges count reaches desired value. 
5. As a result, BPE significantly **shrinks the sequence length** that the Transformer must process, which is critical because computation cost grows quadratically with sequence length.

Example of BPE algorithm: 

```plaintext
1. aabdaabc -> aa (2x) ab (2x) 
Vocab = {}
2. Replace aa with Y -> YbdYbc -> Yb (2x)
Vocab = { Y: aa }
3. Replace Yb with Z -> ZdZc 
Vocab = { Y: aa, Z: Yb }
```

The vocabulary only ever **grows** - nothing is removed. That matters because a
merge is allowed to refer to symbols built by earlier merges (`Z` is built out
of `Y`), so every token can be expanded all the way back down to raw bytes.
That is exactly what makes decoding a plain table lookup.

Note that step 1 had a tie: `aa` and `ab` both appear twice. Which one wins is
arbitrary, but it must be **deterministic**, or the same text would train two
different tokenizers. This repo takes whichever `max()` finds first.

## In this repo

| Step | Code |
| --- | --- |
| Count pairs | `get_stats()` in [../src/tokenizer.py](../src/tokenizer.py) |
| Replace a pair | `merge()` |
| The loop above | `Tokenizer._train_chunks()` |

See also [tokenization_basics.md](tokenization_basics.md) for why the initial
alphabet is bytes, [chunking.md](chunking.md) for why the text is split up
first, and [encoding.md](encoding.md) for how the learned merges get replayed.