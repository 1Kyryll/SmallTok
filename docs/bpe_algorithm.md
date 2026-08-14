# Byte-Pair Encoding Algorithm 

This whole repo is based on that single algorithm. It was initially developed in 1994 for data compression purposes, yet was adopted later by GPT series and others particularly as a main logic for tokenizers. 

# The Logic Behind It

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
Vocab = { Z: Yb }
```