"""Base tokenizer class with helper functions for tokenization."""

def get_stats(ids, counts=None): 
    """Count common pair appearances in a list of ids."""
    counts = {} if counts is None else counts

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts 

def merge(ids, pair, idx): 
    """Merge a pair into new idx. E.g. (116, 32) -> 256"""
    newids = []
    i = 0 

    while i < len(ids): 
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]: 
            newids.append(idx)
            i += 2
        else: 
            newids.append(ids[i])
            i += 1

    return newids 


def check_merge(ids, new_ids, pair, idx):
    """Check that a merge was done correctly."""
    remaining = sum(1 for a, b in zip(new_ids, new_ids[1:]) if (a, b) == pair)
    
    occurrences = 0
    i = 0
    while i < len(ids) - 1:
        if (ids[i], ids[i+1]) == pair:
            occurrences += 1
            i += 2
        else:
            i += 1
    
    len_ok = len(new_ids) == len(ids) - occurrences

    expanded = []
    for t in new_ids:
        expanded.extend(pair if t == idx else [t])
    
    print(f"occurrences found:   {occurrences}")
    print(f"pair left in output: {remaining}  (expected 0)")
    print(f"len {len(ids)} -> {len(new_ids)}, expected {len(ids) - occurrences}  {'OK' if len_ok else 'FAIL'}")
    print(f"round-trip:          {'OK' if expanded == ids else 'FAIL'}")
    print(f"idx count:           {new_ids.count(idx)}  (expected {occurrences})")

class Tokenizer: 
    """Base tokenizer class with helper functions for tokenization."""

    def __init__(self, merge_count=20):
        self.merges_count = merge_count
        self.vocab = self._build_vocab() 
        self.merges = {}

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items(): 
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def encode(self, text):
        """Encode a string into a list of token ids."""
        raise NotImplementedError("Subclasses should implement this method.")

    def decode(self, ids):
        """Decode a list of token ids back into a string."""
        raise NotImplementedError("Subclasses should implement this method.")

    