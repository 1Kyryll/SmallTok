"""Base tokenizer class with helper functions for tokenization."""

import re


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


def check_merge(ids, new_ids, pair, idx, verbose=True):
    """Check that a merge was done correctly. Returns True if all checks pass."""
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

    if verbose:
        print(f"occurrences found:   {occurrences}")
        print(f"pair left in output: {remaining}  (expected 0)")
        print(f"len {len(ids)} -> {len(new_ids)}, expected {len(ids) - occurrences}  {'OK' if len_ok else 'FAIL'}")
        print(f"round-trip:          {'OK' if expanded == ids else 'FAIL'}")
        print(f"idx count:           {new_ids.count(idx)}  (expected {occurrences})")

    return len_ok and remaining == 0 and expanded == ids and new_ids.count(idx) == occurrences


class Tokenizer:
    """Base byte-level BPE tokenizer.

    Subclasses decide how the text is split up before training/encoding
    (not at all for `Basic`, on a regex for `RegexTokenizer`); everything
    else - the merge loop, the vocab and decoding - lives here.
    """

    def __init__(self):
        self.merges = {}            # (int, int) -> int
        self.special_tokens = {}    # str -> int, e.g. {"<|endoftext|>": 50256}
        self.inverse_special_tokens = {}
        self.byte_shuffle = None    # optional raw byte -> id permutation (for GPT-2/4 compat)
        self.inverse_byte_shuffle = None
        self.vocab = self._build_vocab()

    def register_special_tokens(self, special_tokens):
        """Register out-of-band tokens like "<|endoftext|>".

        These never come out of BPE - they live above the merged vocab and are
        spliced in by `encode`, so their text can never be split up or merged
        into anything else.
        """
        first_free = 256 + len(self.merges)
        for token, idx in special_tokens.items():
            if idx < first_free:
                raise ValueError(
                    f"special token {token!r} wants id {idx}, which collides with "
                    f"the BPE vocab (ids 0..{first_free - 1})")

        self.special_tokens = dict(special_tokens)
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _build_vocab(self):
        """Derive the id -> bytes table from the merges. Order matters: a merge
        can only reference ids that were created before it."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def _to_ids(self, text_bytes):
        """Raw bytes -> starting ids. The single place `byte_shuffle` is applied,
        so training and encoding can never disagree about it."""
        if self.byte_shuffle is not None:
            return [self.byte_shuffle[b] for b in text_bytes]
        return list(text_bytes)

    def _train_chunks(self, chunks, num_merges, verbose=False):
        """Run `num_merges` BPE merges over a list of `bytes` chunks.

        Pair counts are accumulated across ALL chunks, but merging happens
        within each chunk only - that is what stops a merge from ever
        spanning a chunk boundary.
        """
        self.merges = {}
        ids_list = [self._to_ids(chunk) for chunk in chunks]

        for i in range(num_merges):
            stats = {}
            for ids in ids_list:
                get_stats(ids, stats)
            if not stats:
                break                          # nothing left to merge

            top_pair = max(stats, key=stats.get)
            if stats[top_pair] < 2:
                break                          # every pair is unique, merging buys nothing

            new_id = 256 + i
            ids_list = [merge(ids, top_pair, new_id) for ids in ids_list]
            self.merges[top_pair] = new_id

            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {new_id} "
                      f"({stats[top_pair]} occurrences)")

        self.vocab = self._build_vocab()
        return ids_list

    def _encode_chunk(self, text_bytes):
        """BPE-encode a bytes object, applying merges in the order they were learned."""
        ids = self._to_ids(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            # the eligible pair with the lowest merge index = the earliest merge learned
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            ids = merge(ids, pair, self.merges[pair])

        return ids

    def train(self, text, num_merges, verbose=False):
        """Learn `num_merges` merges from `text`."""
        raise NotImplementedError("Subclasses should implement this method.")

    def encode_ordinary(self, text):
        """Encode a string, treating special-token text as ordinary text."""
        raise NotImplementedError("Subclasses should implement this method.")

    def encode(self, text, allowed_special="none_raise"):
        """Encode a string into a list of token ids.

        `allowed_special` controls what happens to registered special tokens:
          "none_raise"  don't recognise them, but refuse text containing one
                        (the default - stops user text smuggling in a control
                        token, which is why tiktoken defaults the same way)
          "all"         recognise every registered special token
          "none"        encode them as ordinary text
          a collection  recognise only those
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(
                        f"text contains special token {token!r}; pass "
                        f"allowed_special='all' to encode it as a token, or "
                        f"'none' to encode it as plain text")
        else:
            special = {k: v for k, v in self.special_tokens.items() if k in set(allowed_special)}

        if not special:
            return self.encode_ordinary(text)

        # longest first, so "<|end|>" can't shadow "<|endoftext|>"
        pattern = "(" + "|".join(re.escape(k) for k in sorted(special, key=len, reverse=True)) + ")"

        ids = []
        for part in re.split(pattern, text):
            if part in special:
                ids.append(special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        """Decode a list of token ids back into a string."""
        parts = []
        for idx in ids:
            if idx in self.vocab:
                chunk = self.vocab[idx]
                if self.inverse_byte_shuffle is not None:
                    chunk = bytes(self.inverse_byte_shuffle[b] for b in chunk)
                parts.append(chunk)
            elif idx in self.inverse_special_tokens:
                # a special token carries its own text, and is never byte-shuffled
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"id {idx} is not in the vocab "
                                 f"(vocab size is {len(self.vocab)})")

        # a token can end mid-codepoint, so invalid utf-8 is expected, not a bug
        return b"".join(parts).decode("utf-8", errors="replace")
