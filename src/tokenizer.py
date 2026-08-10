"""Base tokenizer class with helper functions for tokenization."""

import re
import unicodedata

MODEL_VERSION = "smalltok-v1"


def render_token(token_bytes):
    """Show a token's bytes readably: undecodable bytes and control characters
    become escapes, so one token always prints as one line."""
    text = token_bytes.decode("utf-8", errors="replace")
    return "".join(ch if unicodedata.category(ch)[0] != "C" else repr(ch)[1:-1]
                   for ch in text)


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
        self.pattern = ""           # split pattern, empty when the text isn't chunked
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

    def _set_pattern(self, pattern):
        """Hook so `load` can restore the split pattern on subclasses that use one."""
        self.pattern = pattern

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

    def save(self, prefix):
        """Write `prefix.model` (reloadable) and `prefix.vocab` (for humans).

        Only the .model file is ever read back. The .vocab file is lossy - it
        renders undecodable bytes - and exists purely to inspect what was
        learned.
        """
        with open(prefix + ".model", "w", encoding="utf-8") as f:
            f.write(f"{MODEL_VERSION} {type(self).__name__}\n")
            f.write(self.pattern + "\n")

            f.write(f"{len(self.special_tokens)}\n")
            for token, idx in self.special_tokens.items():
                f.write(f"{token} {idx}\n")

            if self.byte_shuffle is None:
                f.write("-\n")
            else:
                f.write(" ".join(str(self.byte_shuffle[i]) for i in range(256)) + "\n")

            # ids are implied by position (the first merge is 256), so only the pair is stored
            for (p0, p1), _ in sorted(self.merges.items(), key=lambda kv: kv[1]):
                f.write(f"{p0} {p1}\n")

        inverted = {idx: pair for pair, idx in self.merges.items()}
        with open(prefix + ".vocab", "w", encoding="utf-8") as f:
            for idx, token in sorted(self.vocab.items()):
                if idx in inverted:
                    p0, p1 = inverted[idx]
                    f.write(f"[{render_token(self.vocab[p0])}][{render_token(self.vocab[p1])}] -> "
                            f"[{render_token(token)}] {idx}\n")
                else:
                    f.write(f"[{render_token(token)}] {idx}\n")
            for token, idx in self.special_tokens.items():
                f.write(f"[{token}] {idx} (special)\n")

    def load(self, path):
        """Restore a tokenizer written by `save`, in place."""
        if not path.endswith(".model"):
            raise ValueError(f"expected a .model file, got {path!r}")

        with open(path, "r", encoding="utf-8") as f:
            version, class_name = f.readline().rstrip("\n").split(" ", 1)
            if version != MODEL_VERSION:
                raise ValueError(f"{path} is {version}, this build reads {MODEL_VERSION}")
            # a regex model loaded into Basic would tokenize differently and silently
            if class_name != type(self).__name__:
                raise ValueError(f"{path} was saved from {class_name}, "
                                 f"cannot load it into {type(self).__name__}")

            self._set_pattern(f.readline().rstrip("\n"))

            special_tokens = {}
            for _ in range(int(f.readline())):
                token, idx = f.readline().rstrip("\n").rsplit(" ", 1)
                special_tokens[token] = int(idx)

            shuffle_line = f.readline().rstrip("\n")
            if shuffle_line == "-":
                self.byte_shuffle = self.inverse_byte_shuffle = None
            else:
                self.byte_shuffle = {i: int(v) for i, v in enumerate(shuffle_line.split())}
                self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

            self.merges = {}
            for i, line in enumerate(f):
                p0, p1 = line.split()
                self.merges[(int(p0), int(p1))] = 256 + i

        self.vocab = self._build_vocab()
        self.register_special_tokens(special_tokens)
        return self
