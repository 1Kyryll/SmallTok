"""Load OpenAI's real GPT-2 tokenizer into our own classes.

tiktoken ships a flat `bytes -> rank` table, not the (int, int) -> int merge
list our BPE loop needs. `recover_merges` reconstructs the merge list by
replaying BPE on each token's own bytes.
"""

from src.regex import RegexTokenizer, GPT2_PAT


def bpe_split(mergeable_ranks, token_bytes, max_rank):
    """Re-derive which two pieces merged to create `token_bytes`, by redoing
    BPE on its own bytes with every merge at or above `max_rank` forbidden."""
    parts = [bytes([b]) for b in token_bytes]
    while True:
        best_i, best_rank = None, None
        for i in range(len(parts) - 1):
            rank = mergeable_ranks.get(parts[i] + parts[i + 1])
            if rank is not None and (best_rank is None or rank < best_rank):
                best_i, best_rank = i, rank
        if best_i is None or best_rank >= max_rank:
            break
        parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return parts


def recover_merges(mergeable_ranks):
    """tiktoken's `bytes -> rank` table -> our `(id, id) -> id` merge dict."""
    merges = {}
    for token_bytes, rank in mergeable_ranks.items():
        if len(token_bytes) == 1:
            continue  # a raw byte, not a merge result
        p0, p1 = bpe_split(mergeable_ranks, token_bytes, max_rank=rank)
        merges[(mergeable_ranks[p0], mergeable_ranks[p1])] = rank
    return merges


class GPT2Tokenizer(RegexTokenizer):
    """RegexTokenizer preloaded with GPT-2's merges, so it reproduces
    `tiktoken.get_encoding("gpt2")` exactly.

    GPT-2 does not assign byte 0 to id 0 - `mergeable_ranks` permutes the 256
    raw bytes - so we carry that permutation as `byte_shuffle` and undo it
    when decoding.
    """

    def __init__(self):
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        ranks = enc._mergeable_ranks

        super().__init__(GPT2_PAT, byte_shuffle={i: ranks[bytes([i])] for i in range(256)})
        self.merges = recover_merges(ranks)
        self.vocab = self._build_vocab()
