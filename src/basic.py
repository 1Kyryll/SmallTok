from src.tokenizer import Tokenizer
from src.tokenizer import get_stats, merge

class Basic(Tokenizer):
    """Basic tokenizer implementation that uses a simple byte-level BPE algorithm."""

    def __init__(self, merge_count=20):
        super().__init__(merge_count=merge_count)

    def encode(self, text):
        """Encode a string into a list of token ids."""
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2: 
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def decode(self, ids):
        """Decode a list of token ids back into a string."""
        tokens = b"".join(self.vocab[i] for i in ids)
        return tokens.decode("utf-8", errors="replace")