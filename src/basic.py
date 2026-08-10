from src.tokenizer import Tokenizer


class Basic(Tokenizer):
    """Basic tokenizer implementation that uses a simple byte-level BPE algorithm.

    No chunking at all: merges are free to span any two adjacent bytes,
    including across word and whitespace boundaries.
    """

    def train(self, text, num_merges, verbose=False):
        """Learn `num_merges` merges from `text`."""
        self._train_chunks([text.encode("utf-8")], num_merges, verbose)

    def encode(self, text):
        """Encode a string into a list of token ids."""
        return self._encode_chunk(text.encode("utf-8"))
