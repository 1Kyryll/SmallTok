import regex as re

from src.tokenizer import Tokenizer

# GPT-2 (Radford et al.)
GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# GPT-4 (cl100k_base) - better
GPT4_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """Regex-based tokenizer.

    The text is first split on a regex, and BPE runs inside each chunk
    independently. That keeps merges from crossing category boundaries,
    so you never learn a token like "dog." or " the12".
    """

    def __init__(self, pattern=GPT4_PAT, byte_shuffle=None):
        super().__init__()
        self._set_pattern(pattern)
        if byte_shuffle is not None:
            self.byte_shuffle = byte_shuffle
            self.inverse_byte_shuffle = {v: k for k, v in byte_shuffle.items()}

    def _set_pattern(self, pattern):
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)

    def train(self, text, num_merges, verbose=False):
        """Learn `num_merges` merges from `text`, one id-list per regex chunk."""
        chunks = [chunk.encode("utf-8") for chunk in self.compiled_pattern.findall(text)]
        self._train_chunks(chunks, num_merges, verbose)

    def encode_ordinary(self, text):
        """Encode a string chunk by chunk, treating special-token text as ordinary."""
        ids = []
        for chunk in self.compiled_pattern.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids
