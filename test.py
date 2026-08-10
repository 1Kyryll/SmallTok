"""Tests for the tokenizers, trained on a real corpus.

Runs standalone (`python test.py`) and under pytest, which collects the
test_* functions automatically. No third-party test dependency.
"""

import os
import tempfile

from src.basic import Basic
from src.regex import RegexTokenizer, GPT2_PAT, GPT4_PAT
from src.tokenizer import get_stats, merge, check_merge

CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "taylorswift.txt")

with open(CORPUS_PATH, encoding="utf-8") as f:
    CORPUS = f.read()

# inputs that have historically broken byte-level tokenizers
EDGE_CASES = [
    "",                                  # empty
    "a",                                 # single byte, shorter than any merge
    "hello world",
    "     leading and\ttrailing   ",     # whitespace runs
    "\n\n\r\n",                          # newline handling differs per pattern
    "café 안녕하세요",                    # multi-byte codepoints
    "👋🏽 👨‍👩‍👧‍👦",                          # emoji with modifiers and ZWJ sequences
    "1234567890",                        # GPT-4 splits digits in groups of 3
    "'s 'S 've n't",                     # contraction rules
    "\x00\x01\x7f",                      # control bytes
    CORPUS[:5000],
]


def _trained(cls, *args, num_merges=256):
    tokenizer = cls(*args)
    tokenizer.train(CORPUS, num_merges=num_merges)
    return tokenizer


# built once - training on 186KB is the slow part, so tests share these
BASIC = _trained(Basic)
GPT2 = _trained(RegexTokenizer, GPT2_PAT)
GPT4 = _trained(RegexTokenizer, GPT4_PAT)
ALL = [("Basic", BASIC), ("Regex/GPT2", GPT2), ("Regex/GPT4", GPT4)]


def test_roundtrip():
    """decode(encode(x)) == x for every tokenizer and every edge case."""
    for name, tokenizer in ALL:
        for text in EDGE_CASES:
            assert tokenizer.decode(tokenizer.encode(text)) == text, \
                f"{name} failed to round-trip {text[:40]!r}"


def test_roundtrip_full_corpus():
    for name, tokenizer in ALL:
        assert tokenizer.decode(tokenizer.encode(CORPUS)) == CORPUS, f"{name} corpus round-trip"


def test_training_reached_the_merge_budget():
    """186KB of prose has plenty of repetition, so all 256 merges should land."""
    for name, tokenizer in ALL:
        assert len(tokenizer.merges) == 256, f"{name} only learned {len(tokenizer.merges)}"
        assert len(tokenizer.vocab) == 256 + 256, f"{name} vocab size"


def test_vocab_ids_are_contiguous():
    """A hole in the id space means decode would raise on a valid token."""
    for name, tokenizer in ALL:
        expected = list(range(256 + len(tokenizer.merges)))
        assert sorted(tokenizer.vocab) == expected, f"{name} has gaps in its vocab"


def test_every_merge_is_reachable():
    """Each learned merge should refer only to ids that already existed."""
    for name, tokenizer in ALL:
        for (p0, p1), idx in tokenizer.merges.items():
            assert p0 < idx and p1 < idx, f"{name}: merge {(p0, p1)} -> {idx} refers forward"
            assert tokenizer.vocab[idx] == tokenizer.vocab[p0] + tokenizer.vocab[p1], \
                f"{name}: vocab for {idx} disagrees with its merge"


def test_compression_beats_raw_bytes():
    for name, tokenizer in ALL:
        raw = len(CORPUS.encode("utf-8"))
        encoded = len(tokenizer.encode(CORPUS))
        assert encoded < raw, f"{name} did not compress at all"
        assert raw / encoded > 1.3, f"{name} compressed only {raw / encoded:.2f}x"


def test_untrained_tokenizer_is_identity():
    """With no merges, encoding is just the utf-8 bytes."""
    tokenizer = RegexTokenizer(GPT4_PAT)
    text = "hello world"
    assert tokenizer.encode(text) == list(text.encode("utf-8"))
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_merges_never_cross_chunk_boundaries():
    """The point of the regex: no learned token may span a chunk split."""
    for name, tokenizer in [("Regex/GPT2", GPT2), ("Regex/GPT4", GPT4)]:
        chunks = tokenizer.compiled_pattern.findall(CORPUS)
        allowed = set()
        for chunk in set(chunks):
            encoded = chunk.encode("utf-8")
            for i in range(len(encoded)):
                for j in range(i + 1, len(encoded) + 1):
                    allowed.add(encoded[i:j])

        for idx in range(256, 256 + len(tokenizer.merges)):
            token = tokenizer.vocab[idx]
            assert token in allowed, f"{name}: token {token!r} spans a chunk boundary"


def test_chunking_costs_raw_compression():
    """Chunking makes compression *worse* at an equal merge budget, and that is
    the point: Basic is free to spend merges on cross-boundary junk like " the "
    or "dog.", which packs prose tighter but ties a word's tokenization to its
    neighbours. The regex tokenizers buy consistency with those bytes.

    Pinned as a test so the trade-off stays visible rather than looking like a
    regression the first time someone compares the two ratios.
    """
    assert len(BASIC.merges) == len(GPT4.merges)  # equal budget, fair comparison
    assert len(BASIC.encode(CORPUS)) < len(GPT4.encode(CORPUS))


def test_decode_rejects_unknown_ids():
    try:
        GPT4.decode([999999])
    except ValueError as e:
        assert "not in the vocab" in str(e)
    else:
        raise AssertionError("decode accepted an out-of-vocab id")


def test_decode_of_partial_token_does_not_crash():
    """A token can end mid-codepoint; that must produce U+FFFD, not an exception."""
    ids = GPT4.encode("👋")
    assert GPT4.decode(ids[:1]) == "�" or len(ids) == 1


def test_special_tokens():
    tokenizer = _trained(RegexTokenizer, GPT4_PAT)
    eot = 256 + len(tokenizer.merges)
    tokenizer.register_special_tokens({"<|endoftext|>": eot, "<|fim|>": eot + 1})
    text = "before<|endoftext|>after<|fim|>"

    ids = tokenizer.encode(text, allowed_special="all")
    assert eot in ids and eot + 1 in ids
    assert tokenizer.decode(ids) == text

    plain = tokenizer.encode(text, allowed_special="none")
    assert eot not in plain
    assert tokenizer.decode(plain) == text

    subset = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    assert eot in subset and eot + 1 not in subset

    try:
        tokenizer.encode(text)
    except ValueError:
        pass
    else:
        raise AssertionError("default allowed_special should refuse special-token text")

    # text without specials is unaffected by registration
    assert tokenizer.encode("ordinary") == tokenizer.encode_ordinary("ordinary")


def test_longest_special_token_wins():
    tokenizer = _trained(RegexTokenizer, GPT4_PAT, num_merges=64)
    base = 256 + len(tokenizer.merges)
    tokenizer.register_special_tokens({"<|end|>": base, "<|endoftext|>": base + 1})
    ids = tokenizer.encode("a<|endoftext|>b", allowed_special="all")
    assert base + 1 in ids and base not in ids


def test_special_token_id_collision_is_rejected():
    tokenizer = _trained(RegexTokenizer, GPT4_PAT, num_merges=64)
    try:
        tokenizer.register_special_tokens({"<|bad|>": 5})
    except ValueError as e:
        assert "collides" in str(e)
    else:
        raise AssertionError("a special token id inside the BPE vocab should be rejected")


def test_save_load_roundtrip():
    for name, tokenizer in ALL:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "t")
            tokenizer.save(prefix)
            assert os.path.exists(prefix + ".model") and os.path.exists(prefix + ".vocab")

            reloaded = type(tokenizer)().load(prefix + ".model")
            assert reloaded.merges == tokenizer.merges, f"{name} merges"
            assert reloaded.pattern == tokenizer.pattern, f"{name} pattern"
            assert reloaded.special_tokens == tokenizer.special_tokens, f"{name} specials"
            for text in EDGE_CASES:
                assert reloaded.encode(text) == tokenizer.encode(text), f"{name} encode after load"


def test_save_load_preserves_special_tokens():
    tokenizer = _trained(RegexTokenizer, GPT4_PAT, num_merges=64)
    eot = 256 + len(tokenizer.merges)
    tokenizer.register_special_tokens({"<|endoftext|>": eot})

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "t")
        tokenizer.save(prefix)
        reloaded = RegexTokenizer().load(prefix + ".model")

    text = "a<|endoftext|>b"
    assert reloaded.encode(text, allowed_special="all") == tokenizer.encode(text, allowed_special="all")
    assert reloaded.decode(reloaded.encode(text, allowed_special="all")) == text


def test_load_refuses_a_model_from_another_class():
    """Loading a chunked model into Basic would silently tokenize differently."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "t")
        BASIC.save(prefix)
        try:
            RegexTokenizer().load(prefix + ".model")
        except ValueError as e:
            assert "Basic" in str(e)
        else:
            raise AssertionError("cross-class load should be refused")


def test_load_rejects_non_model_path():
    try:
        RegexTokenizer().load("something.vocab")
    except ValueError as e:
        assert ".model" in str(e)
    else:
        raise AssertionError("load should insist on a .model file")


def test_helpers():
    ids = list(b"abababab")
    stats = get_stats(ids)
    assert stats[(97, 98)] == 4

    merged = merge(ids, (97, 98), 256)
    assert merged == [256, 256, 256, 256]
    assert check_merge(ids, merged, (97, 98), 256, verbose=False)

    # accumulating into an existing dict, as training does across chunks
    shared = {}
    get_stats([1, 2], shared)
    get_stats([1, 2], shared)
    assert shared[(1, 2)] == 2

    # a pair that overlaps itself must be consumed left to right, not double-counted
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]


def test_matches_tiktoken_gpt2():
    """The strongest check available: byte-identical output to the real thing."""
    try:
        import tiktoken

        from src.gpt2 import GPT2Tokenizer
    except ImportError:
        print("  skipped (tiktoken not installed)")
        return

    tokenizer = GPT2Tokenizer.from_tiktoken()
    reference = tiktoken.get_encoding("gpt2")

    assert len(tokenizer.merges) == 50000
    for text in EDGE_CASES + [CORPUS]:
        assert tokenizer.encode(text) == reference.encode(text), \
            f"diverged from tiktoken on {text[:40]!r}"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    special = "a<|endoftext|>b"
    assert tokenizer.encode(special, allowed_special="all") == \
        reference.encode(special, allowed_special="all")

    # a saved GPT-2 model must reload without tiktoken present
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "gpt2")
        tokenizer.save(prefix)
        reloaded = GPT2Tokenizer().load(prefix + ".model")
    assert reloaded.byte_shuffle == tokenizer.byte_shuffle
    assert reloaded.encode(CORPUS) == tokenizer.encode(CORPUS)


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]

    print(f"corpus: {len(CORPUS):,} chars / {len(CORPUS.encode('utf-8')):,} bytes")
    for name, tokenizer in ALL:
        ratio = len(CORPUS.encode("utf-8")) / len(tokenizer.encode(CORPUS))
        print(f"  {name:<12} {len(tokenizer.merges)} merges, {ratio:.2f}x compression")
    print()

    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {name}\n        {e}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
