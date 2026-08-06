from src.basic import Basic

print("Testing Basic Tokenizer")
basic = Basic(merge_count=20)

raw = "Hello, world! This is a test of the Basic Tokenizer. Let's see how it performs with some sample text."
print(f"Raw text: {raw}")

encoded = basic.encode(raw)
print(f"Encoded: {encoded}")

decoded = basic.decode(encoded)
print(f"Decoded: {decoded}")

print(f"Encoding and decoding successful: {raw == decoded}")