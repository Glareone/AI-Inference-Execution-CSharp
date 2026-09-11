# InferenceEngine.Tokenizers.Tests

Tests `src/InferenceEngine.Tokenizers`: the hand-rolled byte-level BPE tokenizer and its
supporting byte<->char alphabet.

## Business scenarios covered

### `GgufBpeTokenizer` (`GgufBpeTokenizerTests.cs`)

- Text survives an `Encode` then `Decode` round trip back to the original string.
- The BPE merge loop always applies the globally lowest-rank merge available, not just the first
  adjacent pair it happens to scan — verified with a vocab/merge list (`a`,`b`,`c` with merges
  `["b c", "a b"]`) constructed so the two orderings would produce different results if the
  implementation picked the wrong one.
- The "smollm" pre-tokenizer isolates digits into their own segments before the main
  letter/punctuation pattern runs: `"a1b"` produces three separate tokens, not one merged chunk.
- Special/literal tokens (e.g. `<|im_start|>`) are found by their exact text via `TryGetId`, and
  an unknown string correctly reports "not found" rather than throwing.
- An unrecognized `PreTokenizerName` fails clearly at construction (`NotSupportedException`)
  instead of silently mis-tokenizing later.

### `ByteLevelAlphabet` (`ByteLevelAlphabetTests.cs`)

- Every byte value 0-255 round-trips through `ByteToChar` then `CharToByte` back to itself — the
  bijection that lets BPE merge rules be expressed as plain text.
