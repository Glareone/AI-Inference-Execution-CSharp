# InferenceEngine.Engine.Tests

Tests `src/InferenceEngine.Engine`: the KV-cache, the sampling pipeline, and chat prompt
assembly.

## Business scenarios covered

### `SimpleKvCache` (`SimpleKvCacheTests.cs`)

- A key/value slot written for a given `(layer, position)` is read back unchanged.
- Different `(layer, position)` pairs never overlap — writing to one slot never leaks into or
  overwrites another, which is what makes causal attention over history correct.

### `Sampling/SamplingPipeline`, `TopKStep`, `TopPStep` (`Sampling/*.cs`)

- Greedy decoding (`Temperature: 0`) always returns the single most probable token.
- Top-k narrows the candidate pool to exactly `k` surviving logits; the rest become
  `float.NegativeInfinity`.
- Top-p (nucleus) sampling keeps the smallest prefix of sorted probabilities whose cumulative
  mass reaches the threshold. The test picks logits as `ln(probability)` for a known distribution
  so the exact cutoff point can be reasoned about directly rather than inferred from the code.
- Sampling is reproducible: the same seed and logits always produce the same chosen token.
- Sampling never mutates the caller's input logits span (the pipeline works on its own scratch
  buffer).

### `Prompting/ChatMlTemplate`, `ChatPromptBuilder`, `IChatPromptStep` (`Prompting/*.cs`)

Exercised together as the single unit a caller depends on, using a hand-written fake
`ITokenizer` (`FakeTokenizer.cs` — the interface is small enough that a mocking library isn't
worth the dependency) that "encodes" text as one id per character, so the expected id sequence
for any turn can be computed directly in the test.

- The assembled prompt follows the exact system -> user -> assistant-priming turn order a
  chat-tuned model expects, wrapped in the model's own special tokens (`<|im_start|>`,
  `<|im_end|>`).
- A special token missing from the tokenizer's vocabulary fails with `KeyNotFoundException`
  instead of silently producing a malformed prompt.
