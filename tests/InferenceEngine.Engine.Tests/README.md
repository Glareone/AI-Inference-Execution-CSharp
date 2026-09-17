# InferenceEngine.Engine.Tests

Tests `src/InferenceEngine.Engine`: the KV-cache, the sampling pipeline, and chat prompt
assembly.

## Business scenarios covered

### `KvBlockPool` (`KvBlockPoolTests.cs`)

The physical block allocator every `PagedKvCache` session shares — see the
[kv-cache ADR](../../docs/architecture/260914-kv-cache.md).

- A fresh pool allocates ascending physical block ids (0, 1, 2, …), so a single sequence's
  physical bytes land in the same order a contiguous cache would.
- Exhausting the pool throws `InvalidOperationException` naming the pool's block capacity,
  not a silent wraparound.
- Freeing a block and re-allocating it returns the same id with its backing arrays retained
  (not zeroed, not reallocated) — the property that makes `Reset()`/`Rollback()` allocation-free.
- `(layer, kvHead, slot)` offsets never overlap within one physical block's array.

### `PagedKvCache` (`PagedKvCacheTests.cs`)

- A key/value slot written for a given `(layer, kvHead, position)` is read back unchanged, and
  a tile read via `KeyBlockForHead`/`ValueBlockForHead` lines up with the same per-slot writes —
  including across a logical block boundary.
- Different `(layer, kvHead, position)` triples never overlap, which is what makes causal
  attention over history correct.
- `Reserve` is idempotent within a logical block and consumes exactly one physical block when
  crossing into a new one — not one physical block per `Reserve` call.
- Reading a slot before it's been `Reserve`d throws clearly, instead of silently returning
  stale or out-of-range data.
- `Rollback(toPosition)` frees exactly the logical blocks past the kept boundary; the freed
  physical ids become available for a later `Reserve` to reuse, and kept data survives untouched.
- `Reset()` returns every assigned block to the pool.
- The block size must be a power of two — the constructor rejects anything else.

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
- **Regression**: banning the single highest-logit token still changes the result under greedy
  decoding (`Temperature: 0`) — before this fix, `SamplingPipeline` only ran its
  `ISamplerStep` chain when `Temperature > 0`, so a ban could never reach the default, greedy path
  at all.
- All-masked fallback: if every logit is `float.NegativeInfinity` after logits processors run,
  `Sample` returns the EOS token id directly, on both the greedy path and the stochastic path.
  The stochastic case is also the regression test for a NaN-propagation bug: `Exp(-inf - -inf)`
  is `NaN`, and NaN comparisons are always `false`, so the pre-fix cumulative-probability loop
  fell through to an arbitrary last index instead of EOS.

### `Sampling/BannedSequenceLogitsProcessor` (`Sampling/BannedSequenceLogitsProcessorTests.cs`)

Bans literal words/phrases from ever being generated (e.g. a competitor's name), applied as a
deterministic hard constraint *before* `SamplingPipeline`'s probabilistic sampler steps run —
see the [logits-processing ADR](../../docs/architecture/260917-logits-processing.md).

- A single-token banned sequence is always masked, regardless of what's been generated so far.
- A multi-token banned sequence (the common case — most real words BPE-split into more than one
  token) is masked only once the tokens generated so far end with that sequence's prefix, not
  before.
- Multiple independent banned sequences apply correctly together in one `Apply` call, and a
  sequence longer than the history generated so far is simply not yet reachable — it does not
  throw.
- The EOS token can never be banned, whether it's named directly as a length-1 sequence or as the
  completing token of a longer one whose prefix matches — the invariant `SamplingPipeline`'s
  all-masked fallback depends on.
- `FromWords` silently drops a word that tokenizes to a single EOS token, without affecting any
  other word's ban.

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
