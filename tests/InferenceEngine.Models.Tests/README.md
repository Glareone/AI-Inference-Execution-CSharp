# InferenceEngine.Models.Tests

Tests `src/InferenceEngine.Models`: the hand-rolled GGUF file reader and the transformer math
primitives it feeds.

## Business scenarios covered

### `Gguf/GgufFile` (`GgufFileTests.cs`, fixtures via `GgufFileBuilder.cs`)

GGUF has no maintained NuGet parser, so this reader is hand-rolled (see the format-loading ADR).
Fixtures are synthetic GGUF v3 files built byte-for-byte with `BinaryWriter` in
`GgufFileBuilder`, not a downloaded model, so every case is small and deterministic.

- Scalar metadata (string/u32/f32) round-trips exactly; a missing key falls back to the caller's
  supplied default instead of throwing.
- Array metadata (string arrays and int32 arrays) round-trips exactly.
- An F32 tensor's bytes come back unchanged.
- An F16 tensor is dequantized to F32 within floating-point tolerance.
- A quantized type this POC doesn't support (Q4_0, ggml type 2) parses as metadata/shape but
  throws `NotSupportedException` on `ReadTensorAsF32` instead of silently returning wrong data.
- An unknown tensor name throws `KeyNotFoundException`.
- Two adjacent tensors placed after a header whose length isn't itself a multiple of the default
  32-byte alignment read back independently and correctly — proving the data-section alignment
  padding is computed correctly and neither tensor's bytes leak into the other's.
- A `general.alignment` metadata override (64 instead of the 32 default) is honored, not just
  parsed.

### `Math/Ops` (`OpsTests.cs`)

The attention/FFN math primitives, checked against hand-computed values rather than mirrored
implementation logic:

- `RmsNorm` on `x=[3,4]`, `weight=[1,1]`, `eps=0` matches `invRms = 1/sqrt(12.5)`.
- `MatVec` on a small hand-written row-major `[out,in]` matrix matches manual dot products.
- `Rope` at `position=0` is an exact no-op for any `freqBase` (the rotation angle is zero), and
  at a non-zero position rotates by the expected angle.
- `Softmax` output sums to 1 and matches the standard `[1,2,3]` softmax constants.
- `SwiGlu` satisfies `Silu(0) = 0` and matches a hand-computed value at `gate=1, up=1`.

Note: `Ops` lives in a namespace literally named `Math` (shadowing `System.Math` once imported),
so `OpsTests.cs` only ever uses `MathF` for scalar math to avoid the ambiguity.

### `Llama/GoldenLogitBaselineTests` — golden KV-cache logit baseline

A frozen-in-time correctness oracle captured against the pre-rewrite `SimpleKvCache`
implementation, before the KV-cache rewrite in `docs/architecture/260914-kv-cache.md` begins:
prefilling the real SmolLM2-135M-Instruct GGUF model with a fixed short prompt and a fixed long
prompt (long enough to span several future 32-token cache blocks) always produces the identical
next-token logit distribution, bit-for-bit. Every later step of the rewrite must reproduce the
hardcoded `Fnv1a` hashes in that file exactly; the top-10 `(tokenId, logit)` pairs are captured
alongside for human debuggability if a hash ever mismatches. `LogitHash.Fnv1a` (hashing raw
IEEE-754 bit patterns, not decimal text) is written as a small reusable static method because a
later end-to-end golden test, after the rewrite, needs to compute the same hash for comparison.

Requires the real 270 MB model and is slow (a few seconds of real prefill), so it skips cleanly —
not a failure — unless `INFERENCE_MODEL` is set to an existing GGUF file path:

```
INFERENCE_MODEL=~/.cache/inference-engine/models/SmolLM2-135M-Instruct-f16.gguf dotnet test ...
```

(with `~` already expanded by the shell). This is the only test in this project that references
`InferenceEngine.Engine` in addition to `InferenceEngine.Models` — it needs
`InferenceSession.PrefillTopLogits` (the exact code path `InferenceEngine.Cli`'s debug-logits
flag drives) because the chat-template wrapping that produces the fixed 164-token long prompt is
internal to `Engine` and can't be reconstructed from `Models` alone.
