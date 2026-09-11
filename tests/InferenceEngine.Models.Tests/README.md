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
