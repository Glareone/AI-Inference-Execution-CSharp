# Attention and transformer forward pass

- Status: accepted
- Date: 2026-09-17

## Where this problem is isolated

Owned by **`InferenceEngine.Models`**, using tensor types from **`Core`**:

- **Lives here:** the layer wiring (embeddings, RMSNorm, RoPE, GQA attention, SwiGLU FFN, LM
  head) and every call into the math library. `Ops` (`src/InferenceEngine.Models/Math/Ops.cs`)
  holds the primitives; `GqaAttention` (`src/InferenceEngine.Models/Llama/GqaAttention.cs`) holds
  the attention loop as its own testable unit; `LlamaModel.Forward`
  (`src/InferenceEngine.Models/Llama/LlamaModel.cs:109`) wires them into the per-layer pipeline
  (see Context).
- **Exposed as:** `IModel.Forward(int tokenId, int position, IKvCache kvCache, bool needLogits) ->
  ReadOnlySpan<float>` (`Core`). This corrects the placeholder's stated shape: the call is
  **single-token, not batched** — one token id and one position in, not `tokens`/`positions`
  arrays. Prefill is a caller loop of single-token `Forward` calls (outside this ADR's scope), not
  a batched path inside `Models`.
- **Boundary:** the KV-cache is passed in, owned by `Engine`, contract defined in `Core` — so the
  forward pass and the cache stay independently swappable. Unchanged from the placeholder.

## Context and Problem Statement

This ADR retroactively records a decision already made, implemented, tested, and lived with
through two restructuring rounds (the KV-cache ADR's per-KV-head split and loop reorder) — not a
live fork to resolve.

**Does this engine have an attention mechanism?** Yes: **grouped-query attention (GQA) with
rotary position embeddings (RoPE)**. `NumAttentionHeads` query heads share `NumKvHeads` key/value
heads, `groupSize = NumAttentionHeads / NumKvHeads` query heads per KV head. The constructor
rejects a non-exact ratio outright (`LlamaModel.cs:42-48`, `NotSupportedException`). The same
formula degrades to plain multi-head attention (`groupSize == 1`) and multi-query attention
(`NumKvHeads == 1`) — no separate code paths, just edge-case ratios of one formula. Only GQA 3:1
has run against a real model (SmolLM2-135M: 9 query heads, 3 KV heads); MHA and MQA are reachable
by the same formula, not separately tested.

**Per-layer pipeline** (`LlamaModel.Forward`, `LlamaModel.cs:109-179`, pre-norm, once per
`Config.NumLayers`): RMSNorm (`Ops.RmsNorm`, `Ops.cs:12-18`) → Q/K/V projection (`Ops.MatVec`,
`Ops.cs:25-31`) → RoPE per head (`LlamaModel.cs:136-139` for Q; `:146-154` for K/V, written
directly into the paged, head-major KV-cache) → `GqaAttention.Attend` (`GqaAttention.cs:30-99`:
causal via loop bound, no materialized mask; `scale = 1/sqrt(headDim)`, `Ops.Softmax`) → output
projection + residual (`LlamaModel.cs:160-161`) → RMSNorm → SwiGLU (`Ops.SwiGlu`, `Ops.cs:100-106`:
gate/up projection, `Silu(gate) * up`, down-projection) → residual (`:163-168`). After all layers:
final RMSNorm + LM head `MatVec` to `VocabSize` logits, but only when `needLogits` — prefill skips
this for every position except the last (a real perf win the caller relies on). The LM head weight
is the token-embedding matrix when the GGUF omits `output.weight` (tied embeddings, e.g. SmolLM2;
`LlamaWeights.Load`, `LlamaWeights.cs:38-42`), otherwise a separate matrix.

The RoPE rotation-angle table (`Ops.RopeTable`) is built once per `Forward` call
(`LlamaModel.cs:125`), not once per layer — it depends only on `(position, freqBase, headDim)` —
cutting 60 redundant table rebuilds per token to 1 on a 30-layer model.

**RoPE layout:** interleaved-pair / GPT-J-style rotation, **not** HF's half-split. `Ops.Rope`'s doc
comment (`Ops.cs:33-38`) explains why: `convert_hf_to_gguf.py` permutes GGUF llama weights
specifically so this layout reproduces the original model's behavior — any other layout would
silently compute a wrong result against real GGUF files.

**Load-time gate:** only `general.architecture == "llama"` loads (`LlamaModel.LoadFromGguf`,
`LlamaModel.cs:80-84`); anything else throws `NotSupportedException`. Config comes from GGUF
`llama.*` metadata into `ModelConfig` (`src/InferenceEngine.Core/ModelConfig.cs`).

**Numerics:** pure F32. Weights are dequantized to `float[]` at load time; only F32/F16 GGUF
tensor types are supported — quantized types (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/…) parse as metadata/shape
but throw `NotSupportedException` on read. That gap belongs to `planned-format-loading.md`, not
resolved here.

**No custom kernels:** every numeric op routes through `System.Numerics.Tensors.TensorPrimitives`
(`Dot`, `SumOfSquares`, `Multiply`, `SoftMax`, `Sigmoid`, `MultiplyAdd`, `Add`) or a thin
hand-written loop composing those calls. No `unsafe`, no hand-rolled SIMD.

## Decision Drivers

- **Understanding payoff over throughput** ([260901](260901-project-challenges-and-how-to-address-them.md)
  rates transformer/attention wiring **High** payoff to build) — read everything below against
  this, not against raw speed.
- **Build what real Llama-family checkpoints ship** (SmolLM2, and dotLLM's own Llama/Mistral/Phi/
  Qwen — `docs/investigation/dotllm-architecture-trace.md`): GQA, pre-norm, SwiGLU, not a
  hand-picked variant.
- **GGUF's conversion pins the RoPE layout** (see Context) — not a style choice.
- **Reuse a math library over hand-rolling low-level math** (AGENTS.md,
  [260901](260901-project-challenges-and-how-to-address-them.md)) — only the dataflow
  (RMSNorm/RoPE/GQA/SwiGLU composition) is worth hand-writing; `Ops` is exactly that split.
- **Smallest surface first** — single-token `Forward` captures the attention/RoPE/SwiGLU math
  with minimal surface; batching is deferred (see Considered Options, Option C).
- **`unsafe` off by default** (AGENTS.md) — no raw-pointer arithmetic anywhere in `Ops`,
  `GqaAttention`, or `LlamaModel`.

## Considered Options

- **A — Hand-roll custom SIMD/intrinsics kernels instead of `TensorPrimitives`.**
- **B — Special-case separate MHA / GQA / MQA code paths instead of one generalized formula.**
- **C — Batch multi-token prefill now, instead of single-token-only `Forward`.**

## Decision Outcome

Chosen: **generalized GQA + RoPE attention, pre-norm layers, SwiGLU FFN, built entirely on
`TensorPrimitives`, single-token `Forward` only.**

- **Option A rejected** — `TensorPrimitives` already covers every primitive this pass needs;
  nothing is left to hand-write below the dataflow level.
- **Option B rejected** — the ratio formula covers MHA/MQA as edge cases at zero extra code.
- **Option C deferred** — real perf upside, but lower understanding payoff than getting the
  attention math right first (see Consequences for the follow-up).

### Consequences

Legend: 🟢 upside · 🟡 accepted trade-off · 🔴 downside.

- 🟢 **MHA and MQA checkpoints need zero new code.** They hit the same `Attend` call as GQA, just
  at `groupSize == 1` or `NumKvHeads == 1`. Genuinely untested beyond the 3:1 ratio SmolLM2
  exercises — treat MHA/MQA as reachable, not verified, until a real checkpoint runs through them.
- 🟡 **Three scope boundaries, each with its own future ADR:** F32-only (quantized dequant →
  `planned-format-loading.md`), single-token-only (batched prefill → this ADR's deferred Option C,
  no round scheduled), Llama-only (`LlamaModel.LoadFromGguf`'s hard `NotSupportedException` gate —
  Mistral/Qwen/Phi need their own ADR when there's a real model to load).
- 🔴 **The attention loop is tested directly; the full per-layer composition is not.**
  `GqaAttention.Attend` is checked against an independent reference transcription across multiple
  context lengths and block-boundary cases
  (`tests/InferenceEngine.Models.Tests/Llama/GqaAttentionTests.cs`), and `Ops`'s primitives
  (RmsNorm, MatVec, Rope, SwiGlu, Softmax) against hand-computed values (`OpsTests.cs`). The full
  wiring in `LlamaModel.Forward` — RMSNorm → Q/K/V → RoPE → `Attend` → output+residual → RMSNorm →
  SwiGLU → residual, composed exactly as `Forward` does it — has no dedicated test; it's checked
  only end-to-end via the golden FNV-1a logit-hash regression (`GoldenLogitBaselineTests.cs`). If
  that composition is restructured again, the golden hash is the only tripwire.
- 🟢 **Bit-identity against the golden hash is the verification discipline for this file.** Two
  restructuring rounds already relied on it (KV-cache round's per-KV-head write/RoPE split, loop
  reorder) because float addition isn't associative — a "harmless" reorder can silently change
  output. Any future change to `LlamaModel.Forward`, `Ops`, or `GqaAttention` must re-run and
  re-diff that hash, not just eyeball the diff.

## Pros and Cons of the Options

### Option A — Hand-rolled SIMD/intrinsics kernels

- 🟢 Theoretical throughput ceiling — full control over vectorization and cache tiling.
- 🔴 High effort for a Medium-at-best payoff
  ([260901](260901-project-challenges-and-how-to-address-them.md)) — the project cares about the
  transformer's dataflow, not squeezing AVX-512.
- 🔴 Needs `unsafe` or hand-verified intrinsics, against AGENTS.md's default-off policy.

### Option B — Separate MHA / GQA / MQA code paths

- 🟢 Each variant reads simply in isolation.
- 🔴 Three code paths for one formula — `groupSize == 1` and `NumKvHeads == 1` are edge cases, not
  distinct algorithms.
- 🔴 More surface to keep bit-identical under future restructuring, for no behavioral gain.

### Option C — Batch multi-token prefill now

- 🟢 Real, measured perf upside — prefill dominates wall-clock on long prompts (KV-cache
  benchmark: ~29s for a ~1,900-token prompt).
- 🟢 Proven technique — dotLLM already does this
  (`docs/investigation/dotllm-architecture-trace.md`: fused single-token decode, batched GEMM
  prefill).
- 🔴 Lower understanding payoff than the attention math itself — a batched matmul path teaches
  throughput engineering, not GQA/RoPE/SwiGLU.
- 🔴 New surface (multi-position `Forward`, batched KV-cache writes) ahead of a concrete caller
  need — AGENTS.md's no-speculative-abstractions concern; named as a seam, not built.

## More Information

- [260811-solution-and-project-layout.md](260811-solution-and-project-layout.md) — the `Models`/
  `Core` ownership split this ADR operates inside.
- [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md) —
  the build-vs-reuse effort/payoff table that put this row on the "Build" side.
- [260914-kv-cache.md](260914-kv-cache.md) — the per-KV-head write/RoPE split, loop-reorder
  precedent, and bit-identity verification pattern (see Consequences).
- [260917-logits-processing.md](260917-logits-processing.md) — what sits downstream of this ADR's
  LM-head logits.
- [planned-format-loading.md](planned-format-loading.md) — the quantization gap (F32/F16-only),
  named but not resolved here.
- [docs/investigation/dotllm-architecture-trace.md](../investigation/dotllm-architecture-trace.md) —
  dotLLM's attention/forward-pass design (fused decode, batched prefill GEMM), the reference
  Option C is weighed against.

## Decision Log

| Date       | Change              | By                 |
|------------|---------------------|--------------------|
| 2026-09-01 | Placeholder created | Aleksei Kolesnikov |
| 2026-09-17 | Documented the already-implemented attention/transformer forward pass; Status set directly to accepted (code has been running and tested since before this ADR was written) | Aleksei Kolesnikov |
| 2026-09-17 | Editorial compression pass: cut reasoning repeated across Context/Decision Drivers/Decision Outcome/Pros-and-Cons per the tightened adr-author standard; no decision change, no factual content removed | Aleksei Kolesnikov |
