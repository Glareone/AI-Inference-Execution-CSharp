# Paged + head-major KV cache

- Status: proposed
- Date: 2026-09-14

## Where this problem is isolated

Contained in **`InferenceEngine.Engine`** (storage) and **`InferenceEngine.Core`** (the contract):

- **Lives here:** the key/value store, its block allocation policy, and the per-step update
  logic. `Engine` owns `KvBlockPool` and `PagedKvCache`; `Core` owns the `IKvCache` interface both
  `Engine` and `Models.Forward` code against.
- **Exposed as:** `IKvCache` (defined in `Core`), which `Models.Forward` receives and reads/writes
  through — the model still doesn't own cache memory or its allocation policy, only the per-token
  math. This ADR changes `IKvCache`'s member shape (per-KV-head slot/tile accessors instead of
  per-layer row accessors, see Decision Outcome) but keeps the ownership split from
  [260811](260811-solution-and-project-layout.md) unchanged.
- **Contained change:** moving from `SimpleKvCache` to `PagedKvCache` changes `Engine` (the
  storage) and the call sites in `LlamaModel.Forward` that address it (accessor names and the
  attention loop's iteration order), not the transformer math itself — RMSNorm, RoPE, SwiGLU, and
  the matmuls are untouched.

## Context and Problem Statement

`SimpleKvCache` (`src/InferenceEngine.Engine/SimpleKvCache.cs`) is an Era-1 (2017) design: one
`float[length * kvDim]` per layer, exact-fit allocated to `promptLength + maxNewTokens` at
`Generate` start, never grown, never reset, never shared between sessions. That was the correct
first step under [260901](260901-project-challenges-and-how-to-address-them.md)'s "start simple,
understand before optimizing" stance, and its placeholder ADR
(`docs/architecture/planned-kv-cache.md`) was left with every section `_TBD._` — this ADR replaces
that placeholder and supersedes its stated stance (see Decision Log).

Two things prompt revisiting it now:

1. **Which library should we use?** This is the direct question the ADR exists to answer. Every
   .NET inference option in AGENTS.md's reference list was checked
   ([kv-cache-research.md](../investigation/kv-cache-research.md)): LLamaSharp and ONNX Runtime
   GenAI both manage the KV cache inside native code reached only through P/Invoke — opaque, not
   inspectable, not something we can read or modify. TorchSharp exposes raw libtorch tensors and no
   cache concept at all. dotLLM, our primary reference architecture and the one pure-C# engine in
   the comparison, hand-builds its own (block-allocated, ref-counted, copy-on-write, Q8_0/Q4_0
   quantized, hash-based prefix cache) — proof that a hand-built cache is the normal approach even
   outside C/C++, not a workaround for a gap in the ecosystem. **There is no library to adopt.**
   Building one, deliberately mirroring vLLM's PagedAttention and llama.cpp's block design, is the
   only option that leaves the KV cache visible for the understanding this project exists to
   produce.
2. **Bandwidth.** Decode attention is memory-bandwidth-bound: every step reads the entire cache
   history for a single new token, so arithmetic intensity is low and the loop's DRAM traffic
   pattern dominates its cost. The current attention loop
   (`LlamaModel.cs:122-144`) has two bandwidth problems independent of paging:
   - It iterates query heads outermost and re-reads each KV-head row once per query head sharing
     it (GQA group size 3 on SmolLM2-135M), tripling K/V DRAM traffic versus reading each KV-head
     row once and reusing it in L1 across the 3 query heads that share it.
   - Its layout is NHD-equivalent (`[position, kv_head, head_dim]`): a query head's 64-float
     (256 B) dot product is a strided read out of a 192-float (768 B) per-position stripe, wasting
     2/3 of every stripe it touches and defeating sequential prefetch.

   [kv-cache-research.md](../investigation/kv-cache-research.md) quantifies both: at context length
   2048, across all 30 layers, per decode token, today's loop moves ~283 MB of K/V traffic;
   reordering the loop alone (no layout change) cuts that to ~94 MB — a 3x reduction in KV
   traffic, and a ~23% cut in *total* per-token memory traffic (weights + KV) from ~820 MB to
   ~631 MB. **This is the single largest change in this ADR, and it has nothing to do with
   paging.**

The research also surfaces the load-bearing finding for how this work is sequenced: of the three
changes bundled together in a naive "paged KV cache" story — loop reordering, head-major (HND)
layout, and block paging — **paging itself contributes zero measurable throughput for a single
sequence.** Ascending block ids under a bump allocator place physical bytes exactly where a
contiguous cache already would; the only runtime difference is one `int[]` block-table
indirection per lookup. Paging's payoff is structural (lazy growth, and the addressing scheme
that prefix caching, copy-on-write, and speculative decoding are built on), not a bandwidth win —
so this ADR keeps loop-reordering, layout, and paging as separable, separately-measured changes
rather than one commit that would misattribute a layout/paging label to a loop-order effect.

**Scope.** Single sequence — no scheduler, no multi-sequence locking, no shared pool across
concurrent requests (this engine doesn't have concurrent requests). Prefix caching, ref-counting,
copy-on-write, KV quantization, and eviction policies are designed for — the block-table structure
this ADR introduces is exactly the seam vLLM and llama.cpp hang those features off — but not built
in this pass; see Consequences and the seams below for what's left open and why.

## Decision Drivers

- **Understanding payoff over throughput.** Per [260901](260901-project-challenges-and-how-to-address-them.md)'s
  effort-vs-understanding rule, KV-cache design is rated **High** payoff to build — the paged
  design is what makes that payoff real; a benchmark-driven "just fix the loop" stop would leave
  the paging/addressing lesson (the actual subject of eras 2-4 in the research) unbuilt.
  Throughput matters, but it is not the driver that picks the option here — see the Decision
  Outcome for what does and does not follow from it.
- **Decode is memory-bandwidth-bound.** Every design choice (block size, layout, loop order) has
  to be justified by what it does to DRAM/cache-line traffic, not by operation count — the
  quantitative estimate in [kv-cache-research.md](../investigation/kv-cache-research.md) exists so
  this ADR can point at a number instead of asserting one.
- **`unsafe` off, `TensorPrimitives`-only**, per AGENTS.md's `unsafe` policy — no raw-pointer
  arithmetic to hand-optimize the block indexing; the block/head/slot index formula stays plain
  managed array/span math.
- **Single sequence, no premature multi-tenancy.** Building a scheduler, locking, or a
  multi-sequence shared pool now would be speculative abstraction for a feature this engine
  doesn't have a caller for yet (AGENTS.md's "no speculative abstractions").
- **`unsafe`-free bit-for-bit reproducibility.** Every change in this pass is a restructuring of
  *where* the same float operations happen, not what they compute — verification has to prove
  that (see More Information), which constrains the design to changes that don't reorder floating-
  point accumulation (float addition isn't associative; the softmax/weighted-sum accumulation
  order must stay identical before and after).

## Considered Options

- **A — Reuse ONNX Runtime GenAI or LLamaSharp's built-in KV management.** Both manage the cache
  internally in native code reached through P/Invoke.
- **B — Keep `SimpleKvCache`'s contiguous layout; fix only the loop order (and optionally the
  layout).** Cheapest change, captures most of the measured bandwidth win.
- **C — Block-paged, head-major (HND) layout, single sequence.** Loop reorder + HND layout +
  block-table addressing, mirroring vLLM/llama.cpp's block design without a multi-sequence
  scheduler.
- **D — Option C plus prefix caching, ref-counted/copy-on-write sharing, and KV quantization.**

## Decision Outcome

Chosen option: **C — block-paged, head-major (HND) layout, single sequence**, landed as three
separate, separately-measured commits (loop reorder, then HND layout, then block paging) rather
than one bundled change, so the ADR and its benchmark can state which change bought what instead
of crediting the whole thing to "paging."

This is not the option the bandwidth numbers alone would pick — Option B captures essentially all
of the measured throughput win (the loop reorder) for less new surface. Option C is chosen because
the **understanding driver**, not the bandwidth driver, is the one doing the work here: the
project's stated goal is to understand how a modern KV cache is built, and Option B teaches
nothing about block addressing, leaves the cache exact-fit and non-growable, and closes off the
seams (prefix caching, copy-on-write, speculative decoding) that motivate paging in every real
engine this project is benchmarked against. Option D is deferred, not rejected: this pass builds
the block-table structure those features attach to, without building the features themselves,
because the *seams* (naming a specific method or field each future feature would touch) are cheap
to design for now, while the *features* (hashing, ref-counting, dequantizing reads) are separate,
independently-reviewable units of work with their own correctness surface. Option A is rejected
outright — it contradicts the project's premise (AGENTS.md, [260901](260901-project-challenges-and-how-to-address-them.md))
of building the parts of inference worth understanding rather than wrapping them.

### Design

- **Block size:** 32 tokens (power of two, so `logicalBlock = position >> 5` and
  `slot = position & 31` are shifts/masks, not divisions).
- **Layout per physical block:** all layers, K and V in separate `float[]` arrays, each shaped
  `[layer][kvHead][slot][headDim]` (head-major / HND). Index formula:
  `index = layer*layerStride + kvHead*headStride + slot*headDim + d`.
- **`IKvCache` surface change:** the current per-layer row accessors `Key(layer, position)` /
  `Value(layer, position)` return a `kvDim`-wide row spanning *all* KV heads at one position —
  incompatible with a head-major layout, where a given head's data across positions is contiguous
  but a given position's data across heads is not. They're replaced with per-KV-head accessors:
  `KeySlot`/`ValueSlot` (write, one head's `headDim` floats at one position) and
  `KeyBlockForHead`/`ValueBlockForHead` (read, a contiguous `[count, headDim]` tile for one KV head
  across a run of positions). This is also what makes the loop-reorder change possible: the
  attention loop can now iterate KV heads outermost and pull one contiguous tile per head instead
  of walking per-position rows once per query head.
- **Allocation:** `Reserve(position)` explicitly assigns a physical block on demand, called from
  the generation loop rather than hidden inside `KeySlot`/`ValueSlot` — keeping it a visible,
  separately-callable hook is what lets copy-on-write intercept it later without changing the
  write path's call sites.
- **New cache lifecycle methods:** `Reset()` and `Rollback(toPosition)`. Neither has a production
  caller yet in this pass — they're a deliberate seam for multi-turn session reuse and speculative
  decoding's speculative-token rollback, documented here rather than built silently later so a
  future reader can see they were anticipated, not discovered as an afterthought.
- **New types:** `KvBlockPool` (physical block storage + free list; arrays are retained across
  `Free()`, not reallocated, so `Reset()` is allocation-free) and `PagedKvCache` (the block table —
  logical block index to physical block id — plus the `IKvCache` accessors). `SimpleKvCache` is
  deleted once `PagedKvCache` lands; they are not kept side by side.
- **Sizing, on this development machine (Apple M4 Pro; 128 B cache line, 16 KiB page, 64 KiB L1D,
  4 MiB L2 — measured via `sysctl`, not assumed):** `headDim = 64` floats = 256 B = exactly 2 cache
  lines, so one KV-head tile at 32-token block size is 8 KiB — long enough to amortize per-tile
  overhead, short enough (≤2 pages) to stay TLB-friendly. One physical block (K+V, all 30 layers of
  SmolLM2-135M) is ~1.4 MiB. A 2048-token context is 64 blocks, ~90 MiB — the same total bytes as
  today's exact-fit `SimpleKvCache` sizing, but allocated lazily as position advances instead of up
  front at `Generate` start.

### Consequences

Legend: 🟢 upside · 🟡 accepted trade-off · 🔴 downside.

- 🟢 The loop-reorder change (outer loop over KV heads, inner over the query heads sharing each)
  is the one change in this pass with a real, quantified bandwidth win: ~283 MB → ~94 MB of K/V
  traffic per decode token at context 2048 (3x), a ~23% cut in total per-token memory traffic. It
  requires no new types and is not a paging benefit.
- 🟡 **Paging buys a single-sequence engine no measurable speedup today.** With ascending block
  ids under a bump allocator, physical bytes land in the same order a contiguous cache would put
  them; the only per-lookup cost added is one `int[]` block-table indirection. Anyone reading a
  future benchmark that shows this pass faster than `SimpleKvCache` should attribute that entirely
  to the loop reorder (and, more modestly, the HND layout), not to paging — this is the
  distinction the three-commit sequencing exists to make legible.
- 🟡 **Benchmarking trap.** A memory microbenchmark simulating a single layer will show the layout
  change doing almost nothing (a single layer's K+V at context 2048 is ~3 MiB, nearly fitting
  inside this CPU's 4 MiB L2); only a benchmark that cycles all 30 layers reproduces the real
  cold-cache DRAM traffic this ADR's numbers describe.
- 🟡 **Memory overhead at short contexts.** Block granularity (32 tokens) means a session that
  generates only a handful of tokens still reserves whole blocks — worse than `SimpleKvCache`'s
  exact-fit `promptLength + maxNewTokens` sizing for short runs, better for long or unpredictable-
  length runs where exact-fit would have to be pessimistic up front. At the context lengths this
  project actually exercises (short demo runs), this is a real, accepted cost, not a rounding
  error.
- 🟢 **Lazy growth.** `SimpleKvCache` commits `promptLength + maxNewTokens` worth of memory before
  generation starts and can never exceed it. `PagedKvCache` allocates one block at a time as
  `position` advances, so memory tracks actual usage rather than a worst-case bound fixed at
  session start.
- 🟢 **Seams for future work, named rather than built:** ref-counting (an `int[] _refCount` in
  `KvBlockPool`, decremented on `Free`, reclaimed at zero), copy-on-write (a check in
  `PagedKvCache.Reserve` — if a block's refcount > 1, copy before write), prefix-cache hashing (a
  `Dictionary<hash, blockId>` in the pool plus a `SealBlock` hook when a block fills — the cache
  doesn't see token ids today, so this also needs a token-id parameter threaded in), sliding-window
  / attention-sink eviction (replacing the block table's linear walk with a caller-supplied
  enumeration of attended blocks), and KV quantization (the `ReadOnlySpan<float>` tile-accessor
  return type would need to become a `Span<float> destination`-style dequantizing read — exactly
  two call sites in the attention loop would change). None of these are built in this pass; each
  is deferred to its own future ADR when it has a real caller, per AGENTS.md's "no speculative
  abstractions."
- 🔴 **New surface to maintain and review.** `KvBlockPool` and `PagedKvCache` are two new types
  plus a changed `IKvCache` contract (row accessors → per-head slot/tile accessors), which is
  strictly more code and more index-arithmetic than `SimpleKvCache`'s two flat arrays. This is the
  direct cost of the understanding payoff the Decision Outcome cites, and is accepted on those
  terms, not because it is free.
- 🔴 **`Reset()`/`Rollback()` are unverified by production use.** They exist as a documented seam
  with unit-test coverage (block-pool contract, non-overlap, capacity exhaustion — see More
  Information) but no caller in the generation loop yet; their real API shape may need revision
  once multi-turn reuse or speculative decoding actually calls them.

## Pros and Cons of the Options

### Option A — Reuse ONNX Runtime GenAI / LLamaSharp's internal KV management

- 🟢 Zero implementation and maintenance cost; both are mature, widely used, and almost certainly
  faster than anything this project would hand-build.
- 🔴 The cache lives inside native code reached only through P/Invoke — not inspectable, not
  modifiable, not a source of the understanding this project exists to produce. Directly
  contradicts AGENTS.md and [260901](260901-project-challenges-and-how-to-address-them.md)'s
  build-vs-reuse line, which puts KV-cache design on the "build" side specifically because its
  understanding payoff is rated High.

### Option B — Keep contiguous storage, fix only the loop (and optionally the layout)

- 🟢 Captures the large majority of the measured bandwidth win (the loop reorder, ~3x KV traffic
  reduction) for the smallest possible change — no new types, no `IKvCache` contract change beyond
  what the layout switch alone would need.
- 🟢 Lowest risk to the bit-identical-logits verification requirement, since less code changes.
- 🔴 Teaches nothing about block addressing, the actual subject of PagedAttention and llama.cpp's
  paged design — the two reference architectures this project set out to understand.
- 🔴 Keeps exact-fit sizing (`promptLength + maxNewTokens`, fixed at session start) — no lazy
  growth, and closes off prefix caching / copy-on-write / speculative-decoding rollback, none of
  which have anywhere to attach without block-addressed storage.
- 🔴 As the endpoint of this work (not a step), it would leave the placeholder ADR's original
  "paged is a later step, if at all" stance essentially unrevisited despite writing this ADR to
  revisit it.

### Option C — Block-paged, head-major (HND) layout, single sequence (chosen)

- 🟢 Mirrors vLLM's PagedAttention and llama.cpp's block design closely enough to be a legible
  reference against real systems, not an invented scheme.
- 🟢 Delivers the loop-reorder bandwidth win (Option B's benefit) *and* the block-addressing
  structure that motivates paging in every real engine surveyed, in the same pass.
- 🟢 Names concrete, minimal-diff seams for prefix caching, copy-on-write, quantization, and
  eviction — future ADRs can build on named hooks instead of re-deriving where they'd attach.
- 🟡 Paging itself contributes no measured speedup for a single sequence (see Consequences) — its
  payoff here is structural and educational, not a throughput number this pass can point to.
- 🔴 More new surface (two types, a changed `IKvCache` contract) than Option B for a throughput
  result Option B already captures most of.

### Option D — Option C plus prefix caching, ref-counting/copy-on-write, and KV quantization

- 🟢 Closest to dotLLM's actual feature set and to what a production-grade engine would ship.
- 🔴 Each addition is a separate, non-trivial correctness surface (hash collisions and refcount
  bugs for prefix caching; dequantization accuracy and an extra read-path branch for quantization)
  that would arrive with no caller yet to exercise it — speculative scope by AGENTS.md's own
  definition. Deferred to future ADRs once each has a concrete reason to exist, not rejected.

## More Information

- [kv-cache-research.md](../investigation/kv-cache-research.md) — primary research backing this
  ADR: the five eras of KV-cache design, why no .NET library is a fit, the NHD-vs-HND layout
  analysis, the measured hardware facts this design's block-size argument depends on, and the
  quantitative loop-reorder-vs-paging traffic estimate.
- [260811-solution-and-project-layout.md](260811-solution-and-project-layout.md) — the `Engine`/
  `Core` ownership split this ADR operates inside.
- [260901-project-challenges-and-how-to-address-them.md](260901-project-challenges-and-how-to-address-them.md) —
  the build-vs-reuse decision that put KV-cache design on the "build" side.
- Verification approach (bit-identical logits via FNV-1a hash of raw float bit patterns across the
  loop-reorder / layout / paging commits, plus `KvBlockPool`/`PagedKvCache` unit tests for the
  layout contract, non-overlap, `Reserve`/`Rollback`/`Reset` behavior, and capacity exhaustion, plus
  an attention-loop equivalence test against a reference transcription of the current
  implementation) is tracked with the implementation work, not restated in full here — tolerance-
  based comparison is deliberately not used, since float addition isn't associative and the point
  of the verification is to prove accumulation order was preserved exactly across the restructure.
- [Modular — The Five Eras of KVCache](https://www.modular.com/blog/the-five-eras-of-kvcache)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM paper)](https://arxiv.org/pdf/2309.06180)
- [llama.cpp — Paged KV cache and scheduler, Phase 1 design discussion #21961](https://github.com/ggml-org/llama.cpp/discussions/21961)

## Decision Log

| Date       | Change              | By                 |
|------------|---------------------|--------------------|
| 2026-09-01 | Placeholder created | Aleksei Kolesnikov |
| 2026-09-14 | ADR written: chosen block-paged + head-major design (Option C), superseding the placeholder's "paged is a later step, if at all" stance with a deliberate revision; sequenced as loop-reorder / layout / paging as separate measured commits | Aleksei Kolesnikov |
