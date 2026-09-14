# KV-Cache Rewrite Benchmark: Before/After

Before/after measurement for the [paged, head-major KV-cache ADR](../docs/architecture/260914-kv-cache.md),
comparing the original `SimpleKvCache` (contiguous, NHD-equivalent, query-head-outermost attention
loop) against the rewritten `PagedKvCache` (block-paged, head-major/HND, KV-head-outermost loop with
3x query-head reuse). Both builds are the same repository at two commits on `KV-adjustments`:

- **Before**: `9229230` (golden-baseline test committed, production code still the original).
- **After**: `f30783a` (all six rewrite steps landed — loop reorder, RoPE table hoist, head-major
  layout, block paging, session wiring).

## Environment

- **Machine**: Apple M4 Pro (macOS Darwin 24.6.0), same machine as
  [reference-measurements-dotllm.md](reference-measurements-dotllm.md).
- **Measured via `sysctl`, not assumed** (the block-size argument in
  [kv-cache-research.md](../docs/investigation/kv-cache-research.md) depends on these):
  `hw.cachelinesize=128`, `hw.pagesize=16384`, `hw.l1dcachesize=65536` (64 KiB), `hw.l2cachesize=4194304`
  (4 MiB), `hw.memsize=51539607552` (~48 GB).
- **Runtime**: .NET 10, Release build (`dotnet build -c Release`).
- **Date**: 2026-09-14.

## Model: SmolLM2-135M-Instruct F16

- **File**: `SmolLM2-135M-Instruct-f16.gguf` (270,885,952 bytes) — F16, not the Q4_K_M dotLLM uses;
  this engine has no dequantization kernels yet (see `docs/investigation/gguf-format-research.md`).
- **Architecture**: Llama, 30 layers, 576 hidden, 9 query heads / 3 KV heads (GQA 3:1), head_dim 64.
- **Vocab**: 49,152 tokens. **Context**: 8,192 tokens.

## Method and the trap it avoids

`kv-cache-research.md` flags that at the dotLLM baseline's 108-token context, KV traffic is only
~2.7% of per-token memory traffic (weights dominate at ~537 MB/token, F32) — measuring there cannot
show the effect this rewrite targets. So two prompts, at very different context lengths:

- **Short**: `"What is machine learning? Explain briefly."` (38 tokens after ChatML templating).
- **Long**: [`kv-benchmark-long-prompt.txt`](kv-benchmark-long-prompt.txt), 1,906 tokens — built by
  repeating a technical paragraph, chosen to land near the 2048-token figure used in the ADR's
  traffic estimate and to span 60 logical blocks (blockSize 32).

Both runs use `--stats --temperature 0 --max-tokens 32`. **Caveat on what `--stats` actually times**:
`Program.cs` wraps the *entire* `session.Generate(...)` call — prefill and decode together — and
divides by only the 32 *generated* tokens, not the full token count processed. This engine's prefill
has no batched path (`docs/investigation/dotllm-architecture-trace.md`; `LlamaModel.cs`'s own header
comment), so prefill is a sequential loop of single-token `Forward` calls, each attending over its own
growing history — structurally identical to a decode step at that position. For the short prompt this
metric approximates decode-only throughput (prefill is 38 negligible steps). For the long prompt it is
dominated by prefill's ~1,906 steps, weighted toward the later, more expensive, large-context ones —
not a "decode at steady-state ctx=1906" number, but a legitimate end-to-end stress test of the same
per-call attention cost the ADR's arithmetic is about, summed across every context length from 0 to
~1,906 within one run.

Each configuration run 3x; all three individual runs reported (no cherry-picking).

## Results

### Short prompt (ctx grows 38→70 over the run)

| Build | Run 1 | Run 2 | Run 3 | Mean tok/s |
|---|---|---|---|---|
| **Before** (`SimpleKvCache`) | 30.4 tok/s | 29.2 tok/s | 29.8 tok/s | 29.8 |
| **After** (`PagedKvCache`) | 31.0 tok/s | 31.0 tok/s | 31.6 tok/s | 31.2 |

**+4.7% throughput.** Consistent with the ADR's prediction: at this context length KV traffic is a
small fraction of total per-token traffic, so the loop-reorder/layout change is expected to be
noise-level — which is what this shows.

### Long prompt (prefill 1,906 tokens + decode 32, ctx grows 1,906→1,938)

| Build | Run 1 | Run 2 | Run 3 | Mean wall time |
|---|---|---|---|---|
| **Before** (`SimpleKvCache`) | 38.03s | 38.53s | 39.05s | 38.54s |
| **After** (`PagedKvCache`) | 28.82s | 29.34s | 29.48s | 29.21s |

**−24.2% wall-clock time** (1.32x speedup on this combined prefill+decode workload). The ADR's
independent estimate — reordering the loop alone cuts K/V DRAM traffic 3x, and total per-token
traffic (weights + KV) ~23%, at context 2048 — predicted a ~23% cut; the measured 24.2% wall-clock
reduction over a run that sweeps context 0→1,938 lines up closely with that estimate. Given the
`--stats` caveat above, this is not a clean isolation of "the loop-reorder effect" from "the layout
effect" from "the paging effect" (all three landed between these two commits) — see
[kv-cache-research.md](../docs/investigation/kv-cache-research.md) and the ADR for the reasoning that
the loop reorder is expected to account for nearly all of it, with paging contributing ~0 and layout a
smaller, unmeasured-in-isolation share.

## Memory

| | Before (`SimpleKvCache`, exact-fit) | After (`PagedKvCache`, block-paged, 32-token blocks) |
|---|---|---|
| Short run (70 max positions) | 70 × 45 KiB ≈ 3.08 MiB | ⌈70/32⌉ = 3 blocks × 1.40625 MiB ≈ 4.22 MiB (**+37%**) |
| Long run (1,938 max positions) | 1,938 × 45 KiB ≈ 85.2 MiB | ⌈1,938/32⌉ = 61 blocks × 1.40625 MiB ≈ 85.8 MiB (**+0.7%**) |

Matches the ADR's Consequences: block granularity costs real, visible overhead at short contexts (a
70-token session reserves a full extra block's worth of headroom) and is negligible once the run is
long enough that block rounding is a small fraction of the total — the accepted trade-off for lazy
growth instead of `SimpleKvCache`'s always-exact-but-never-growable sizing.

## Observations

1. **The loop reorder is an expected contributor; this comparison does not isolate its share of
   the speedup.** The before/after commits span all three changes (loop reorder, head-major
   layout, paging), so the ~24% at long context and ~0% at short context can't be attributed to
   any one of them from this measurement alone. What the shape *is* consistent with: a
   bandwidth-bound change targeting KV traffic (which only matters at longer contexts), and the
   ADR's independent estimate that the loop reorder accounts for nearly all of it while paging is
   architecturally incapable of contributing a DRAM-traffic win for a single sequence (see the
   ADR's Consequences) — but confirming that split would need the three changes measured
   separately, which this benchmark doesn't do.
2. **The `--stats` metric is not a clean decode-only number**, and that's fine here — see Method
   above. A future benchmark that wants to isolate decode-at-fixed-context would need a CLI flag to
   seed a cache with a prefix without timing the prefill, which doesn't exist yet and wasn't worth
   building for one benchmark (`AGENTS.md`'s "no speculative abstractions").
3. **This is one machine, one model, one quantization (F16).** The 4 MiB L2 / 128 B cache line facts
   recorded above are specific to this Apple M4 Pro; the benchmarking-trap warning in the ADR (a
   single-layer microbenchmark would hide this effect inside L2 residency) is a reason this
   measurement runs the real 30-layer model end-to-end rather than a synthetic per-layer loop.
4. **Golden-hash correctness held throughout** (`GoldenLogitBaselineTests`, hashes
   `0xB07058A15AA1650A` / `0xECC1669FE5C7DB32`) — this speedup came from restructuring *where* the
   same floating-point operations happen, not from changing what they compute.

## Reproduction

```bash
MODEL=~/.cache/inference-engine/models/SmolLM2-135M-Instruct-f16.gguf
SHORT="What is machine learning? Explain briefly."
LONG="$(cat experiments/kv-benchmark-long-prompt.txt)"

dotnet build -c Release
dotnet run --project src/InferenceEngine.Cli -c Release -- \
  --model "$MODEL" --prompt "$SHORT" --max-tokens 32 --temperature 0 --stats
dotnet run --project src/InferenceEngine.Cli -c Release -- \
  --model "$MODEL" --prompt "$LONG" --max-tokens 32 --temperature 0 --stats --raw
```

To reproduce the "before" numbers, `git checkout 9229230` (detached) before building.
