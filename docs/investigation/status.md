# Investigation Status

## What's in place

- **Solution layout** (solution-layout ADR): five projects targeting .NET 10 — Core, Models, Tokenizers,
  Engine, Cli — mirroring a scoped-down dotLLM structure. Builds and runs.
- **Shared build props**: `Directory.Build.props` sets nullable, implicit usings, and language
  version so individual `.csproj` files stay minimal.
- **CLI entry point**: `InferenceEngine.Cli` wires the layers together end-to-end — loads a GGUF
  model, tokenizes, generates, and streams output. See the [README](../../README.md#status) for
  current status and test coverage.
- **Architecture decisions**: MADR template and the solution-layout ADR (project layout) in `docs/architecture/`.
- **Agentic workflow**: AGENTS.md ground rules, CLAUDE.md with Claude Code subagents
  (ADR writer/reviewer, C#/.NET implementation agent).
- **Investigation agents**: `researcher`, `huggingface-explorer`, `code-reader` in
  `.claude/agents/` — for structured research before implementation.
- **dotLLM installed**: v0.1.0-preview.3 via `dotnet tool install -g DotLLM.Cli --prerelease`.
  Test model downloaded: SmolLM2-135M-Instruct Q4_K_M (101 MB).
- **Baseline measurements**: dotLLM running at 53.9 tok/s decode on our machine.
  See `experiments/reference-measurements-dotllm.md`.
- **dotLLM architecture traced**: full code path from CLI → GGUF load → tokenize → generate.
  See `docs/investigation/dotllm-architecture-trace.md`.
- **GGUF file analyzed**: parsed real metadata and tensor descriptors from our test model,
  documented mixed-precision quantization strategy of Q4_K_M.

## What's next

Seven placeholder ADRs were created (`Status: planned`), one per component challenge from the
[project-challenges ADR](../architecture/260901-project-challenges-and-how-to-address-them.md).
Each carries its owning project and build-vs-reuse stance; the MADR body is filled in — and the
file renamed to `YYMMDD-<slug>.md` — when its round comes. Two (KV-cache, Sampling pipeline) have
been written up; five remain as placeholders below, several already implemented in code ahead of
their ADR.

1. **[Model format loading](../architecture/planned-format-loading.md)** — placeholder created;
   to fill: GGUF vs. SafeTensors and the parsing library. (Format research done — see below.)
2. **[Tokenization](../architecture/planned-tokenization.md)** — placeholder created; to fill:
   tokenizer library choice and pre-tokenizer sourcing. (Ecosystem research done — see below.)
3. **[Attention & transformer](../architecture/planned-attention-and-transformer.md)** —
   placeholder created; to fill: forward-pass approach on a math library.
4. **[KV-cache](../architecture/260914-kv-cache.md)** — done: block-paged, head-major (HND)
   layout, single sequence. See [kv-cache-research.md](kv-cache-research.md).
5. **[Logits processing and sampling pipeline](../architecture/260917-logits-processing.md)** —
   done: four-way tokenizer/logits/logits-processor/sampler split, `ILogitsProcessor` +
   `BannedSequenceLogitsProcessor` design, chosen as an internal split inside `Engine` rather than
   a new project.
6. **[HuggingFace model acquisition](../architecture/planned-huggingface-acquisition.md)** —
   placeholder created; to fill: download library, cache layout, resume/verify behavior.
7. **[Performance baseline](../architecture/planned-performance-baseline.md)** — placeholder
   created; to fill: tokens/s targets, managed vs. `unsafe`, SIMD.
8. **Core abstractions** — done: `IModel`, `ITokenizer`, `IKvCache`, `ModelConfig` in
   `InferenceEngine.Core`.
9. **Generation loop** — done: sampling and the paged KV-cache are implemented in
   `InferenceEngine.Engine`, wired through the CLI for end-to-end token generation. See the
   [README](../../README.md#status).
10. **Serving** (later) — OpenAI-compatible HTTP endpoint. Not started.

## Logits processing — pending implementation (Phase B)

ADR done ([260917](../architecture/260917-logits-processing.md)). Code not started. Docs merge to
main first; implementation is the next round. Do steps 1-3 before 4-7 — they're the dependency.

1. `src/InferenceEngine.Engine/Sampling/ILogitsProcessor.cs` — new. `internal interface`, one
   method: `void Apply(Span<float> logits, ReadOnlySpan<int> generatedTokenIds)`.
2. `src/InferenceEngine.Engine/Sampling/BannedSequenceLogitsProcessor.cs` — new. Implements
   `ILogitsProcessor`. Length-1 banned sequence: always mask to `-inf`. Longer sequence: mask only
   when `generatedTokenIds`'s tail matches its prefix (HuggingFace `NoBadWordsLogitsProcessor`
   algorithm). **Never masks EOS** (read from `ITokenizer.EosTokenId` at construction) — same
   guard HF uses, and the reason is not cosmetic: see step 3's all-masked fallback. Factory
   `FromWords(ITokenizer, IReadOnlyList<string>)` tokenizes each banned word once, at
   construction — not per generated token.
3. `src/InferenceEngine.Engine/Sampling/SamplingPipeline.cs` — edit. Constructor takes
   `IReadOnlyList<ILogitsProcessor>` **and `int eosTokenId`** (needed for the fallback below —
   `BannedSequenceLogitsProcessor` knowing EOS internally isn't enough, since the fallback lives
   in `Sample`, not in any one processor). `Sample` gains `ReadOnlySpan<int> generatedTokenIds`.
   Run every processor unconditionally, before the greedy/stochastic branch splits. This is the
   actual fix: today the greedy path (`Temperature<=0`, the default) skips `_steps` entirely — a
   ban must not skip.
   - **Buffer**: `Apply` needs `Span<float>`. Today greedy never copies (argmaxes the read-only
     input directly); stochastic always copies to `_scratch` before running `ISamplerStep`s,
     processor or not — that copy is unchanged either way. The only *new* copy is on the greedy
     side: when ≥1 processor is registered, greedy also copies to `_scratch` first, runs
     processors, then argmaxes over `_scratch`. Zero processors → greedy keeps its existing
     zero-copy path, unchanged.
   - **All-masked fallback**: if every logit is `-inf` after processors run, return `eosTokenId`
     directly (the new constructor parameter). Today greedy's `best = 0` scan and the stochastic
     path's `Exp(-inf - -inf) = NaN` (NaN comparisons are always `false`, so it falls through to
     `working.Length - 1`) both return an arbitrary index with no such guard — verified against
     the actual code, this is a real bug, not a hypothetical. Step 2's EOS-never-masked rule is
     the primary guarantee; this is the defensive backstop.
4. `src/InferenceEngine.Engine/Config/GenerationOptions.cs` — edit. Add
   `IReadOnlyList<string>? BannedWords = null` at the end — the `= null` is required (a positional
   record can't have a required parameter after an optional one; all six existing ones already
   default), and with it the existing 6-argument call at `Program.cs:53` keeps compiling unchanged.
5. `src/InferenceEngine.Engine/InferenceSession.cs` — edit. `GenerateCore` builds the processor
   list via `BannedSequenceLogitsProcessor.FromWords(_tokenizer, options.BannedWords ?? [])`,
   constructs `SamplingPipeline` passing `_tokenizer.EosTokenId` (step 3's new constructor
   parameter), tracks generated-id history, passes it into `Sample`. `PrefillTopLogits` stays
   untouched — raw diagnostic view, deliberately unprocessed.
6. `src/InferenceEngine.Cli/Config/CliOptions.cs` + `.env.example` — edit. New flag/env var,
   comma-separated: `--ban-words "EPAM,EPAM ,E"` / `INFERENCE_BAN_WORDS`.
7. `src/InferenceEngine.Cli/Program.cs` — edit. Thread the new option through. While here: switch
   the `GenerationOptions` construction from positional to named arguments — the positional call
   breaks silently if a field lands out of order.
8. Tests (`test-writer-runner` agent):
   - `BannedSequenceLogitsProcessor`: single-token always-banned; multi-token sequence-prefix
     match; multiple independent sequences; a sequence longer than history doesn't crash; banning
     the EOS token id is a no-op (still not masked).
   - **Critical regression test**: banned tokens excluded on the default greedy path
     (`Temperature=0`). This is the one thing the whole redesign exists to make true.
   - **All-masked fallback, both paths**: construct a pipeline where every non-EOS token is
     banned (or force-mask via a test double) and assert `Sample` returns EOS in greedy mode
     (`Temperature=0`) and in stochastic mode (`Temperature>0`) — this is the exact bug CodeRabbit
     caught by reading the code, not by running it; prove it's fixed by running it.
   - `CliOptions` parsing test for the new flag.
   - `GoldenLogitBaselineTests` stays green — banning defaults to off, hashes shouldn't move.
9. `CHANGELOG.md` + affected test project `README.md`s +
   `docs/scenarios/InferenceEngine.Engine.feature` — add scenarios for the banned-sequence
   processor once it's implemented and tested (scenarios mirror real coverage here, not written
   ahead of it). Reword the existing sampling scenarios if needed to name the *sampler* phase
   specifically, now that a distinct *processor* phase exists alongside it.

Verify: `dotnet build` clean (0 warnings — `TreatWarningsAsErrors=true`), `dotnet test` green.
Manual smoke test: `--ban-words "EPAM"` on a prompt likely to invoke it — confirm it never
appears. Golden-logit hash unaffected (banning is opt-in, off by default).

## Further phases (not started, no ADR yet)

- **Quantized-tensor dequantization** (Q4_K/Q5_K/Q6_K/Q8_0) — the actual blocker to running
  HuggingFace GGUF models above ~500M params. Math already researched
  (`gguf-format-research.md`), not implemented.
- **Tokenization ADR** — code (hand-rolled BPE) contradicts the placeholder's stated stance
  ("reuse a library"); no ADR records the reversal. `planned-tokenization.md` still all `_TBD_`.
- **Repetition penalty, forced-token bias, grammar/schema-constrained decoding** — named seams in
  [260917](../architecture/260917-logits-processing.md), no ADR of their own yet.
- **HuggingFace model acquisition, performance baseline** — still `planned` placeholders,
  genuinely not started.

## Investigation Documents

| Document | Status |
|----------|--------|
| `docs/investigation/overview.md` | done |
| `docs/investigation/dotllm-architecture-trace.md` | done |
| `docs/investigation/gguf-format-research.md` | done |
| `docs/investigation/huggingface-ecosystem.md` | done |
| `docs/investigation/inference-engine-project-layouts.md` | done |
| `docs/investigation/kv-cache-research.md` | done |
| `experiments/reference-measurements-dotllm.md` | done |
