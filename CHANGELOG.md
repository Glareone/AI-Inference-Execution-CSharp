# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Project scaffolding: README with project goal and inspiration (dotLLM by Konrad Kokosa),
  AGENTS.md / CLAUDE.md ground rules, ADR template (`architecture/`), and Claude Code
  subagents (`adr-writer-reviewer`, `csharp-dotnet`).
- [Solution-layout ADR](docs/architecture/260811-solution-and-project-layout.md): solution/project layout.
- [Project-challenges ADR](docs/architecture/260901-project-challenges-and-how-to-address-them.md):
  the problems each layer solves, the build-vs-reuse strategy per challenge, the fetch-once/mmap
  model-weight dependency, a runtime-call diagram, and how other engines are structured.
- Seven placeholder ADRs (`architecture/planned-*.md`, `Status: planned`) — one per component
  challenge from the project-challenges ADR: format loading, tokenization, attention & transformer,
  KV-cache, sampling pipeline, HuggingFace acquisition, performance baseline. Each is renamed to
  `YYMMDD-<slug>.md` and filled in when its round comes.
- `src/` scaffolding per the solution-layout ADR — `InferenceEngine.{Core,Models,Tokenizers,Engine}` class
  libraries and a runnable `InferenceEngine.Cli` console app (net10.0, no functionality yet).
- Rider run/debug configuration for `InferenceEngine.Cli` (`launchSettings.json` +
  `.idea/runConfigurations/`).
- `adr-writer-reviewer` and `csharp-dotnet` agents now consult context7 for current library
  docs before evaluating/using a package, instead of relying on training-data memory.
- Three investigation agents: `researcher`, `huggingface-explorer`, `code-reader` — for
  structured research into inference internals before implementation.
- `investigation/` directory with research documentation:
  - `overview.md` — investigation goals, references, topic map
  - `dotllm-architecture-trace.md` — full code-path trace of dotLLM's architecture
    (GGUF loading → tokenization → generation loop → sampling)
  - `huggingface-ecosystem.md` — GGUF model providers, Hub API, download options, tokenizer embedding
  - `gguf-format-research.md` — binary format spec, quantization types (Q4_0 through Q6_K),
    K-quant super-block structure, memory-mapping, .NET implementation notes
  - `inference-engine-project-layouts.md` — how other engines are structured (dotLLM, Jlama,
    llama3.java, LLamaSharp, ONNX Runtime GenAI, LM-Kit.NET), with sources — reference for the
    solution-layout comparison
- `experiments/` directory with benchmark data:
  - `reference-measurements-dotllm.md` — dotLLM v0.1.0-preview.3 baseline: 53.9 tok/s decode
    on SmolLM2-135M Q4_K_M, GGUF metadata dump, tensor quantization analysis
- README.md expanded with investigation topics, references, project structure, agent descriptions.
- **Bare-minimum end-to-end inference POC**: `InferenceEngine.Cli` now loads a real GGUF model
  and streams generated tokens, running SmolLM2-135M-Instruct (f16) locally.
  - `InferenceEngine.Core`: `ModelConfig`, `IModel`, `ITokenizer`, `IKvCache`, `TokenizerData` contracts.
  - `InferenceEngine.Models`: hand-rolled GGUF v3 reader (`Gguf/`) — no maintained GGUF-parsing
    library exists on NuGet, so format parsing had to be built rather than reused (see the
    format-loading ADR follow-up); Llama forward pass (`Llama/`) — RMSNorm, GQA attention with
    interleaved-pair RoPE, SwiGLU FFN, tied LM head — built on `System.Numerics.Tensors`
    (`Math/Ops.cs`), no custom kernels, no `unsafe`. Supports F32/F16 tensors only; quantized
    types parse correctly as metadata but throw on dequantization.
  - `InferenceEngine.Tokenizers`: hand-rolled byte-level BPE (`GgufBpeTokenizer`) built from a
    model's embedded vocab/merges, since GGUF doesn't embed the pre-tokenizer's splitting regex.
    Implements the `smollm` pre-tokenizer variant (matched against llama.cpp's `unicode.cpp`).
  - `InferenceEngine.Engine`: `SimpleKvCache` (contiguous per-layer arrays), a composable
    `Sampler` (temperature → top-k → top-p, with a greedy short-circuit), and the
    `InferenceSession` facade (ChatML wrapping, prefill/decode loop, streaming generation).
  - `InferenceEngine.Cli`: `--model`, `--prompt`, `--max-tokens`, `--temperature`, `--top-k`,
    `--top-p`, `--seed`, `--raw`, `--stats`, `--debug-tokenize`, `--debug-logits`.
  - Verified: extracted `ModelConfig` matches the recorded dotLLM baseline exactly; tokenizer
    round-trips exactly; top-1 next-token prediction after a test prompt is semantically
    correct; 64-token greedy generation is coherent and factually correct, at ~38 tok/s decode
    (f16, vs. dotLLM's 53.9 tok/s on Q4_K_M — expected, given ~2.6x the memory traffic per token
    and no fused/quantized kernels).
  - Explicitly out of scope for this POC (see the plan): HuggingFace download (`--model` takes
    a local path only), batched prefill (single-token loop), the model's Jinja2 chat template
    (hard-coded ChatML string instead), multi-threading, and HTTP serving.
- `Directory.Build.props`: `TreatWarningsAsErrors` enabled.
- `.claude/settings.json`: checked-in project permissions policy (read-only allowlist for the
  model cache and dotLLM reference install; narrow, pre-approved `dotnet`/`brew` inspection
  commands; `WebFetch` allowed for github.com, huggingface.co, raw.githubusercontent.com).
- **Automated test suite**: one xUnit v3 (Microsoft Testing Platform) test project per `src/`
  project — `tests/InferenceEngine.{Core,Models,Tokenizers,Engine,Cli}.Tests` — 51 tests, all
  passing, each documenting the business scenario it protects (see each project's README.md).
  `global.json` now selects the .NET 10 SDK's native MTP `dotnet test` runner. Highlights:
  a synthetic-GGUF-file builder exercises the hand-rolled reader (metadata, F32/F16 tensors,
  alignment, unsupported quant types) without needing the real 258 MB model; `Ops.cs`'s math
  (RmsNorm/MatVec/RoPE/Softmax/SwiGLU) is checked against hand-computed values; the BPE
  tokenizer's merge-rank ordering and digit pre-tokenization are verified directly; the sampler
  (greedy/top-k/top-p, seeded reproducibility) and `ChatMlTemplate`'s turn ordering are covered.
  Coverage on these hand-rolled files: `Ops.cs` 100%, `GgufFile.cs` 86%, `GgufBpeTokenizer.cs`
  92–100%, `SamplingPipeline.cs` 98.4%, `ChatMlTemplate.cs` 100%.
  Found and documented (not silently fixed) a real gap: `TokenizerData`'s `string[]` fields give
  it reference-based, not value-based, record equality.
  One minimal production seam: `CliOptions.Load` gained an optional `loadDotEnv` parameter
  (default `true`, so production behavior is unchanged) so tests can exercise the flags/env-var
  precedence logic without touching the filesystem or a stray real `.env`.
  New `test-writer-runner` Claude Code agent (`.claude/agents/`) owns writing and running these
  going forward — one test project per `src/` project, business-case-documented, always run
  before being reported done.
- `docs/scenarios/`: one Gherkin (`.feature`) file per `src/` project, documenting in plain
  Given/When/Then form the business scenarios the automated tests and manual CLI verification
  cover — plain specification files, not wired to a BDD execution framework.

### Fixed

CodeRabbit review findings on the POC PR:

- `CliOptions.Load` no longer throws `IndexOutOfRangeException`/`FormatException` for a flag
  missing its value or a malformed numeric flag/environment-variable value — both now raise a
  clear `ArgumentException` naming the offending flag or variable.
- `Program.cs` now catches `ArgumentException` around model loading and generation, not just
  around CLI option parsing, so an `InferenceSession.Generate` validation failure (see below)
  reports a clean one-line error and exit code 1 instead of an unhandled-exception stack trace.
- `InferenceSession.Generate` now validates `MaxNewTokens >= 0`, a non-empty encoded prompt, and
  the requested sequence length against `ModelConfig.MaxSeqLen` — and validates *eagerly*, before
  returning, rather than only once the caller starts enumerating (it was refactored from a single
  iterator method into a plain validating wrapper around a private iterator, since code before a
  `yield` in an iterator method doesn't run until the first `MoveNext`). Previously, a negative
  `--max-tokens` or `--raw` with an empty prompt could throw a confusing exception deep inside
  `SimpleKvCache`, or silently sample from an empty logits span.
- Streamed generation could show a UTF-8 replacement character when a single multi-byte
  character's bytes were split across two generated tokens, because each token was decoded to
  text independently. `InferenceSession` now feeds each token's raw bytes (`ITokenizer` gained
  `GetTokenBytes`) through a new `IncrementalUtf8Decoder`, which buffers an incomplete character
  across calls the way `System.Text.Decoder` is designed to.

Second round of CodeRabbit review findings:

- `GgufFile`: a file-declared count (metadata KV count, tensor count, tensor dimension count,
  string length, array length) exceeding the file's own size now throws `InvalidDataException`
  before any allocation sized to it — a small malformed or truncated file could otherwise trigger
  an excessive allocation. Separately, reading a metadata string now uses `Stream.ReadExactly`
  instead of `BinaryReader.ReadBytes`, which silently returns a shorter-than-requested array on a
  truncated file instead of throwing.
- `GgufTensorDescriptor.ElementCount`: the dimension-count multiplication is now `checked`, so
  dimensions that would overflow `long` throw `OverflowException` instead of silently wrapping to
  a smaller, incorrect (but plausible-looking) tensor size.
- `GgufBpeTokenizer`: a merge-list entry that isn't a space-separated pair now throws a clear
  `InvalidDataException` naming the entry, instead of `IndexOutOfRangeException` from `Split`.
- `InferenceSession.Load` now validates that the model's vocab size (`llama.vocab_size`) matches
  the tokenizer's vocabulary length (`tokenizer.ggml.tokens`) — both are read independently from
  the same GGUF file, and if they disagree, a sampled id could be out of range for the tokenizer.
  Caught at load time with a clear cause instead of an obscure exception mid-generation.
- `LlamaModel.Forward` now rejects a `position` outside `[0, MaxSeqLen)` — its attention scratch
  buffers are sized to `MaxSeqLen` — and `InferenceSession.PrefillTopLogits` (the `--debug-logits`
  path, which had no such guard) now rejects a prompt longer than `MaxSeqLen` before allocating
  the KV-cache, matching the guard already added to `Generate`.
- `TokenizerData.UnknownTokenId`: removed — read from GGUF metadata but never consumed anywhere.
- `Ops.Rope`: the per-pair rotation angle (`cos`/`sin`) depended only on position and pair index,
  not on which head was being rotated, but was recomputed (`MathF.Pow`/`Cos`/`Sin`) once per head
  per pair. Now computed once per pair and reused across all heads — fewer redundant transcendental
  calls on the decode hot path, same result.
- Removed several XML doc comments across the sampling types that restated what the code already
  says (pure "what", no non-obvious "why"), per this project's comment policy.

10 new tests across Models/Tokenizers/Engine (71 total, all passing): implausible GGUF counts,
truncated-string reads, tensor dimension overflow, a malformed merge entry, the vocab-size
consistency check, and the `PrefillTopLogits` sequence-length guard. Re-verified end-to-end
against the real model after the `Rope` refactor and the new `GgufFile` bounds checks — output
and load time are unchanged. Clean build (0 warnings/errors, Debug + Release, wiped `bin`/`obj`).

### Changed

- Moved `architecture/`, `investigation/`, and `scenarios/` under a new `docs/` folder
  (`docs/architecture/`, `docs/investigation/`, `docs/scenarios/`), via `git mv` to preserve
  history. `experiments/` stays at the repo root (not part of this move). Updated every
  cross-reference: README, AGENTS.md, CLAUDE.md, CHANGELOG, `.coderabbit.yaml`'s path
  filters/instructions, the `adr-writer-reviewer`/`csharp-dotnet` agent definitions, and the
  one relative link (`planned-performance-baseline.md` → `experiments/`) whose depth changed
  because `architecture/` moved but `experiments/` didn't.
- ADR file naming convention switched from sequential `NNNN-title.md` to `YYMMDD-<slug>.md`
  (chronological by date prefix). Renamed `0001-solution-and-project-layout.md` →
  `260811-solution-and-project-layout.md` and `0000-template.md` → `template.md`; updated
  references in README, AGENTS.md, the `adr-writer-reviewer`/`csharp-dotnet` agents, and the
  investigation notes.
- Polished the solution-layout ADR: added a project-reference diagram (Mermaid) and a
  "which project is responsible for what" table; corrected dotLLM's project count (10 → ~17);
  kept it focused on structure, with the engineering challenges moved out (see below).
