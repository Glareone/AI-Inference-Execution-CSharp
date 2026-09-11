# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Project scaffolding: README with project goal and inspiration (dotLLM by Konrad Kokosa),
  AGENTS.md / CLAUDE.md ground rules, ADR template (`architecture/`), and Claude Code
  subagents (`adr-writer-reviewer`, `csharp-dotnet`).
- [Solution-layout ADR](architecture/260811-solution-and-project-layout.md): solution/project layout.
- [Project-challenges ADR](architecture/260901-project-challenges-and-how-to-address-them.md):
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

### Changed

- ADR file naming convention switched from sequential `NNNN-title.md` to `YYMMDD-<slug>.md`
  (chronological by date prefix). Renamed `0001-solution-and-project-layout.md` →
  `260811-solution-and-project-layout.md` and `0000-template.md` → `template.md`; updated
  references in README, AGENTS.md, the `adr-writer-reviewer`/`csharp-dotnet` agents, and the
  investigation notes.
- Polished the solution-layout ADR: added a project-reference diagram (Mermaid) and a
  "which project is responsible for what" table; corrected dotLLM's project count (10 → ~17);
  kept it focused on structure, with the engineering challenges moved out (see below).
