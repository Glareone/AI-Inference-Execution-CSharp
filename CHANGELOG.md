# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Project scaffolding: README with project goal and inspiration (dotLLM by Konrad Kokosa),
  AGENTS.md / CLAUDE.md ground rules, ADR template (`architecture/`), and Claude Code
  subagents (`adr-writer-reviewer`, `csharp-dotnet`).
- [ADR-0001](architecture/0001-solution-and-project-layout.md): solution/project layout.
- `src/` scaffolding per ADR-0001 — `InferenceEngine.{Core,Models,Tokenizers,Engine}` class
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
- `experiments/` directory with benchmark data:
  - `reference-measurements-dotllm.md` — dotLLM v0.1.0-preview.3 baseline: 53.9 tok/s decode
    on SmolLM2-135M Q4_K_M, GGUF metadata dump, tensor quantization analysis
- README.md expanded with investigation topics, references, project structure, agent descriptions
