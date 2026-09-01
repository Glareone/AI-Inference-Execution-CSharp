# Investigation Status

## What's in place

- **Solution layout** (solution-layout ADR): five projects targeting .NET 10 — Core, Models, Tokenizers,
  Engine, Cli — mirroring a scoped-down dotLLM structure. Builds and runs.
- **Shared build props**: `Directory.Build.props` sets nullable, implicit usings, and language
  version so individual `.csproj` files stay minimal.
- **CLI entry point**: `InferenceEngine.Cli` wires the layers together; currently a stub
  (`Program.cs`) that compiles and runs but does no real work yet.
- **Architecture decisions**: MADR template and the solution-layout ADR (project layout) in `architecture/`.
- **Agentic workflow**: AGENTS.md ground rules, CLAUDE.md with Claude Code subagents
  (ADR writer/reviewer, C#/.NET implementation agent).
- **Investigation agents**: `researcher`, `huggingface-explorer`, `code-reader` in
  `.claude/agents/` — for structured research before implementation.
- **dotLLM installed**: v0.1.0-preview.3 via `dotnet tool install -g DotLLM.Cli --prerelease`.
  Test model downloaded: SmolLM2-135M-Instruct Q4_K_M (101 MB).
- **Baseline measurements**: dotLLM running at 53.9 tok/s decode on our machine.
  See `experiments/reference-measurements-dotllm.md`.
- **dotLLM architecture traced**: full code path from CLI → GGUF load → tokenize → generate.
  See `investigation/dotllm-architecture-trace.md`.
- **GGUF file analyzed**: parsed real metadata and tensor descriptors from our test model,
  documented mixed-precision quantization strategy of Q4_K_M.

## What's next

Seven placeholder ADRs now exist (`Status: planned`), one per component challenge from the
[project-challenges ADR](../architecture/260901-project-challenges-and-how-to-address-them.md).
Each carries its owning project and build-vs-reuse stance; the MADR body is filled in — and the
file renamed to `YYMMDD-<slug>.md` — when its round comes.

1. **[Model format loading](../architecture/planned-format-loading.md)** — placeholder created;
   to fill: GGUF vs. SafeTensors and the parsing library. (Format research done — see below.)
2. **[Tokenization](../architecture/planned-tokenization.md)** — placeholder created; to fill:
   tokenizer library choice and pre-tokenizer sourcing. (Ecosystem research done — see below.)
3. **[Attention & transformer](../architecture/planned-attention-and-transformer.md)** —
   placeholder created; to fill: forward-pass approach on a math library.
4. **[KV-cache](../architecture/planned-kv-cache.md)** — placeholder created; to fill: simple
   vs. paged design.
5. **[Sampling pipeline](../architecture/planned-sampling-pipeline.md)** — placeholder created;
   to fill: composable sampler chain design.
6. **[HuggingFace model acquisition](../architecture/planned-huggingface-acquisition.md)** —
   placeholder created; to fill: download library, cache layout, resume/verify behavior.
7. **[Performance baseline](../architecture/planned-performance-baseline.md)** — placeholder
   created; to fill: tokens/s targets, managed vs. `unsafe`, SIMD.
8. **Core abstractions** — define `IModel`, `ITokenizer`, tensor/config types in
   `InferenceEngine.Core` once ADR decisions inform the interfaces.
9. **Generation loop** — implement sampling and KV-cache in `InferenceEngine.Engine`,
   wired through CLI for end-to-end token generation.
10. **Serving** (later) — OpenAI-compatible HTTP endpoint.

## Investigation Documents

| Document | Status |
|----------|--------|
| `investigation/overview.md` | done |
| `investigation/dotllm-architecture-trace.md` | done |
| `investigation/gguf-format-research.md` | done |
| `investigation/huggingface-ecosystem.md` | done |
| `investigation/inference-engine-project-layouts.md` | done |
| `experiments/reference-measurements-dotllm.md` | done |
