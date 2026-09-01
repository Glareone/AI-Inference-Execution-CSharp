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

1. **GGUF model loading ADR** — write ADR based on investigation findings.
   Research agents still running (GGUF format spec, HuggingFace ecosystem).
2. **Tokenization ADR** — pick a tokenizer library. Research pending.
3. **Attention & transformer ADR** — document understanding, decide approach.
4. **KV-cache ADR** — simple vs paged, implementation decision.
5. **Sampling pipeline ADR** — composable sampler chain design.
6. **HuggingFace model acquisition ADR** — download strategy.
7. **Performance baseline ADR** — targets, managed vs unsafe, SIMD.
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
| `experiments/reference-measurements-dotllm.md` | done |
