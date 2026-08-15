# Investigation Status

## What's in place

- **Solution layout** (ADR-0001): five projects targeting .NET 10 — Core, Models, Tokenizers,
  Engine, Cli — mirroring a scoped-down dotLLM structure. Builds and runs.
- **Shared build props**: `Directory.Build.props` sets nullable, implicit usings, and language
  version so individual `.csproj` files stay minimal.
- **CLI entry point**: `InferenceEngine.Cli` wires the layers together; currently a stub
  (`Program.cs`) that compiles and runs but does no real work yet.
- **Architecture decisions**: MADR template and ADR-0001 (project layout) in `architecture/`.
- **Agentic workflow**: AGENTS.md ground rules, CLAUDE.md with Claude Code subagents
  (ADR writer/reviewer, C#/.NET implementation agent).

## What's next

1. **GGUF model loading** — pick a library (or write a minimal parser) to read GGUF files in
   `InferenceEngine.Models`. Needs its own ADR for the library choice.
2. **Tokenization** — pick a tokenizer library for `InferenceEngine.Tokenizers`. Needs its own
   ADR.
3. **Core abstractions** — define `IModel`, `ITokenizer`, tensor/config types in
   `InferenceEngine.Core` once the library choices inform what the interfaces should look like.
4. **Generation loop** — implement sampling and KV-cache in `InferenceEngine.Engine`, wired
   through the CLI so we can load a model and generate tokens end to end.
5. **Serving** (later) — an OpenAI-compatible HTTP endpoint, once the core loop works.
