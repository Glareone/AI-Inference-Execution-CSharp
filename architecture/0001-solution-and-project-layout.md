# 0001. Solution and project layout

- Status: proposed
- Date: 2026-08-11

## Context and Problem Statement

We need a `src/` layout to start writing code against. dotLLM (our reference architecture)
uses ten separate projects (Core, Models, Tokenizers, Cpu, Cuda, Engine, Server,
HuggingFace, Diagnostics, Cli) shipped as individual NuGet packages. Our goal here is
smaller: understand how the pieces of an inference engine fit together, reusing libraries
instead of hand-rolling kernels (see AGENTS.md). What project layout fits that scope without
either being a single undifferentiated blob or copying dotLLM's full package surface before
we've written a line of engine code?

## Decision Drivers

- Reuse libraries for model loading, tokenization, and math — we are not writing custom
  SIMD/CUDA kernels, so there's no driver for `Cpu`/`Cuda` backend projects yet.
- Keep the boundaries that actually matter for *understanding* the architecture visible as
  project boundaries (model loading vs. tokenization vs. orchestration/sampling/KV-cache),
  since that separation is the main learning goal.
- Avoid scaffolding layers with no near-term content (a `Server` project before a working
  generation loop exists is premature) — matches our own "no speculative abstractions" rule.
- Something runnable end-to-end early: a CLI that loads a model and generates tokens is more
  useful for learning than a set of empty libraries.

## Considered Options

- Option A: Single console project, everything inline.
- Option B: Layered class libraries mirroring a scoped-down subset of dotLLM (Core, Models,
  Tokenizers, Engine, Cli), no Cpu/Cuda/Server/HuggingFace/Diagnostics yet.
- Option C: Full dotLLM-style layering from day one (all ten projects).

## Decision Outcome

Chosen option: "Option B", because it mirrors the reference architecture closely enough to
stay legible against dotLLM's design (the whole point of using it as a map), but doesn't
build out layers we have no content for yet. Cpu/Cuda kernel projects contradict the
library-reuse goal; Server, HuggingFace, and Diagnostics are follow-on concerns once a basic
generate-loop exists.

Projects, each a class library targeting `net10.0` unless noted, referencing only the layers
below them:

- `InferenceEngine.Core` — shared abstractions: tensor/config types, `IModel`,
  `ITokenizer`, etc. No dependencies on other project layers.
- `InferenceEngine.Models` — model loading (format parsing, e.g. GGUF), via a library rather
  than a hand-rolled parser (library choice is its own future ADR). References `Core`.
- `InferenceEngine.Tokenizers` — tokenization, via a library where a suitable one exists
  (library choice is its own future ADR). References `Core`.
- `InferenceEngine.Engine` — orchestration: the generation loop, sampling, KV-cache.
  References `Core`, `Models`, `Tokenizers`.
- `InferenceEngine.Cli` — console entry point wiring the above together end-to-end.
  References `Engine`. Executable (`OutputType=Exe`).

Solution file `InferenceEngine.sln` at the repo root, with solution folders matching the
project names above, `src/<ProjectName>/<ProjectName>.csproj` on disk. A root
`Directory.Build.props` sets shared settings (nullable enable, implicit usings enable,
current C# language version) so individual `.csproj` files stay minimal.

### Consequences

- Good, because the project boundaries map directly onto the concepts we're trying to
  understand (loading vs. tokenizing vs. orchestrating), which is the point of the exercise.
- Good, because `Cli` gives an early, runnable end-to-end target instead of a pile of
  libraries with nothing wiring them together.
- Bad, because if we later *do* want a custom CPU/CUDA kernel experiment or a server, we'll
  add those projects then — this ADR doesn't pre-approve that; it'll need its own decision
  when it has real content.
- Bad, because five projects for essentially no code yet is more ceremony than Option A. We
  accept this because the separation is deliberate and we'd have to introduce it very soon
  anyway once real content lands in each layer.

## Pros and Cons of the Options

### Option A: Single console project

- Good, because zero ceremony, fastest to start writing code.
- Bad, because it actively works against the learning goal — no visible boundary between
  "this is model loading" and "this is sampling logic" as the code grows.

### Option B: Scoped-down layered libraries (chosen)

- Good, because it mirrors dotLLM's architecture (our reference) at a scope matching our
  actual goals.
- Bad, because it's more upfront structure than we have code to fill, today.

### Option C: Full dotLLM-style layering (10 projects)

- Good, because it's a complete match to the reference architecture, nothing to add later.
- Bad, because `Cpu`/`Cuda` custom kernels directly contradict the "reuse libraries, don't
  hand-roll math kernels" goal from AGENTS.md; `Server`/`HuggingFace`/`Diagnostics` have no
  content until the core loop works. Building empty shells for all of these is exactly the
  kind of speculative scaffolding our working-style rules say to avoid.

## More Information

Reference: [dotLLM](https://github.com/kkokosa/dotLLM) project structure (see README.md).
Library choices for `Models` (GGUF loading) and `Tokenizers` are deferred to their own ADRs.

## Decision Log

| Date       | Change            | By              |
|------------|-------------------|-----------------|
| 2026-08-11 | Initial proposal  | Claude (agent)  |
