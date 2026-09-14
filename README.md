# AI-Inference-Execution-CSharp

An investigation into how LLM inference engines work, implemented in C#/.NET 10.

## What This Project Is

A hands-on learning project: understand every piece of an LLM inference engine — model
loading, tokenization, attention mechanisms, KV-cache, sampling, serving — by building a
working version in C#. Not a production engine. Not a wrapper. A guided investigation where
every component is understood deeply enough to explain to anyone.

## What We're Investigating

1. **GGUF model format** — how quantized weights are stored, parsed, and memory-mapped
2. **Tokenization** — BPE and SentencePiece algorithms, vocabulary, merge rules
3. **Transformer architecture** — self-attention (MHA, GQA, MQA), RoPE, RMSNorm, SwiGLU FFN
4. **KV-cache** — why it exists, simple vs paged designs, memory cost
5. **Logits processing & sampling** — temperature, top-k, top-p, repetition penalties
6. **HuggingFace integration** — model discovery, downloading, format ecosystem
7. **Performance** — tokens/second, memory bandwidth vs compute, SIMD in .NET, GC pressure

Each topic is documented as an ADR (Architecture Decision Record) in `docs/architecture/`,
capturing both what we learned and what we decided for our implementation.

## References & Inspiration

### Primary Reference

- **[dotLLM](https://github.com/kkokosa/dotLLM)** by Konrad Kokosa — a ground-up LLM
  inference engine in pure C#/.NET 10. Not a wrapper around llama.cpp. Reaches 66–88%
  of llama.cpp decode throughput on CPU. Supports Llama, Mistral, Phi, Qwen, DeepSeek.
  Our architecture is inspired by its layered design (Core → Models/Tokenizers → Engine).
  - Blog post: [Introducing dotLLM](https://kokosa.dev/blog/2026/dotllm/)
  - Foundational posts: [Logits, logprobs, and temperature](https://kokosa.dev/blog/2026/temperature/),
    [Visualizing logprobs](https://kokosa.dev/blog/2026/logprobs/)

### Other C# Projects Studied

- **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** — mature P/Invoke wrapper
  around llama.cpp. High-level API patterns (executors, context management). Not a learning
  tool for internals, but useful for API design reference.
- **[ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)** — Microsoft's
  ONNX-based inference with generate loop. Different format (ONNX, not GGUF).
- **[Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers)** —
  standalone .NET tokenizer library (BPE, SentencePiece, Tiktoken). Candidate for our
  tokenizer implementation.

### Background Reading

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the C/C++ inference engine
  that defined the GGUF format and local LLM inference
- [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Implementing BPE from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) — Sebastian Raschka
- [HuggingFace Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [KV-Cache Explained](https://www.emergentmind.com/topics/kv-cache)
- Konrad Kokosa, *Pro .NET Memory Management* (2nd ed.) — relevant for understanding
  GC-free inference and native memory patterns in .NET

## Project Structure

```
src/
  InferenceEngine.Core/         Shared abstractions (IModel, ITokenizer, etc.)
  InferenceEngine.Models/       GGUF loading, model config
  InferenceEngine.Tokenizers/   Tokenization (BPE, SentencePiece)
  InferenceEngine.Engine/       Generation loop, KV-cache, sampling
  InferenceEngine.Cli/          Console entry point
tests/
  InferenceEngine.{Core,Models,Tokenizers,Engine,Cli}.Tests/
                                 xUnit test project per src/ project (see each README.md)
docs/
  scenarios/                   Gherkin (.feature) specs, one per src/ project — the business
                                scenarios the tests and manual CLI verification cover
  architecture/                 ADRs in MADR format
  investigation/                Research notes, architecture traces
experiments/                    Benchmark data, reference measurements
```

See [the solution-layout ADR](docs/architecture/260811-solution-and-project-layout.md) for the
rationale behind this layout.

## Test Model

**SmolLM2-135M-Instruct**, Llama architecture, 30 layers, 576 hidden dim, GQA 3:1 (9 query / 3 KV
heads), 49K vocab. Small enough to iterate on, large enough to produce coherent text.

Two GGUF variants are in play, for different reasons:

- **F16** (`bartowski/SmolLM2-135M-Instruct-GGUF`, ~270 MB) — what *this* engine actually loads
  and runs. Our GGUF reader only materializes F32/F16 tensor data (see Status below), so this is
  the variant `--model` points at.
- **Q4_K_M** (101 MB) — installed via the dotLLM CLI and used only as the reference baseline
  (53.9 tok/s decode) in [experiments/reference-measurements-dotllm.md](experiments/reference-measurements-dotllm.md).
  Our engine cannot load this file yet — quantized tensors parse as metadata but throw on read.

## Claude Code Agents

This project uses AI-assisted development with structured agents:

| Agent | Purpose |
|-------|---------|
| `adr-author` | Write/review Architecture Decision Records (MADR format) |
| `csharp-dotnet` | C#/.NET implementation and code review |
| `test-writer-runner` | Write/update/run tests, one project per `src/` project, business-case documented |
| `researcher` | Web research on inference internals (algorithms, specs, papers) |
| `huggingface-explorer` | HuggingFace ecosystem (model formats, APIs, downloads) |
| `code-reader` | Read/analyze external repos (dotLLM, LLamaSharp) |

All agents use [context7](https://github.com/upstash/context7) for current library
documentation instead of relying on training data.

## Status

**End-to-end generation works.** `dotnet run --project src/InferenceEngine.Cli -- --model
<path.gguf> --prompt "..."` loads a GGUF file, tokenizes, runs the full transformer forward pass,
samples, and streams generated text — on real weights, not a stub. Last verified: 2026-09-14.

### What works today

- **Model loading**: hand-rolled GGUF v3 reader (`InferenceEngine.Models/Gguf/`), memory-mapped,
  metadata + tensor descriptors + F32/F16 tensor data.
- **Tokenization**: hand-rolled byte-level BPE (`InferenceEngine.Tokenizers/`) sourced from the
  GGUF file's own embedded vocab/merges — no external tokenizer file needed.
- **Transformer forward pass**: embeddings → RMSNorm → grouped-query attention with RoPE →
  SwiGLU FFN → LM head (`InferenceEngine.Models/Llama/`), built entirely on
  `System.Numerics.Tensors.TensorPrimitives` — no hand-written SIMD/intrinsics, no `unsafe`.
- **KV-cache**: block-paged, head-major (HND) layout — 32-token blocks, lazy allocation, the
  attention loop reordered to cut K/V memory traffic ~3x over a naive per-query-head walk. See the
  [KV-cache ADR](docs/architecture/260914-kv-cache.md) and its
  [benchmark](experiments/kv-layout-benchmark.md) (+4.7% at short context, −24% wall-clock at a
  1,900-token prompt vs. the original contiguous cache).
- **Sampling**: composable temperature / top-k / top-p pipeline (`InferenceEngine.Engine/Sampling/`).
- **CLI**: streamed token output, `--stats`, `--debug-tokenize`, `--debug-logits`, `.env` config.

On this dev machine (Apple M4 Pro), SmolLM2-135M-Instruct-F16 generates at **~30 tok/s decode**
at short context (see [experiments/kv-layout-benchmark.md](experiments/kv-layout-benchmark.md) for
longer-context numbers).

### Test coverage

**97 test cases, 0 failing**, one xUnit v3 project per `src/` project:

| Project | Test cases (incl. `[Theory]` rows) |
|---|---|
| `InferenceEngine.Core.Tests` | 4 |
| `InferenceEngine.Tokenizers.Tests` | 8 |
| `InferenceEngine.Cli.Tests` | 20 |
| `InferenceEngine.Models.Tests` | 33 (incl. a bit-exact FNV-1a golden-logit-hash regression test against the real model) |
| `InferenceEngine.Engine.Tests` | 32 |

Run with `dotnet test`. Two of the `Models.Tests` cases (the golden-logit baseline) additionally
need `INFERENCE_MODEL=<path.gguf>` set to a real GGUF file — they skip cleanly, not fail, when it
isn't. See each test project's own `README.md` for the business scenarios covered.

### Architecture decisions

| ADR | Status |
|---|---|
| [Solution and project layout](docs/architecture/260811-solution-and-project-layout.md) | proposed |
| [Project challenges and how to address them](docs/architecture/260901-project-challenges-and-how-to-address-them.md) | proposed |
| [Paged + head-major KV-cache](docs/architecture/260914-kv-cache.md) | proposed |
| Attention & transformer, model format loading, tokenization, sampling pipeline | still `planned` placeholders — each is already implemented in code, the ADR write-up just hasn't caught up |
| HuggingFace model acquisition, performance baseline | still `planned` placeholders, and genuinely not started — models are fetched via the dotLLM CLI as a stopgap, not a real download path of our own, and there's no formal performance-baseline methodology yet beyond the ad hoc measurements in `experiments/` |

### Known limitations

- **Only F32/F16 GGUF tensors load.** Every quantized type (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, …)
  parses as metadata/shape but throws `NotSupportedException` on read. Dequantization math is
  already researched ([docs/investigation/gguf-format-research.md](docs/investigation/gguf-format-research.md))
  but not implemented.
- **Only `general.architecture == "llama"` loads** — Mistral/Qwen/Phi-family GGUF files are
  rejected today, even though they're structurally similar.
- **Single-token forward pass only** — prefill is a sequential loop of single-token calls, not a
  batched matmul. Fine for short prompts; for a long prompt this is the dominant cost (see the
  KV-cache benchmark: ~29s for a ~1,900-token prompt + 32 decode tokens on the 135M model).
- **Weights are copied, not zero-copy** — `LlamaWeights` materializes every tensor into a managed
  `float[]` at load, ~2x memory versus the on-disk F16 size.

### Path to running bigger HuggingFace models

Yes — **GGUF**, matching llama.cpp/HuggingFace's own local-inference ecosystem, not ONNX or
safetensors. Where each size class stands:

- **Tiny models (≤ ~500M params): already possible today**, no code changes needed — point
  `--model` at any **F16 (or F32) GGUF file of a Llama-architecture model** from HuggingFace (e.g.
  a `bartowski/*-GGUF` repo's `-f16.gguf` file, when one is published alongside the quantized
  variants). SmolLM2-135M-Instruct is the proof; SmolLM2-360M or TinyLlama-1.1B in F16 should work
  unmodified, memory and CPU speed permitting.
- **Medium models (roughly 1B–8B params): blocked mainly on one thing** — quantized tensor
  dequantization. Almost every GGUF repo on HuggingFace above a few hundred million parameters
  publishes Q4_K_M/Q5_K_M/Q6_K/Q8_0 as the practical download; full F16/F32 files for models that
  size are large (a 7–8B model in F16 is 14–16 GB) and often not published at all. Q4_K/Q6_K/Q8_0
  dequant math is already worked out in the GGUF research notes — implementing it (following the
  same research → ADR → implementation → benchmark cycle the KV-cache work just went through) is
  the natural next milestone, and the one that actually unlocks "download a model from HuggingFace
  and run it" for anything beyond the current tiny test model. Zero-copy mmap reads and
  multi-architecture support (Mistral/Qwen/Phi) would matter more at this size too, but are
  secondary to quantization.

This project has no sprint schedule or committed dates (see AGENTS.md's working style — small,
reviewable steps, not a roadmap with deadlines), so the honest answer to "when" is "whenever the
quantization ADR and implementation land," not a calendar date.

See [docs/investigation/status.md](docs/investigation/status.md) for the investigation-phase
history and [CHANGELOG.md](CHANGELOG.md) for the detailed change-by-change log.

## Project Docs

- [AGENTS.md](AGENTS.md) — ground rules for any AI coding assistant (stack, `unsafe` policy,
  ADR process)
- [CLAUDE.md](CLAUDE.md) — Claude Code–specific config (subagents, hooks)
- [docs/architecture/](docs/architecture/) — architecture decisions in [MADR](https://adr.github.io/madr/) format
- [docs/investigation/](docs/investigation/) — research notes and architecture traces
- [docs/scenarios/](docs/scenarios/) — Gherkin (`.feature`) business-scenario specs, one per `src/` project
- [experiments/](experiments/) — benchmark data and reference measurements
- [CHANGELOG.md](CHANGELOG.md) — notable changes
