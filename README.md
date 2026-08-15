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

Each topic is documented as an ADR (Architecture Decision Record) in `architecture/`,
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
architecture/                   ADRs in MADR format
investigation/                  Research notes, architecture traces
experiments/                    Benchmark data, reference measurements
```

See [ADR-0001](architecture/0001-solution-and-project-layout.md) for the rationale behind
this layout.

## Test Model

**SmolLM2-135M-Instruct** (bartowski Q4_K_M, 101 MB) — smallest model that produces
coherent text. Llama architecture, 30 layers, 576 hidden dim, GQA 3:1, 49K vocab.
Fast enough to iterate on (53.9 tok/s decode on our test machine via dotLLM).

## Claude Code Agents

This project uses AI-assisted development with structured agents:

| Agent | Purpose |
|-------|---------|
| `adr-writer-reviewer` | Write/review Architecture Decision Records (MADR format) |
| `csharp-dotnet` | C#/.NET implementation and code review |
| `researcher` | Web research on inference internals (algorithms, specs, papers) |
| `huggingface-explorer` | HuggingFace ecosystem (model formats, APIs, downloads) |
| `code-reader` | Read/analyze external repos (dotLLM, LLamaSharp) |

All agents use [context7](https://github.com/upstash/context7) for current library
documentation instead of relying on training data.

## Status

**Investigation phase** — researching internals before implementation. Solution scaffolding
exists and builds. dotLLM installed as reference tool. No inference code yet.
See [investigation/status.md](investigation/status.md) for detailed progress and
[CHANGELOG.md](CHANGELOG.md) for changes.

## Project Docs

- [AGENTS.md](AGENTS.md) — ground rules for any AI coding assistant (stack, `unsafe` policy,
  ADR process)
- [CLAUDE.md](CLAUDE.md) — Claude Code–specific config (subagents, hooks)
- [architecture/](architecture/) — architecture decisions in [MADR](https://adr.github.io/madr/) format
- [investigation/](investigation/) — research notes and architecture traces
- [experiments/](experiments/) — benchmark data and reference measurements
- [CHANGELOG.md](CHANGELOG.md) — notable changes
