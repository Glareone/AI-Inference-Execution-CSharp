# Investigation Overview

## Goal

Understand how LLM inference engines work from the inside — deeply enough to explain
every component to anyone. Build a working C# implementation that demonstrates each piece.

## Reference: dotLLM by Konrad Kokosa

- **Repo**: https://github.com/kkokosa/dotLLM
- **What it is**: Ground-up LLM inference engine in pure C#/.NET 10. Not a wrapper.
- **License**: GPLv3
- **Scope**: GGUF loading, BPE/SentencePiece tokenization, transformer inference (CPU + CUDA),
  paged KV-cache, composable sampling, constrained decoding, OpenAI-compatible server.
- **Models**: Llama 1–3.2, Mistral, Phi, Qwen, DeepSeek (MLA), SmolLM3, Gemma 4, MoE
- **Performance**: 66–88% of llama.cpp decode throughput on CPU; prefill 2–5× slower
  (compute-bound, RyuJIT register pressure)
- **Architecture**: layered NuGet packages — Core → Models/Tokenizers → Engine → Cpu/Cuda → Server
- **Key techniques**: zero-GC (NativeMemory), SIMD (AVX2/512), memory-mapped GGUF loading,
  paged KV-cache with quantization, PTX CUDA kernels via Driver API
- **Blog**: https://kokosa.dev/blog/2026/dotllm/

## Other C# Inference Projects

| Project | Approach | Notes |
|---------|----------|-------|
| **LLamaSharp** | P/Invoke wrapper around llama.cpp | Most mature, but not a learning tool |
| **ONNX Runtime GenAI** | Microsoft's ONNX-based inference | Different format (ONNX, not GGUF) |
| **LM-Kit.NET** | Commercial SDK | Full product, not open for study |
| **Microsoft.ML.Tokenizers** | Standalone tokenizer library | BPE, SentencePiece, Tiktoken — reusable |

## Key Concepts to Investigate

Each topic gets an ADR (MADR format) documenting both understanding and implementation decision.

Each pending ADR is named `YYMMDD-<slug>.md` when it's written (no number assigned in advance).

| # | Topic | ADR status |
|---|-------|------------|
| 1 | GGUF format & model loading | pending |
| 2 | Tokenization (BPE, SentencePiece) | pending |
| 3 | Transformer architecture & attention | pending |
| 4 | KV-cache design | pending |
| 5 | Logits processing & sampling | pending |
| 6 | HuggingFace model acquisition | pending |
| 7 | Performance baseline & expectations | pending |

## Test Model

Using **SmolLM2-135M-Instruct** (bartowski Q4_K_M, 101 MB) for development and testing:
- Llama architecture, 30 layers, 576 hidden dim
- GQA: 9 query heads, 3 KV heads (3:1 ratio)
- 49,152 token vocabulary, GPT-2 BPE tokenizer
- 8,192 context length
- Small enough to iterate fast, large enough to produce coherent text

## Investigation Agents

Three agents in `.claude/agents/` for structured research:
- **researcher** — web research on algorithms, architecture, specs
- **huggingface-explorer** — HuggingFace ecosystem, model formats, download APIs
- **code-reader** — trace code paths in dotLLM, LLamaSharp, understand patterns
