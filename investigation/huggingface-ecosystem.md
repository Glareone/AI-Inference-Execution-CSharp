# HuggingFace GGUF Ecosystem

## GGUF Quantization Providers

| Provider | Notes |
|----------|-------|
| **bartowski** | Most active, 20+ variants per model, detailed guidance |
| **unsloth** | 6x faster conversion, clean repos |
| **QuantFactory** | Automated pipeline, broad coverage |
| **TheBloke** | Original prolific quantizer, less active now |
| **ggml-org/gguf-my-repo** | Official HF Space for self-service conversion |

Repo naming: `{quantizer}/{ModelName}-GGUF`
File naming: `{ModelName}-{QuantType}.gguf`

## Quantization Quick Reference

| Quant | Bits/Weight | 7B Size | Quality vs F16 | When to use |
|-------|-------------|---------|-----------------|-------------|
| Q2_K | 2.6 | 2.7 GB | ~85% | Extreme compression only |
| Q4_K_M | 4.5 | 4.1 GB | ~96.5% | **Best general-purpose 4-bit** |
| Q5_K_M | 5.5 | 5.0 GB | ~98% | Good balance |
| Q6_K | 6.6 | 5.5 GB | ~99% | Near-lossless |
| Q8_0 | 8.0 | 7.7 GB | ~99.5% | Effectively lossless |
| F16 | 16.0 | 14 GB | baseline | Reference only |

K_S/K_M/K_L suffix: S = uniform precision, M = mixed (attention higher),
L = embeddings+output at Q8_0.

## HuggingFace Hub API

**Search**: `GET /api/models?search=SmolLM2&filter=gguf&sort=downloads`
**Model info**: `GET /api/models/{namespace}/{repo}` (includes `siblings` with file list)
**Download**: `GET /{namespace}/{repo}/resolve/{revision}/{filename}` (302 → CDN)
**Auth**: `Authorization: Bearer hf_xxxxx` — always pass a token, even for public repos
  (anonymous shares lower per-IP rate limit pool).

Rate limits (per 5 min): 500 API / 3000 downloads (anonymous),
  1000 / 5000 (free user with token).

## .NET Download Options

| Option | Package | Features |
|--------|---------|----------|
| `HuggingfaceHub` | NuGet | DownloadFileAsync, DownloadSnapshotAsync, resume, progress |
| `ElBruno.HuggingFace.Downloader` | NuGet | Resumable, atomic writes, RequiredFiles/OptionalFiles |
| Raw `HttpClient` | built-in | Full control, /resolve/ endpoint, Range header for resume |
| dotLLM's approach | DotLLM.HuggingFace | Stores at `~/.dotllm/models/{owner}/{repo}/` |

## GGUF-Embedded Tokenizer

GGUF files embed tokenizer data under `tokenizer.ggml.*`:

| Key | What it stores |
|-----|---------------|
| `tokenizer.ggml.model` | Algorithm: `"gpt2"` (BPE), `"llama"` (SPM) |
| `tokenizer.ggml.pre` | Pre-tokenizer variant (e.g., `"llama3"`, `"smollm"`) |
| `tokenizer.ggml.tokens` | Full vocabulary (string[]) |
| `tokenizer.ggml.merges` | BPE merge rules (string[]) |
| `tokenizer.ggml.token_type` | Per-token type: NORMAL, CONTROL, BYTE, etc. |
| `tokenizer.ggml.bos/eos_token_id` | Special token IDs |
| `tokenizer.chat_template` | Jinja2 chat template |

**Caveat**: GGUF embeds raw vocab + merges but NOT the pre-tokenizer pipeline
(regex splitting, normalization). For correct tokenization you must either:
1. Hard-code known pre-tokenizer patterns per `tokenizer.ggml.pre` value (llama.cpp approach)
2. Fetch `tokenizer.json` from the source model repo for the full pipeline

## Small Models for Testing

| Model | Params | Q4_K_M | Good for |
|-------|--------|--------|----------|
| **SmolLM2-135M** | 135M | ~105 MB | Fastest iteration, smallest coherent model |
| SmolLM2-360M | 360M | ~250 MB | Better quality, still tiny |
| Qwen2.5-0.5B | 0.5B | ~350 MB | Alternative architecture |
| TinyLlama-1.1B | 1.1B | ~670 MB | Solid quality, widely tested |
