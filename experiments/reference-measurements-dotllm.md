# Reference Measurements: dotLLM

Baseline measurements from dotLLM v0.1.0-preview.3, installed via `dotnet tool install -g DotLLM.Cli --prerelease`.

## Environment

- **Machine**: MacBook (Apple Silicon / macOS Darwin 24.6.0)
- **dotLLM version**: 0.1.0-preview.3
- **Runtime**: .NET 10
- **Date**: 2026-08-15

## Model: SmolLM2-135M-Instruct Q4_K_M

- **Source**: `bartowski/SmolLM2-135M-Instruct-GGUF`
- **File**: `SmolLM2-135M-Instruct-Q4_K_M.gguf` (101 MB)
- **Architecture**: Llama, 30 layers, 576 hidden, 9 query heads, 3 KV heads (GQA 3:1)
- **Vocab**: 49,152 tokens (GPT-2 BPE)
- **Context**: 8,192 tokens

### Run: single-shot generation

```
Prompt: "What is machine learning? Explain briefly."
Max tokens: 100
Threads: 12
Sampling: greedy
```

### Results

| Metric | Value |
|--------|-------|
| **Decode throughput** | 53.91 tok/s |
| **Prefill throughput** | 27.60 tok/s |
| **Total throughput** | 50.37 tok/s |
| **Model load time** | 89.9 ms |
| **Prefill latency** | 289.8 ms (8 tokens) |
| **Decode latency** | 1,836.4 ms (99 tokens) |
| **Sampling time** | 8.0 ms (100 tokens) |
| **Total time** | 2,144.0 ms (108 tokens) |

### Memory

| Component | Size |
|-----------|------|
| **Weights** | 98.9 MiB (memory-mapped) |
| **Compute** | 3.8 MiB |
| **KV Cache** | 4.7 MiB (108 slots) |
| **Total** | 107.4 MiB |

## GGUF File Analysis

### Metadata (37 KV pairs)

Key model config extracted from GGUF metadata:

| Key | Value |
|-----|-------|
| `general.architecture` | llama |
| `general.name` | Smollm2 135M 8k Lc100K Mix1 Ep2 |
| `llama.block_count` | 30 |
| `llama.context_length` | 8192 |
| `llama.embedding_length` | 576 |
| `llama.feed_forward_length` | 1536 |
| `llama.attention.head_count` | 9 |
| `llama.attention.head_count_kv` | 3 |
| `llama.rope.freq_base` | 100,000 |
| `llama.rope.dimension_count` | 64 |
| `llama.vocab_size` | 49,152 |
| `tokenizer.ggml.model` | gpt2 |
| `tokenizer.ggml.merges` | 48,900 merge rules |

### Tensor Quantization (272 tensors)

Q4_K_M uses importance-weighted mixed precision — not uniform quantization:

| Tensor | Quant | Shape | Rationale |
|--------|-------|-------|-----------|
| `token_embd.weight` | Q8_0 | 576 × 49,152 | Embedding — critical for quality |
| `blk.N.attn_v.weight` | Q8_0 | 576 × 192 | V projection — impacts output directly |
| `blk.N.ffn_down.weight` | Q6_K | 1,536 × 576 | FFN output — medium-high precision |
| `blk.N.attn_q.weight` | Q5_0 | 576 × 576 | Q projection — less sensitive |
| `blk.N.attn_k.weight` | Q5_0 | 576 × 192 | K projection — less sensitive |
| `blk.N.ffn_gate.weight` | Q5_0 | 576 × 1,536 | FFN gate — less sensitive |
| `blk.N.ffn_up.weight` | Q5_0 | 576 × 1,536 | FFN up — less sensitive |
| `blk.N.attn_norm.weight` | F32 | 576 | RMSNorm — tiny, keep full precision |
| `blk.N.ffn_norm.weight` | F32 | 576 | RMSNorm — tiny, keep full precision |

### Observations

1. **Load time is negligible** (90ms) thanks to memory-mapped file — the OS pages data
   on demand, no upfront copy of 101 MB into RAM.
2. **Decode is ~2× faster than prefill** — decode is memory-bandwidth bound (one token
   at a time), prefill is compute-bound (matrix multiply over all prompt tokens).
3. **KV cache is tiny** for short sequences (4.7 MiB for 108 tokens) but would grow
   linearly: at 8,192 context length it would be ~350 MiB.
4. **Mixed quantization** in Q4_K_M assigns higher precision to tensors that matter more
   for output quality (embeddings, V projections, FFN output). This is why K_M ("medium")
   preserves better quality than uniform Q4_0 at similar file size.

## Next Steps

- [ ] Run with Q8_0 for comparison (higher precision, larger file)
- [ ] Test with longer prompts to see prefill scaling
- [ ] Measure with different thread counts
- [ ] Compare with LLamaSharp on the same model (when we set that up)
