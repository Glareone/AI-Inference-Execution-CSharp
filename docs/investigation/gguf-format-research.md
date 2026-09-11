# GGUF Format Research

## Granny Explanation

Imagine a massive book (an AI model) that is 50 volumes long. GGUF is a way to compress
those 50 volumes into 5–10 while keeping it readable enough, and to organize it so your
reader can flip to any page instantly without loading the whole thing into memory first.

Technically: a single-file binary format packaging everything needed to run an AI model —
compressed weights, vocabulary, architecture description — into one file that can be
memory-mapped for near-instant loading. Created by Georgi Gerganov (llama.cpp author) to
replace older fragmented formats (GGML, GGJT).

## Binary Layout

```
+=====================+
| HEADER (24 bytes)   |  magic "GGUF" + version(3) + tensor_count + kv_count
+=====================+
| METADATA KV PAIRS   |  length-prefixed UTF-8 keys + typed values
+=====================+
| TENSOR INFO ARRAY   |  name + dims + quant_type + offset (per tensor)
+---------------------+
| ALIGNMENT PADDING   |  pad to 32 bytes (default)
+=====================+
| TENSOR DATA BLOB    |  bulk of the file
+=====================+
```

### Header (24 bytes, all little-endian)

| Offset | Size | Field | Type |
|--------|------|-------|------|
| 0x00 | 4 | magic | `GGUF` (0x47475546) |
| 0x04 | 4 | version | uint32 (currently 3) |
| 0x08 | 8 | tensor_count | uint64 |
| 0x10 | 8 | metadata_kv_count | uint64 |

### Metadata Encoding

Each KV: `[key_len: u64][key: utf8][value_type: u32][value: variable]`

13 value types: UINT8(0), INT8(1), UINT16(2), INT16(3), UINT32(4), INT32(5),
FLOAT32(6), BOOL(7), STRING(8), ARRAY(9), UINT64(10), INT64(11), FLOAT64(12).

Arrays: `[elem_type: u32][count: u64][elements...]` — homogeneous.

### Tensor Info (per tensor)

`[name_len: u64][name: utf8][n_dims: u32][dims: u64[n_dims]][type: u32][offset: u64]`

Offset is relative to tensor data section start. Dimensions in GGML order (reversed
from row-major).

## Quantization Types

### Legacy (block size = 32)

| Type | Bytes/Block | Bits/Weight | Dequant |
|------|-------------|-------------|---------|
| Q4_0 | 18 (2 scale + 16 data) | 4.5 | `w = d × (q - 8)` |
| Q5_0 | 22 (2 scale + 4 high_bits + 16 data) | 5.5 | `w = d × (q - 16)` |
| Q8_0 | 34 (2 scale + 32 data) | 8.5 | `w = d × q` |

### K-Quants (block size = 256, "super-blocks")

Double quantization: 8 sub-blocks of 32 weights. Sub-block scales are themselves
quantized to 6–8 bits, with a single FP16 super-scale.

**Q4_K** (used in Q4_K_M): 144 bytes / 256 weights = 4.5 bpw
- `d: f16` — super-scale
- `dmin: f16` — super-minimum scale
- `scales[12]` — 8 sub-block scales + 8 minimums packed as 6-bit values
- `qs[128]` — 256 × 4-bit quantized weights
- Dequant: `w = d × sub_scale × q + dmin × sub_min`

**Q6_K**: 210 bytes / 256 weights = 6.5625 bpw
- Symmetric (no minimum), 16 sub-blocks of 16, 8-bit sub-scales
- Dequant: `w = d × scale × q`

### Mixed-Precision Policy (_S/_M/_L)

Not a different format — same block structures, but different tensors get different types:
- **K_S**: uniform, only norms at FP16
- **K_M**: attention V + FFN down → Q6_K, rest → Q4_K
- **K_L**: even more layers promoted

## Memory-Mapped Loading

`mmap()` creates a virtual address mapping without reading the file. Pages load on
first access (page fault → OS reads 4KB from disk). Benefits:
- Multi-GB models "load" in milliseconds (only sets up virtual mapping)
- Multiple processes share physical pages
- OS manages eviction automatically

32-byte alignment satisfies AVX2 SIMD requirements.

## .NET Implementation

**Parsing** (no unsafe needed):
- `MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, Read)`
- `BinaryPrimitives.ReadUInt32LittleEndian(span)` for endian-safe reads

**Tensor access** (unsafe for zero-copy):
- `accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr)`
- Cast to `ReadOnlySpan<byte>` for block reads

**Dequantization** (SIMD hot path):
- `System.Runtime.Intrinsics` (AVX2/NEON)
- Define C# structs mirroring C block layouts with `[StructLayout(Sequential, Pack=1)]`

## Sources

- [GGUF Spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [K-Quants Docs](https://github.com/iuliaturc/gguf-docs/blob/main/k-quants.md)
- [HuggingFace GGUF](https://huggingface.co/docs/hub/en/gguf)
- [dotLLM Blog](https://kokosa.dev/blog/2026/dotllm/)
