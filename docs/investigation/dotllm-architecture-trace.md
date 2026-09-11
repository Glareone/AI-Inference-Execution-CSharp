# dotLLM Architecture Trace

Source: https://github.com/kkokosa/dotLLM (v0.1.0-preview.3, GPLv3)

## Solution Structure (17 projects)

```
Core (leaf — no deps)
  ├── Tokenizers
  ├── Cpu
  ├── HuggingFace
  ├── Diagnostics
  ├── Telemetry
  ├── Models (Core + Cpu + Tokenizers)
  ├── Engine (Core + Models + Tokenizers)
  ├── Cuda (Core + Cpu + Engine + Models)
  ├── Server (all above + Telemetry)
  └── Cli (all above)
```

## Core Interfaces

| Interface | Purpose | Key members |
|-----------|---------|-------------|
| `ITensor` | Tensor abstraction | Shape, DType, DeviceId, DataPointer (nint), IDisposable |
| `TensorShape` | Shape value type | `readonly record struct`, int[] Dimensions, Rank, ElementCount |
| `IBackend` | Compute backend | AllocateOnDevice, CopyBetweenDevices, multi-GPU |
| `IModel` | Model abstraction | Config, Forward(tokenIds, positions, deviceId, kvCache) → logits |
| `IModelArchitecture` | Model factory | SupportedArchitectures + CreateModel(config, backend) |
| `ModelConfig` | Model metadata | record: Architecture, VocabSize, HiddenSize, NumLayers, NumAttentionHeads, NumKvHeads, HeadDim, MaxSeqLen, RoPEConfig... |
| `IKvCache` | KV-cache | Update, GetKeys/Values, Rollback (for speculative decoding) |
| `IAttentionStrategy` | Attention dispatch | ComputeAttention(q, k, v, mask, scale) — strategy pattern |
| `ISamplerStep` | Sampling pipeline step | Apply(Span<float> logits, SamplerContext) — composable chain |
| `IDecodingConstraint` | Constrained generation | GetAllowedTokens() → TokenMask, Advance(tokenId) |

## GGUF Loading Path

```
GgufFile.Open(filePath)
  1. FileStream + BinaryReader (sequential read for header/metadata/tensor-info)
  2. GgufReader.ReadHeader()
     → validate magic 0x46554747 ("GGUF")
     → version (v2: uint32 counts, v3: uint64 counts)
     → GgufHeader(Version, TensorCount, MetadataKvCount) readonly record struct
  3. GgufReader.ReadMetadata()
     → iterate KV pairs: length-prefixed UTF-8 key + type enum + typed value
     → boxing acceptable (one-time load)
     → Dictionary<string, GgufMetadataValue>
  4. GgufReader.ReadTensorInfos()
     → name, nDims, dimensions[], quantType, offset
     → List<GgufTensorDescriptor> (readonly record struct)
  5. dataSectionOffset = AlignUp(position, alignment=32)
  6. Validate tensor bounds
  7. MemoryMappedFile.CreateFromFile(path, Read)
     → CreateViewAccessor(0, 0, Read)  // entire file
     → AcquirePointer → raw byte*
     → DataBasePointer = base + PointerOffset + dataSectionOffset
```

Key types: `GgufHeader`, `GgufMetadataValue`, `GgufMetadata` (typed getters),
`GgufTensorDescriptor`, `GgufFile` (owns mmap + exposes DataBasePointer).

## Token Generation Flow

```
CLI (RunCommand)
  → GgufFile.Open(path)
  → GgufModelConfigExtractor.Extract(metadata)   // reads {arch}.* keys
  → GgufBpeTokenizerFactory.Load(metadata)        // tokenizer from GGUF
  → TransformerModel.LoadFromGguf(gguf, config)
      → TransformerWeights.LoadFromGguf()          // resolve mmap pointers
      → weights.RepackWeights()                    // R4-interleaved for cache locality
      → TransformerForwardState(...)               // pre-allocate scratch buffers
  → TextGenerator(model, tokenizer, kvFactory)
  → generator.GenerateStreamingTokensAsync(prompt, options)
```

## TextGenerator Loop

```
1. ENCODE:  tokenizer.Encode(prompt) → int[] promptIds
2. BUILD:   SamplerPipeline (temperature → topK → topP → minP)
            IDecodingConstraint (if structured output)
            Stop conditions (EOS, maxTokens, stop strings)
3. KV-CACHE: check PrefixCache for hit, else allocate SimpleKvCache
4. PREFILL:  model.Forward(promptTokens, positions, kvCache)
             → logits for last position → sample first token
5. DECODE LOOP:
   for each step:
     model.Forward([lastToken], [pos], kvCache)  // single token
     apply constraint mask
     pipeline.Sample(logits, generatedIds)
     constraint.Advance(tokenId)
     check stop → yield token
```

## TransformerModel.Forward() — CPU

```
1. EMBEDDING LOOKUP — dequantize/copy one row per token
2. TRANSFORMER LAYERS (×numLayers):
   a. residual = hidden
   b. RMSNorm (pre-attention)
   c. Q/K/V projections (quantized GEMV for decode, GEMM for prefill)
   d. RoPE (in-place on Q and K)
   e. Attention: KV-cache update + attend over full context
   f. Output projection + residual add
   g. residual = hidden
   h. FFN: RMSNorm → Gate/Up → SwiGLU fusion → Down
   i. residual add
3. FINAL NORM (RMSNorm)
4. LM HEAD (GEMM → logits [seqLen, vocabSize])
```

Decode (seqLen=1): fused dispatch (FusedQkvDecode, FusedGateUpDecode).
Prefill (seqLen>1): unfused batched GEMM.

## SamplerPipeline.Sample()

```
1. Run logit processors (repetition penalty)
2. If greedy (temp≤0): return IndexOfMax(logits)
3. Run sampler steps: temperature → top-k → top-p → min-p
4. CategoricalSampler.Sample(logits, rng)
```
