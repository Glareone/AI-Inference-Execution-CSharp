using InferenceEngine.Core;
using InferenceEngine.Models.Gguf;

namespace InferenceEngine.Models.Llama;

internal sealed record LlamaLayerWeights(
    float[] AttnNorm,
    float[] AttnQ,
    float[] AttnK,
    float[] AttnV,
    float[] AttnOutput,
    float[] FfnNorm,
    float[] FfnGate,
    float[] FfnUp,
    float[] FfnDown);

/// <summary>
/// Materialized (F32) weights for a Llama-architecture model, loaded once from a
/// <see cref="GgufFile"/>. Trades the mmap's zero-copy residency for a plain <c>float[]</c>
/// per tensor — simpler math code at the cost of ~2x memory vs. the on-disk F16 size; see the
/// format-loading ADR for the zero-copy follow-up.
/// </summary>
internal sealed class LlamaWeights
{
    public required float[] TokenEmbedding { get; init; }

    /// <summary>LM head. Same array as <see cref="TokenEmbedding"/> when the model ties embeddings.</summary>
    public required float[] Output { get; init; }

    public required float[] OutputNorm { get; init; }

    public required LlamaLayerWeights[] Layers { get; init; }

    public static LlamaWeights Load(GgufFile gguf, ModelConfig config)
    {
        var tokenEmbedding = gguf.ReadTensorAsF32("token_embd.weight");

        // SmolLM2 (and many small models) tie the LM head to the embedding table and omit
        // "output.weight" entirely — fall back to the embedding, per the format-loading note.
        var output = gguf.HasTensor("output.weight")
            ? gguf.ReadTensorAsF32("output.weight")
            : tokenEmbedding;

        var layers = new LlamaLayerWeights[config.NumLayers];
        for (var i = 0; i < config.NumLayers; i++)
        {
            layers[i] = new LlamaLayerWeights(
                AttnNorm: gguf.ReadTensorAsF32($"blk.{i}.attn_norm.weight"),
                AttnQ: gguf.ReadTensorAsF32($"blk.{i}.attn_q.weight"),
                AttnK: gguf.ReadTensorAsF32($"blk.{i}.attn_k.weight"),
                AttnV: gguf.ReadTensorAsF32($"blk.{i}.attn_v.weight"),
                AttnOutput: gguf.ReadTensorAsF32($"blk.{i}.attn_output.weight"),
                FfnNorm: gguf.ReadTensorAsF32($"blk.{i}.ffn_norm.weight"),
                FfnGate: gguf.ReadTensorAsF32($"blk.{i}.ffn_gate.weight"),
                FfnUp: gguf.ReadTensorAsF32($"blk.{i}.ffn_up.weight"),
                FfnDown: gguf.ReadTensorAsF32($"blk.{i}.ffn_down.weight"));
        }

        return new LlamaWeights
        {
            TokenEmbedding = tokenEmbedding,
            Output = output,
            OutputNorm = gguf.ReadTensorAsF32("output_norm.weight"),
            Layers = layers,
        };
    }
}
