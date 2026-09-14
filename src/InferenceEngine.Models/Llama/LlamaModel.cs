using System.Numerics.Tensors;
using InferenceEngine.Core;
using InferenceEngine.Models.Gguf;
using InferenceEngine.Models.Math;

namespace InferenceEngine.Models.Llama;

/// <summary>
/// Single-token Llama forward pass (embeddings -> RMSNorm -> GQA attention with RoPE ->
/// SwiGLU FFN -> LM head), built on <see cref="TensorPrimitives"/> with no custom kernels —
/// see the attention-and-transformer ADR. Prefill is a loop of single-token calls; there is no
/// batched (multi-token) path in this POC.
/// </summary>
public sealed class LlamaModel : IModel
{
    private readonly LlamaWeights _weights;
    private readonly int _qDim;
    private readonly int _kvDim;
    private readonly float _attentionScale;
    private readonly int _scoreStride;

    // Scratch buffers, allocated once so a decode step does no heap allocation.
    private readonly float[] _hidden;
    private readonly float[] _normed;
    private readonly float[] _q;
    private readonly float[] _attnOut;
    private readonly float[] _attnProjected;
    private readonly float[] _ffnGate;
    private readonly float[] _ffnUp;
    private readonly float[] _ffnSilu;
    private readonly float[] _ffnDown;
    private readonly float[] _scores;
    private readonly float[] _probs;
    private readonly float[] _logits;
    private readonly float[] _ropeCos;
    private readonly float[] _ropeSin;

    public ModelConfig Config { get; }

    private LlamaModel(ModelConfig config, LlamaWeights weights)
    {
        if (config.NumAttentionHeads % config.NumKvHeads != 0)
        {
            throw new NotSupportedException(
                $"NumAttentionHeads ({config.NumAttentionHeads}) must be an exact multiple of " +
                $"NumKvHeads ({config.NumKvHeads}) for grouped-query attention — an inexact " +
                "ratio would silently truncate the query-head group size.");
        }

        Config = config;
        _weights = weights;
        _qDim = config.NumAttentionHeads * config.HeadDim;
        _kvDim = config.NumKvHeads * config.HeadDim;
        _attentionScale = 1f / MathF.Sqrt(config.HeadDim);

        var groupSize = config.NumAttentionHeads / config.NumKvHeads;
        _scoreStride = config.MaxSeqLen;

        _hidden = new float[config.HiddenSize];
        _normed = new float[config.HiddenSize];
        _q = new float[_qDim];
        _attnOut = new float[_qDim];
        _attnProjected = new float[config.HiddenSize];
        _ffnGate = new float[config.FfnHiddenSize];
        _ffnUp = new float[config.FfnHiddenSize];
        _ffnSilu = new float[config.FfnHiddenSize];
        _ffnDown = new float[config.HiddenSize];
        _scores = new float[groupSize * _scoreStride];
        _probs = new float[groupSize * _scoreStride];
        _logits = new float[config.VocabSize];
        _ropeCos = new float[config.HeadDim / 2];
        _ropeSin = new float[config.HeadDim / 2];
    }

    public static LlamaModel LoadFromGguf(string path)
    {
        using var gguf = GgufFile.Open(path);

        var architecture = gguf.Metadata.GetString("general.architecture");
        if (architecture != "llama")
        {
            throw new NotSupportedException(
                $"Architecture '{architecture}' is not implemented yet (this POC supports 'llama' only).");
        }

        var hiddenSize = (int)gguf.Metadata.GetU32("llama.embedding_length");
        var numHeads = (int)gguf.Metadata.GetU32("llama.attention.head_count");

        // Fallbacks below match llama.cpp/HF conventions for the llama architecture, used only
        // when a GGUF file omits a key that's normally present — not this POC's own defaults.
        var config = new ModelConfig(
            Architecture: architecture,
            VocabSize: (int)gguf.Metadata.GetU32("llama.vocab_size"),
            HiddenSize: hiddenSize,
            NumLayers: (int)gguf.Metadata.GetU32("llama.block_count"),
            NumAttentionHeads: numHeads,
            NumKvHeads: (int)gguf.Metadata.GetU32("llama.attention.head_count_kv"),
            HeadDim: (int)gguf.Metadata.GetU32OrDefault("llama.rope.dimension_count", (uint)(hiddenSize / numHeads)),
            FfnHiddenSize: (int)gguf.Metadata.GetU32("llama.feed_forward_length"),
            MaxSeqLen: (int)gguf.Metadata.GetU32("llama.context_length"),
            RmsNormEps: gguf.Metadata.GetF32OrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
            RopeFreqBase: gguf.Metadata.GetF32OrDefault("llama.rope.freq_base", 10000f));

        var weights = LlamaWeights.Load(gguf, config);
        return new LlamaModel(config, weights);
    }

    /// <inheritdoc/>
    public ReadOnlySpan<float> Forward(int tokenId, int position, IKvCache kvCache, bool needLogits)
    {
        if (position < 0 || position >= Config.MaxSeqLen)
        {
            throw new ArgumentOutOfRangeException(
                nameof(position), position,
                $"Position must be within [0, {Config.MaxSeqLen}) — the attention scratch buffers are sized to MaxSeqLen.");
        }

        var hiddenSize = Config.HiddenSize;
        var headDim = Config.HeadDim;

        // The rotation angle depends only on (position, freqBase, headDim) — not on layer or
        // head — so the table is built once per Forward call (position is fixed for the whole
        // call) instead of once per layer (60x/token for a 30-layer model: once for Q, once for
        // cached K, per layer).
        Ops.RopeTable(headDim, position, Config.RopeFreqBase, _ropeCos, _ropeSin);

        _weights.TokenEmbedding.AsSpan(tokenId * hiddenSize, hiddenSize).CopyTo(_hidden);

        for (var layer = 0; layer < Config.NumLayers; layer++)
        {
            var lw = _weights.Layers[layer];

            Ops.RmsNorm(_hidden, lw.AttnNorm, Config.RmsNormEps, _normed);

            Ops.MatVec(lw.AttnQ, _qDim, hiddenSize, _normed, _q);
            for (var h = 0; h < Config.NumAttentionHeads; h++)
            {
                Ops.RopeHead(_q.AsSpan(h * headDim, headDim), _ropeCos, _ropeSin);
            }

            // Per-KV-head write + RoPE, not one packed kvDim-wide MatVec/Rope call: the
            // head-major cache exposes one head's slot at a time. Each head's row range
            // [kvh*headDim, (kvh+1)*headDim) of AttnK/AttnV produces the identical dot products,
            // in the identical row order (0..kvDim), as the original single packed call — just
            // split at head boundaries — so this is bit-identical, not a numeric change.
            for (var kvh = 0; kvh < Config.NumKvHeads; kvh++)
            {
                var kSlot = kvCache.KeySlot(layer, kvh, position);
                var vSlot = kvCache.ValueSlot(layer, kvh, position);
                var rowOffset = kvh * headDim * hiddenSize;
                Ops.MatVec(lw.AttnK.AsSpan(rowOffset, headDim * hiddenSize), headDim, hiddenSize, _normed, kSlot);
                Ops.MatVec(lw.AttnV.AsSpan(rowOffset, headDim * hiddenSize), headDim, hiddenSize, _normed, vSlot);
                Ops.RopeHead(kSlot, _ropeCos, _ropeSin);
            }

            GqaAttention.Attend(
                kvCache, layer, position, _q, Config.NumAttentionHeads, Config.NumKvHeads,
                Config.HeadDim, _attentionScale, _scoreStride, _scores, _probs, _attnOut);

            Ops.MatVec(lw.AttnOutput, hiddenSize, _qDim, _attnOut, _attnProjected);
            TensorPrimitives.Add(_hidden, _attnProjected, _hidden);

            Ops.RmsNorm(_hidden, lw.FfnNorm, Config.RmsNormEps, _normed);
            Ops.MatVec(lw.FfnGate, Config.FfnHiddenSize, hiddenSize, _normed, _ffnGate);
            Ops.MatVec(lw.FfnUp, Config.FfnHiddenSize, hiddenSize, _normed, _ffnUp);
            Ops.SwiGlu(_ffnGate, _ffnUp, _ffnSilu, _ffnSilu);
            Ops.MatVec(lw.FfnDown, hiddenSize, Config.FfnHiddenSize, _ffnSilu, _ffnDown);
            TensorPrimitives.Add(_hidden, _ffnDown, _hidden);
        }

        if (!needLogits)
        {
            return ReadOnlySpan<float>.Empty;
        }

        Ops.RmsNorm(_hidden, _weights.OutputNorm, Config.RmsNormEps, _normed);
        Ops.MatVec(_weights.Output, Config.VocabSize, hiddenSize, _normed, _logits);
        return _logits;
    }
}
