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

    public ModelConfig Config { get; }

    private LlamaModel(ModelConfig config, LlamaWeights weights)
    {
        Config = config;
        _weights = weights;
        _qDim = config.NumAttentionHeads * config.HeadDim;
        _kvDim = config.NumKvHeads * config.HeadDim;
        _attentionScale = 1f / MathF.Sqrt(config.HeadDim);

        _hidden = new float[config.HiddenSize];
        _normed = new float[config.HiddenSize];
        _q = new float[_qDim];
        _attnOut = new float[_qDim];
        _attnProjected = new float[config.HiddenSize];
        _ffnGate = new float[config.FfnHiddenSize];
        _ffnUp = new float[config.FfnHiddenSize];
        _ffnSilu = new float[config.FfnHiddenSize];
        _ffnDown = new float[config.HiddenSize];
        _scores = new float[config.MaxSeqLen];
        _probs = new float[config.MaxSeqLen];
        _logits = new float[config.VocabSize];
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

    public ReadOnlySpan<float> Forward(int tokenId, int position, IKvCache kvCache, bool needLogits)
    {
        var hiddenSize = Config.HiddenSize;
        var groupSize = Config.NumAttentionHeads / Config.NumKvHeads;

        _weights.TokenEmbedding.AsSpan(tokenId * hiddenSize, hiddenSize).CopyTo(_hidden);

        for (var layer = 0; layer < Config.NumLayers; layer++)
        {
            var lw = _weights.Layers[layer];

            Ops.RmsNorm(_hidden, lw.AttnNorm, Config.RmsNormEps, _normed);

            var kSlot = kvCache.KeySlot(layer, position);
            var vSlot = kvCache.ValueSlot(layer, position);
            Ops.MatVec(lw.AttnQ, _qDim, hiddenSize, _normed, _q);
            Ops.MatVec(lw.AttnK, _kvDim, hiddenSize, _normed, kSlot);
            Ops.MatVec(lw.AttnV, _kvDim, hiddenSize, _normed, vSlot);

            Ops.Rope(_q, Config.NumAttentionHeads, Config.HeadDim, position, Config.RopeFreqBase);
            Ops.Rope(kSlot, Config.NumKvHeads, Config.HeadDim, position, Config.RopeFreqBase);

            var contextLength = position + 1;
            for (var qh = 0; qh < Config.NumAttentionHeads; qh++)
            {
                var kvh = qh / groupSize;
                var qHead = _q.AsSpan(qh * Config.HeadDim, Config.HeadDim);

                for (var t = 0; t < contextLength; t++)
                {
                    var kHead = kvCache.Key(layer, t).Slice(kvh * Config.HeadDim, Config.HeadDim);
                    _scores[t] = TensorPrimitives.Dot(qHead, kHead) * _attentionScale;
                }

                var scores = _scores.AsSpan(0, contextLength);
                var probs = _probs.AsSpan(0, contextLength);
                Ops.Softmax(scores, probs);

                var outHead = _attnOut.AsSpan(qh * Config.HeadDim, Config.HeadDim);
                outHead.Clear();
                for (var t = 0; t < contextLength; t++)
                {
                    var vHead = kvCache.Value(layer, t).Slice(kvh * Config.HeadDim, Config.HeadDim);
                    TensorPrimitives.MultiplyAdd(vHead, probs[t], outHead, outHead);
                }
            }

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
