using InferenceEngine.Core;

namespace InferenceEngine.Engine;

/// <summary>
/// Contiguous <c>float[]</c>-per-layer KV-cache — the "start simple" design from the
/// kv-cache ADR. Sized to exactly <c>promptLength + maxNewTokens</c> rather than the model's
/// full <see cref="ModelConfig.MaxSeqLen"/>, since a short-lived POC session never needs the
/// full context window reserved.
/// </summary>
internal sealed class SimpleKvCache : IKvCache
{
    private readonly float[][] _keys;
    private readonly float[][] _values;
    private readonly int _kvDim;

    public int Length { get; }

    public SimpleKvCache(int numLayers, int length, int kvDim)
    {
        Length = length;
        _kvDim = kvDim;
        _keys = new float[numLayers][];
        _values = new float[numLayers][];
        for (var layer = 0; layer < numLayers; layer++)
        {
            _keys[layer] = new float[length * kvDim];
            _values[layer] = new float[length * kvDim];
        }
    }

    public Span<float> KeySlot(int layer, int position) => _keys[layer].AsSpan(position * _kvDim, _kvDim);

    public Span<float> ValueSlot(int layer, int position) => _values[layer].AsSpan(position * _kvDim, _kvDim);

    public ReadOnlySpan<float> Key(int layer, int position) => KeySlot(layer, position);

    public ReadOnlySpan<float> Value(int layer, int position) => ValueSlot(layer, position);
}
