using InferenceEngine.Core;

namespace InferenceEngine.Engine.Tests;

/// <summary>
/// Hand-written <see cref="IModel"/> test double: returns a fixed-size zero logits buffer and
/// writes a marker into the KV-cache slot it's given, so a bug in cache sizing/indexing still
/// surfaces as an exception rather than being silently absorbed by a no-op fake.
/// </summary>
internal sealed class FakeModel(ModelConfig config) : IModel
{
    private readonly float[] _logits = new float[config.VocabSize];

    public ModelConfig Config { get; } = config;

    public ReadOnlySpan<float> Forward(int tokenId, int position, IKvCache kvCache, bool needLogits)
    {
        kvCache.KeySlot(0, position)[0] = tokenId;
        kvCache.ValueSlot(0, position)[0] = tokenId;
        return needLogits ? _logits : ReadOnlySpan<float>.Empty;
    }
}
