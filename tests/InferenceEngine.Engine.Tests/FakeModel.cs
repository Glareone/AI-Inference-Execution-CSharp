using InferenceEngine.Core;

namespace InferenceEngine.Engine.Tests;

/// <summary>Writes a marker into the KV-cache slot it's given, so a cache sizing/indexing bug surfaces as an exception instead of being silently absorbed by a no-op fake.</summary>
internal sealed class FakeModel(ModelConfig config, int? preferredTokenId = null) : IModel
{
    // All-zero logits make greedy sampling pick index 0 (the first element never strictly
    // exceeded) — which happens to equal FakeTokenizer's hardcoded EosTokenId, so any test that
    // needs the decode loop to run past its first step must give a non-zero token an edge.
    private readonly float[] _logits = BuildLogits(config.VocabSize, preferredTokenId);

    public ModelConfig Config { get; } = config;

    /// <summary>Number of times <see cref="Forward"/> has been called — lets a test assert a forward pass was (or wasn't) run.</summary>
    public int ForwardCallCount { get; private set; }

    public ReadOnlySpan<float> Forward(int tokenId, int position, IKvCache kvCache, bool needLogits)
    {
        ForwardCallCount++;
        kvCache.Reserve(position);
        kvCache.KeySlot(0, 0, position)[0] = tokenId;
        kvCache.ValueSlot(0, 0, position)[0] = tokenId;
        return needLogits ? _logits : ReadOnlySpan<float>.Empty;
    }

    private static float[] BuildLogits(int vocabSize, int? preferredTokenId)
    {
        var logits = new float[vocabSize];
        if (preferredTokenId is { } id)
        {
            logits[id] = 1f;
        }

        return logits;
    }
}
