namespace InferenceEngine.Engine.Sampling;

internal interface ILogitsProcessor
{
    /// <summary>
    /// Implementations must never mask the EOS token to -infinity. This guarantees at least
    /// one valid choice always survives, so generation can end instead of erroring or picking
    /// an arbitrary (possibly banned) token when every other position is masked.
    /// </summary>
    void Apply(Span<float> logits, ReadOnlySpan<int> generatedTokenIds);
}
