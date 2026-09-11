namespace InferenceEngine.Engine.Sampling;

/// <summary>One step in the logits-processing chain (temperature, top-k, top-p, ...), applied in place.</summary>
internal interface ISamplerStep
{
    void Apply(Span<float> logits);
}
