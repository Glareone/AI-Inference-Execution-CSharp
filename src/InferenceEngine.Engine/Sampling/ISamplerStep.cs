namespace InferenceEngine.Engine.Sampling;

internal interface ISamplerStep
{
    void Apply(Span<float> logits);
}
