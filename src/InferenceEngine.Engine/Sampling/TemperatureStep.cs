namespace InferenceEngine.Engine.Sampling;

internal sealed class TemperatureStep(float temperature) : ISamplerStep
{
    public void Apply(Span<float> logits)
    {
        for (var i = 0; i < logits.Length; i++)
        {
            logits[i] /= temperature;
        }
    }
}
