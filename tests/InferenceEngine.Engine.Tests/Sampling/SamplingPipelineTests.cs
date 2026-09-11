using InferenceEngine.Engine.Config;
using InferenceEngine.Engine.Sampling;

namespace InferenceEngine.Engine.Tests.Sampling;

/// <summary>
/// Business case: <see cref="SamplingPipeline"/> is the temperature -> top-k -> top-p chain that
/// turns raw logits into a chosen next token. Greedy decoding must be deterministic, sampling
/// must be reproducible given the same seed, and the pipeline must never mutate the caller's
/// logits buffer (callers may reuse it, e.g. to log the original values afterward).
/// </summary>
public class SamplingPipelineTests
{
    [Fact]
    public void Sample_WithZeroTemperature_AlwaysPicksMostLikelyToken()
    {
        var options = new GenerationOptions(Temperature: 0f);
        var pipeline = new SamplingPipeline(options, vocabSize: 4);
        float[] logits = [1f, 5f, 3f, -2f];

        var chosen = pipeline.Sample(logits);

        Assert.Equal(1, chosen);
    }

    [Fact]
    public void Sample_WithSameSeedAndLogits_IsReproducible()
    {
        var options = new GenerationOptions(Temperature: 1f, Seed: 42);
        float[] logits = [1f, 2f, 3f, 4f, 5f];

        var first = new SamplingPipeline(options, vocabSize: 5).Sample(logits);
        var second = new SamplingPipeline(options, vocabSize: 5).Sample(logits);

        Assert.Equal(first, second);
    }

    [Fact]
    public void Sample_DoesNotMutateCallersLogitsSpan()
    {
        var options = new GenerationOptions(Temperature: 0.7f, TopK: 2, TopP: 0.9f, Seed: 1);
        var pipeline = new SamplingPipeline(options, vocabSize: 4);
        float[] logits = [1f, 2f, 3f, 4f];
        float[] original = (float[])logits.Clone();

        pipeline.Sample(logits);

        Assert.Equal(original, logits);
    }
}
