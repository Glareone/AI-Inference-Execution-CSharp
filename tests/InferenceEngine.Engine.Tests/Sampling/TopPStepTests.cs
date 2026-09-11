using InferenceEngine.Engine.Sampling;

namespace InferenceEngine.Engine.Tests.Sampling;

/// <summary>
/// Business case: nucleus (top-p) sampling keeps the smallest prefix of sorted-by-probability
/// candidates whose cumulative probability mass reaches <c>p</c>, masking the rest to
/// <see cref="float.NegativeInfinity"/>. Logits are chosen as <c>ln(probability)</c> for a known
/// probability distribution, so the exact cutoff point can be reasoned about directly instead of
/// inferred from the implementation.
/// </summary>
public class TopPStepTests
{
    [Fact]
    public void Apply_KeepsSmallestPrefixReachingCumulativeThreshold()
    {
        // Probabilities (already sorted descending): 0.5, 0.3, 0.15, 0.05 (sums to 1).
        // Cumulative mass: 0.5, 0.8, 0.95, 1.0. With p=0.75, the first two entries alone (0.5,
        // then 0.5+0.3=0.8) already reach the threshold, so only the first two must survive.
        float[] probabilities = [0.5f, 0.3f, 0.15f, 0.05f];
        Span<float> logits = probabilities.Select(p => MathF.Log(p)).ToArray();
        var step = new TopPStep(p: 0.75f);

        step.Apply(logits);

        Assert.False(float.IsNegativeInfinity(logits[0]));
        Assert.False(float.IsNegativeInfinity(logits[1]));
        Assert.True(float.IsNegativeInfinity(logits[2]));
        Assert.True(float.IsNegativeInfinity(logits[3]));
    }
}
