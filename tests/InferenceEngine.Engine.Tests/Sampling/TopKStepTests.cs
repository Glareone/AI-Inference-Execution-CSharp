using InferenceEngine.Engine.Sampling;

namespace InferenceEngine.Engine.Tests.Sampling;

/// <summary>
/// Business case: top-k narrows the candidate pool to exactly the <c>k</c> highest-logit tokens,
/// so sampling afterward can only ever pick one of those <c>k</c> — every other logit must become
/// <see cref="float.NegativeInfinity"/> (probability zero after softmax).
/// </summary>
public class TopKStepTests
{
    [Fact]
    public void Apply_KeepsExactlyKHighestLogits_MaskingTheRest()
    {
        var step = new TopKStep(k: 2);
        Span<float> logits = [1f, 5f, 3f, 2f];

        step.Apply(logits);

        // Indices 1 (5) and 2 (3) are the two highest and must survive; 0 and 3 must be masked.
        Assert.True(float.IsNegativeInfinity(logits[0]));
        Assert.Equal(5f, logits[1]);
        Assert.Equal(3f, logits[2]);
        Assert.True(float.IsNegativeInfinity(logits[3]));

        var survivorCount = 0;
        foreach (var v in logits)
        {
            if (!float.IsNegativeInfinity(v))
            {
                survivorCount++;
            }
        }

        Assert.Equal(2, survivorCount);
    }
}
