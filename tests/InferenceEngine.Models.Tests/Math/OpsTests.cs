using InferenceEngine.Models.Math;

namespace InferenceEngine.Models.Tests.Math;

/// <summary>
/// Business case: the transformer's core math primitives must match their mathematical
/// definitions exactly, since a subtly wrong RMSNorm/RoPE/softmax/SwiGLU would still "run" and
/// produce plausible-looking (but wrong) logits with no exception to catch it. Each test checks
/// a hand-computed expected value rather than mirroring the implementation.
/// </summary>
/// <remarks>
/// <see cref="InferenceEngine.Models.Math.Ops"/> lives in a namespace named <c>Math</c>, which
/// shadows <c>System.Math</c> once imported via <c>using</c>. This file therefore only ever uses
/// <see cref="MathF"/> for scalar math (never the bare, ambiguous <c>Math.*</c>) to keep that
/// unambiguous.
/// </remarks>
public class OpsTests
{
    [Fact]
    public void RmsNorm_MatchesHandComputedValue()
    {
        // meanSquare = (3^2 + 4^2) / 2 = 12.5, invRms = 1/sqrt(12.5), weight is all-ones so the
        // result is just x * invRms.
        ReadOnlySpan<float> x = [3f, 4f];
        ReadOnlySpan<float> weight = [1f, 1f];
        Span<float> destination = new float[2];

        Ops.RmsNorm(x, weight, eps: 0f, destination);

        var invRms = 1f / MathF.Sqrt(12.5f);
        Assert.Equal(3f * invRms, destination[0], 1e-5f);
        Assert.Equal(4f * invRms, destination[1], 1e-5f);
    }

    [Fact]
    public void MatVec_MatchesManualDotProducts()
    {
        // Row-major [outDim=2, inDim=3]: row0 = [1,2,3], row1 = [4,5,6].
        ReadOnlySpan<float> weight = [1f, 2f, 3f, 4f, 5f, 6f];
        ReadOnlySpan<float> x = [1f, 1f, 1f];
        Span<float> destination = new float[2];

        Ops.MatVec(weight, outDim: 2, inDim: 3, x, destination);

        Assert.Equal(6f, destination[0], 1e-5f); // 1+2+3
        Assert.Equal(15f, destination[1], 1e-5f); // 4+5+6
    }

    [Fact]
    public void Rope_AtPositionZero_IsNoOp_RegardlessOfFreqBase()
    {
        // The rotation angle is position * freqBase^k; at position 0 it's exactly zero for any
        // freqBase, so cos=1/sin=0 and the vector must come back untouched.
        float[] original = [1f, 2f, 3f, 4f];

        foreach (var freqBase in new[] { 10000f, 500000f, 1f })
        {
            var vec = (float[])original.Clone();
            Ops.Rope(vec, numHeads: 1, headDim: 4, position: 0, freqBase);

            Assert.Equal(original, vec);
        }
    }

    [Fact]
    public void Rope_AtNonZeroPosition_RotatesEachPairByExpectedAngle()
    {
        // Single head, headDim=2 (one pair): theta = position * freqBase^0 = position.
        float[] vec = [1f, 0f];
        const int position = 1;
        const float freqBase = 10000f;

        Ops.Rope(vec, numHeads: 1, headDim: 2, position, freqBase);

        var theta = position * MathF.Pow(freqBase, 0f);
        Assert.Equal(MathF.Cos(theta), vec[0], 1e-5f);
        Assert.Equal(MathF.Sin(theta), vec[1], 1e-5f);
    }

    [Fact]
    public void Softmax_OutputSumsToOne()
    {
        ReadOnlySpan<float> x = [1f, 2f, 3f];
        Span<float> destination = new float[3];

        Ops.Softmax(x, destination);

        var sum = destination[0] + destination[1] + destination[2];
        Assert.Equal(1f, sum, 1e-5f);
    }

    [Fact]
    public void Softmax_MatchesStandardConstantsFor_1_2_3()
    {
        ReadOnlySpan<float> x = [1f, 2f, 3f];
        Span<float> destination = new float[3];

        Ops.Softmax(x, destination);

        Assert.Equal(0.09003057f, destination[0], 1e-6f);
        Assert.Equal(0.24472847f, destination[1], 1e-6f);
        Assert.Equal(0.66524096f, destination[2], 1e-6f);
    }

    [Fact]
    public void SwiGlu_AtZeroGate_IsZero()
    {
        // Silu(0) = 0 * sigmoid(0) = 0, so SwiGlu(0, up) = 0 regardless of `up`.
        ReadOnlySpan<float> gate = [0f];
        ReadOnlySpan<float> up = [42f];
        Span<float> scratch = new float[1];
        Span<float> destination = new float[1];

        Ops.SwiGlu(gate, up, scratch, destination);

        Assert.Equal(0f, destination[0], 1e-6f);
    }

    [Fact]
    public void SwiGlu_MatchesHandComputedValue()
    {
        // Silu(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ~= 0.7310586; up=1 leaves it unchanged.
        ReadOnlySpan<float> gate = [1f];
        ReadOnlySpan<float> up = [1f];
        Span<float> scratch = new float[1];
        Span<float> destination = new float[1];

        Ops.SwiGlu(gate, up, scratch, destination);

        Assert.Equal(0.7310586f, destination[0], 1e-5f);
    }
}
