using System.Numerics.Tensors;

namespace InferenceEngine.Models.Math;

/// <summary>
/// Thin, readable wrappers over <see cref="TensorPrimitives"/> for the handful of operations
/// the Llama forward pass needs. Kept separate from <c>LlamaModel</c> so the layer-wiring code
/// (embed -> norm -> attend -> ffn) reads like the architecture, not like SIMD plumbing.
/// </summary>
internal static class Ops
{
    public static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, float eps, Span<float> destination)
    {
        var meanSquare = TensorPrimitives.SumOfSquares(x) / x.Length;
        var invRms = 1f / MathF.Sqrt(meanSquare + eps);
        TensorPrimitives.Multiply(x, invRms, destination);
        TensorPrimitives.Multiply(destination, weight, destination);
    }

    /// <summary>
    /// Multiplies a row-major <c>[outDim, inDim]</c> weight matrix by <paramref name="x"/>.
    /// GGUF stores weights with the input dimension contiguous (<c>ne[0]</c>), which is
    /// exactly this layout — no transpose needed.
    /// </summary>
    public static void MatVec(ReadOnlySpan<float> weight, int outDim, int inDim, ReadOnlySpan<float> x, Span<float> destination)
    {
        for (var o = 0; o < outDim; o++)
        {
            destination[o] = TensorPrimitives.Dot(weight.Slice(o * inDim, inDim), x);
        }
    }

    /// <summary>
    /// Interleaved-pair ("normal" / GPT-J-style) rotary embedding, applied in place to every
    /// head packed into <paramref name="vec"/>. GGUF llama weights are permuted at conversion
    /// time (see <c>convert_hf_to_gguf.py</c>) specifically so this layout — not HF's
    /// half-split rotation — is the one that reproduces the original model's behavior.
    /// </summary>
    public static void Rope(Span<float> vec, int numHeads, int headDim, int position, float freqBase)
    {
        var half = headDim / 2;
        for (var h = 0; h < numHeads; h++)
        {
            var baseIndex = h * headDim;
            for (var i = 0; i < half; i++)
            {
                var theta = position * MathF.Pow(freqBase, -2f * i / headDim);
                var cos = MathF.Cos(theta);
                var sin = MathF.Sin(theta);
                var i0 = baseIndex + (2 * i);
                var i1 = i0 + 1;
                var x0 = vec[i0];
                var x1 = vec[i1];
                vec[i0] = (x0 * cos) - (x1 * sin);
                vec[i1] = (x0 * sin) + (x1 * cos);
            }
        }
    }

    public static void Softmax(ReadOnlySpan<float> x, Span<float> destination) => TensorPrimitives.SoftMax(x, destination);

    /// <summary>SwiGLU: <c>Silu(gate) * up</c>, where <c>Silu(x) = x * sigmoid(x)</c>.</summary>
    public static void SwiGlu(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> sigmoidScratch, Span<float> destination)
    {
        TensorPrimitives.Sigmoid(gate, sigmoidScratch);
        TensorPrimitives.Multiply(gate, sigmoidScratch, destination);
        TensorPrimitives.Multiply(destination, up, destination);
    }
}
