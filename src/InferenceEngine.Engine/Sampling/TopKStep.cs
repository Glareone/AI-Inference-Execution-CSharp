using System.Buffers;

namespace InferenceEngine.Engine.Sampling;

internal sealed class TopKStep(int k) : ISamplerStep
{
    public void Apply(Span<float> logits)
    {
        if (k <= 0 || k >= logits.Length)
        {
            return;
        }

        var length = logits.Length;
        var keysBuffer = ArrayPool<float>.Shared.Rent(length);
        var indicesBuffer = ArrayPool<int>.Shared.Rent(length);
        try
        {
            var negatedKeys = keysBuffer.AsSpan(0, length);
            var indices = indicesBuffer.AsSpan(0, length);
            for (var i = 0; i < length; i++)
            {
                // Sort ascending on the negated value == descending on the original logit —
                // avoids a captured comparison delegate (Span.Sort dispatches via the
                // IComparable<float> the framework already gives float, no boxing).
                negatedKeys[i] = -logits[i];
                indices[i] = i;
            }

            negatedKeys.Sort(indices);

            for (var i = k; i < length; i++)
            {
                logits[indices[i]] = float.NegativeInfinity;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(keysBuffer);
            ArrayPool<int>.Shared.Return(indicesBuffer);
        }
    }
}
