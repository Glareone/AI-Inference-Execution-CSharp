using System.Buffers;

namespace InferenceEngine.Engine.Sampling;

internal sealed class TopPStep(float p) : ISamplerStep
{
    public void Apply(Span<float> logits)
    {
        if (p <= 0f || p >= 1f)
        {
            return;
        }

        var length = logits.Length;
        var keysBuffer = ArrayPool<float>.Shared.Rent(length);
        var indicesBuffer = ArrayPool<int>.Shared.Rent(length);
        var probsBuffer = ArrayPool<float>.Shared.Rent(length);
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

            var maxLogit = logits[indices[0]];
            var probs = probsBuffer.AsSpan(0, length);
            var expSum = 0f;
            for (var i = 0; i < length; i++)
            {
                probs[i] = MathF.Exp(logits[indices[i]] - maxLogit);
                expSum += probs[i];
            }

            var cumulative = 0f;
            var cutoff = length;
            for (var i = 0; i < length; i++)
            {
                cumulative += probs[i] / expSum;
                if (cumulative >= p)
                {
                    cutoff = i + 1;
                    break;
                }
            }

            for (var i = cutoff; i < length; i++)
            {
                logits[indices[i]] = float.NegativeInfinity;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(keysBuffer);
            ArrayPool<int>.Shared.Return(indicesBuffer);
            ArrayPool<float>.Shared.Return(probsBuffer);
        }
    }
}
