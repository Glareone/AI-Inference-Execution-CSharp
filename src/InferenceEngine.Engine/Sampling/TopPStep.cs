namespace InferenceEngine.Engine.Sampling;

/// <summary>
/// Nucleus sampling: keeps the smallest set of highest-probability logits whose cumulative
/// probability reaches <c>p</c>, setting the rest to <see cref="float.NegativeInfinity"/>.
/// </summary>
internal sealed class TopPStep(float p) : ISamplerStep
{
    public void Apply(Span<float> logits)
    {
        if (p <= 0f || p >= 1f)
        {
            return;
        }

        // Array.Sort's comparator can't capture a Span<float> (a ref struct), so sort against
        // a plain-array snapshot of the values instead.
        var snapshot = logits.ToArray();
        var indices = new int[logits.Length];
        for (var i = 0; i < indices.Length; i++)
        {
            indices[i] = i;
        }

        Array.Sort(indices, (a, b) => snapshot[b].CompareTo(snapshot[a]));

        var maxLogit = snapshot[indices[0]];
        var probs = new float[logits.Length];
        var expSum = 0f;
        for (var i = 0; i < indices.Length; i++)
        {
            probs[i] = MathF.Exp(snapshot[indices[i]] - maxLogit);
            expSum += probs[i];
        }

        var cumulative = 0f;
        var cutoff = indices.Length;
        for (var i = 0; i < indices.Length; i++)
        {
            cumulative += probs[i] / expSum;
            if (cumulative >= p)
            {
                cutoff = i + 1;
                break;
            }
        }

        for (var i = cutoff; i < indices.Length; i++)
        {
            logits[indices[i]] = float.NegativeInfinity;
        }
    }
}
