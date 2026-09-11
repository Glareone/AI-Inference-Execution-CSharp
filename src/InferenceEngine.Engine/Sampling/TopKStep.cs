namespace InferenceEngine.Engine.Sampling;

/// <summary>Keeps only the <c>k</c> highest logits, setting the rest to <see cref="float.NegativeInfinity"/>.</summary>
internal sealed class TopKStep(int k) : ISamplerStep
{
    public void Apply(Span<float> logits)
    {
        if (k <= 0 || k >= logits.Length)
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

        for (var i = k; i < indices.Length; i++)
        {
            logits[indices[i]] = float.NegativeInfinity;
        }
    }
}
