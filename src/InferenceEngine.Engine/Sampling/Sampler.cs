namespace InferenceEngine.Engine.Sampling;

/// <summary>
/// The composable temperature -> top-k -> top-p chain from the sampling-pipeline ADR, with a
/// greedy short-circuit for <see cref="GenerationOptions.Temperature"/> &lt;= 0.
/// </summary>
internal sealed class Sampler
{
    private readonly List<ISamplerStep> _steps = [];
    private readonly Random _random;
    private readonly bool _greedy;
    private readonly float[] _scratch;

    public Sampler(GenerationOptions options, int vocabSize)
    {
        _greedy = options.Temperature <= 0f;
        _random = options.Seed is { } seed ? new Random(seed) : new Random();
        _scratch = new float[vocabSize];

        if (!_greedy)
        {
            _steps.Add(new TemperatureStep(options.Temperature));
            if (options.TopK > 0)
            {
                _steps.Add(new TopKStep(options.TopK));
            }

            if (options.TopP > 0f)
            {
                _steps.Add(new TopPStep(options.TopP));
            }
        }
    }

    public int Sample(ReadOnlySpan<float> logits)
    {
        if (_greedy)
        {
            var best = 0;
            for (var i = 1; i < logits.Length; i++)
            {
                if (logits[i] > logits[best])
                {
                    best = i;
                }
            }

            return best;
        }

        var working = _scratch.AsSpan(0, logits.Length);
        logits.CopyTo(working);
        foreach (var step in _steps)
        {
            step.Apply(working);
        }

        var max = float.NegativeInfinity;
        foreach (var v in working)
        {
            if (v > max)
            {
                max = v;
            }
        }

        var sum = 0f;
        for (var i = 0; i < working.Length; i++)
        {
            working[i] = MathF.Exp(working[i] - max);
            sum += working[i];
        }

        var target = (float)(_random.NextDouble() * sum);
        var cumulative = 0f;
        for (var i = 0; i < working.Length; i++)
        {
            cumulative += working[i];
            if (cumulative >= target)
            {
                return i;
            }
        }

        return working.Length - 1;
    }
}
