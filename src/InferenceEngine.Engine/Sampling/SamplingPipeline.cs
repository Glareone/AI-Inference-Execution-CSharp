using InferenceEngine.Engine.Config;

namespace InferenceEngine.Engine.Sampling;

internal sealed class SamplingPipeline
{
    private readonly List<ISamplerStep> _steps = [];
    private readonly IReadOnlyList<ILogitsProcessor> _processors;
    private readonly int _eosTokenId;
    private readonly Random _random;
    private readonly bool _greedy;
    private readonly float[] _scratch;

    public SamplingPipeline(GenerationOptions options, int vocabSize, IReadOnlyList<ILogitsProcessor> processors, int eosTokenId)
    {
        _greedy = options.Temperature <= 0f;
        _random = options.Seed is { } seed ? new Random(seed) : new Random();
        _scratch = new float[vocabSize];
        _processors = processors;
        _eosTokenId = eosTokenId;

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

    /// <summary>
    /// Runs every <see cref="ILogitsProcessor"/> unconditionally, before the greedy/stochastic
    /// split — <see cref="ISamplerStep"/>'s <c>_steps</c> chain is skipped entirely for greedy
    /// decoding, so a hard constraint (e.g. banning) can only take effect on both paths if it
    /// runs ahead of that split, not inside it. If every logit is <c>-inf</c> after processors
    /// run, returns the EOS token id directly (see the logits-processing ADR's all-masked
    /// fallback) instead of falling into greedy's/softmax's undefined behavior on an all-masked
    /// input.
    /// </summary>
    public int Sample(ReadOnlySpan<float> logits, ReadOnlySpan<int> generatedTokenIds)
    {
        if (_greedy)
        {
            var candidate = logits;
            if (_processors.Count > 0)
            {
                var maskedWorking = _scratch.AsSpan(0, logits.Length);
                logits.CopyTo(maskedWorking);
                foreach (var processor in _processors)
                {
                    processor.Apply(maskedWorking, generatedTokenIds);
                }

                if (AllNegativeInfinity(maskedWorking))
                {
                    return _eosTokenId;
                }

                candidate = maskedWorking;
            }

            var best = 0;
            for (var i = 1; i < candidate.Length; i++)
            {
                if (candidate[i] > candidate[best])
                {
                    best = i;
                }
            }

            return best;
        }

        var working = _scratch.AsSpan(0, logits.Length);
        logits.CopyTo(working);
        foreach (var processor in _processors)
        {
            processor.Apply(working, generatedTokenIds);
        }

        foreach (var step in _steps)
        {
            step.Apply(working);
        }

        if (AllNegativeInfinity(working))
        {
            return _eosTokenId;
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

    private static bool AllNegativeInfinity(ReadOnlySpan<float> values)
    {
        foreach (var v in values)
        {
            if (v != float.NegativeInfinity)
            {
                return false;
            }
        }

        return true;
    }
}
