using InferenceEngine.Engine.Config;
using InferenceEngine.Engine.Sampling;

namespace InferenceEngine.Engine.Tests.Sampling;

/// <summary>
/// Business case: <see cref="SamplingPipeline"/> runs two distinct phases before returning a
/// token — <see cref="ILogitsProcessor"/>s (deterministic hard constraints, e.g. banning) run
/// unconditionally first, then, only for non-greedy decoding, the temperature -> top-k -> top-p
/// <see cref="ISamplerStep"/> chain. Greedy decoding must be deterministic, sampling must be
/// reproducible given the same seed, the pipeline must never mutate the caller's logits buffer
/// (callers may reuse it, e.g. to log the original values afterward), and a hard constraint must
/// take effect on the greedy path too — not just the stochastic one, which the pipeline's own
/// <c>_steps</c> chain skips entirely by design.
/// </summary>
public class SamplingPipelineTests
{
    [Fact]
    public void Sample_WithZeroTemperature_AlwaysPicksMostLikelyToken()
    {
        var options = new GenerationOptions(Temperature: 0f);
        var pipeline = new SamplingPipeline(options, vocabSize: 4, processors: [], eosTokenId: -1);
        float[] logits = [1f, 5f, 3f, -2f];

        var chosen = pipeline.Sample(logits, []);

        Assert.Equal(1, chosen);
    }

    [Fact]
    public void Sample_WithSameSeedAndLogits_IsReproducible()
    {
        var options = new GenerationOptions(Temperature: 1f, Seed: 42);
        float[] logits = [1f, 2f, 3f, 4f, 5f];

        var first = new SamplingPipeline(options, vocabSize: 5, processors: [], eosTokenId: -1).Sample(logits, []);
        var second = new SamplingPipeline(options, vocabSize: 5, processors: [], eosTokenId: -1).Sample(logits, []);

        Assert.Equal(first, second);
    }

    [Fact]
    public void Sample_DoesNotMutateCallersLogitsSpan()
    {
        var options = new GenerationOptions(Temperature: 0.7f, TopK: 2, TopP: 0.9f, Seed: 1);
        var pipeline = new SamplingPipeline(options, vocabSize: 4, processors: [], eosTokenId: -1);
        float[] logits = [1f, 2f, 3f, 4f];
        float[] original = (float[])logits.Clone();

        pipeline.Sample(logits, []);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Sample_WithGreedyDecodingAndTopTokenBanned_ReturnsADifferentTokenThanWithoutTheBan()
    {
        // Before this feature, a ban could never reach the greedy path at all — SamplingPipeline
        // only ran _steps (where a ban would have lived) when Temperature > 0, and Temperature: 0
        // (greedy) is the default. This is the regression test for that fix.
        var options = new GenerationOptions(Temperature: 0f);
        float[] logits = [1f, 5f, 3f, 2f];

        var withoutBan = new SamplingPipeline(options, vocabSize: 4, processors: [], eosTokenId: -1).Sample(logits, []);

        var banTopToken = new BannedSequenceLogitsProcessor(eosTokenId: -1, bannedSequences: [[1]]);
        var withBan = new SamplingPipeline(options, vocabSize: 4, processors: [banTopToken], eosTokenId: -1).Sample(logits, []);

        Assert.Equal(1, withoutBan);
        Assert.NotEqual(withoutBan, withBan);
        Assert.Equal(2, withBan); // next-highest surviving logit
    }

    [Fact]
    public void Sample_GreedyWithEveryTokenMaskedAfterProcessors_FallsBackToEos()
    {
        // Every non-EOS token is banned, and the EOS position's own raw logit is already -inf
        // (a constructed edge case) so the fallback — not just "EOS happens to have the highest
        // finite logit" — is what's actually being exercised.
        var options = new GenerationOptions(Temperature: 0f);
        float[] logits = [5f, 4f, float.NegativeInfinity];
        var banEverythingElse = new BannedSequenceLogitsProcessor(eosTokenId: 2, bannedSequences: [[0], [1]]);
        var pipeline = new SamplingPipeline(options, vocabSize: 3, processors: [banEverythingElse], eosTokenId: 2);

        var chosen = pipeline.Sample(logits, []);

        Assert.Equal(2, chosen);
    }

    [Fact]
    public void Sample_StochasticWithEveryTokenMaskedAfterProcessors_FallsBackToEosInsteadOfNaN()
    {
        // Before the all-masked fallback existed, Exp(-inf - -inf) == NaN for every entry, and
        // NaN comparisons are always false, so the cumulative-probability loop fell through to
        // `working.Length - 1` — an arbitrary index, not EOS. This proves that path is gone.
        var options = new GenerationOptions(Temperature: 1f);
        float[] logits = [5f, 4f, float.NegativeInfinity];
        var banEverythingElse = new BannedSequenceLogitsProcessor(eosTokenId: 2, bannedSequences: [[0], [1]]);
        var pipeline = new SamplingPipeline(options, vocabSize: 3, processors: [banEverythingElse], eosTokenId: 2);

        var chosen = pipeline.Sample(logits, []);

        Assert.Equal(2, chosen);
    }
}
