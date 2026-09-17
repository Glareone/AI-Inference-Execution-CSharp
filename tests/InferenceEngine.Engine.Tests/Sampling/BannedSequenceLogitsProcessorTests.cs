using InferenceEngine.Engine.Sampling;
using InferenceEngine.Engine.Tests.Prompting;

namespace InferenceEngine.Engine.Tests.Sampling;

/// <summary>
/// Business case: a caller can forbid literal words/phrases from ever being generated (e.g. a
/// competitor's name). A word that BPE-splits into more than one token only counts as "banned"
/// once every one of its tokens has actually appeared, consecutively, in the order the model
/// generated them — a single-token check would miss most real words. The EOS token must survive
/// every ban, no matter how it's specified, since it's the only guaranteed way generation can
/// end instead of being forced onto an arbitrary (possibly still-banned) token.
/// </summary>
public class BannedSequenceLogitsProcessorTests
{
    [Fact]
    public void Apply_WithLengthOneBannedSequence_AlwaysMasksRegardlessOfHistory()
    {
        var processor = new BannedSequenceLogitsProcessor(eosTokenId: -1, bannedSequences: [[2]]);

        float[] withEmptyHistory = [1f, 2f, 3f, 4f];
        processor.Apply(withEmptyHistory, []);
        Assert.Equal(float.NegativeInfinity, withEmptyHistory[2]);

        float[] withNonEmptyHistory = [1f, 2f, 3f, 4f];
        processor.Apply(withNonEmptyHistory, [9, 9, 9]);
        Assert.Equal(float.NegativeInfinity, withNonEmptyHistory[2]);
    }

    [Fact]
    public void Apply_WithMultiTokenBannedSequence_MasksOnlyWhenGeneratedTailMatchesPrefix()
    {
        var processor = new BannedSequenceLogitsProcessor(eosTokenId: -1, bannedSequences: [[10, 11, 12]]);

        float[] nonMatchingTail = [0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 5f];
        processor.Apply(nonMatchingTail, [10, 99]);
        Assert.Equal(5f, nonMatchingTail[12]);

        float[] matchingTail = [0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 5f];
        processor.Apply(matchingTail, [5, 10, 11]);
        Assert.Equal(float.NegativeInfinity, matchingTail[12]);
    }

    [Fact]
    public void Apply_WithTwoIndependentBannedSequences_BothApplyWithoutInterference()
    {
        var processor = new BannedSequenceLogitsProcessor(eosTokenId: -1, bannedSequences: [[2], [10, 11, 12]]);
        float[] logits = [1f, 2f, 3f, 4f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 5f];

        processor.Apply(logits, [10, 11]);

        Assert.Equal(float.NegativeInfinity, logits[2]);
        Assert.Equal(float.NegativeInfinity, logits[12]);
        Assert.Equal(1f, logits[0]);
        Assert.Equal(4f, logits[3]);
    }

    [Fact]
    public void Apply_WithBannedSequenceLongerThanGeneratedHistory_DoesNotThrow()
    {
        var processor = new BannedSequenceLogitsProcessor(eosTokenId: -1, bannedSequences: [[1, 2, 3]]);
        float[] logits = [1f, 2f, 3f, 4f];

        processor.Apply(logits, []);

        Assert.Equal([1f, 2f, 3f, 4f], logits);
    }

    [Fact]
    public void Apply_WithLengthOneBannedSequenceEqualToEos_IsNoOp()
    {
        var processor = new BannedSequenceLogitsProcessor(eosTokenId: 1, bannedSequences: [[1]]);
        float[] logits = [1f, 2f, 3f, 4f];

        processor.Apply(logits, []);

        Assert.Equal(2f, logits[1]);
    }

    [Fact]
    public void Apply_WithMultiTokenSequenceCompletingInEos_IsNeverMaskedEvenWhenPrefixMatches()
    {
        var processor = new BannedSequenceLogitsProcessor(eosTokenId: 12, bannedSequences: [[10, 11, 12]]);
        float[] logits = [0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 5f];

        processor.Apply(logits, [10, 11]);

        Assert.Equal(5f, logits[12]);
    }

    [Fact]
    public void FromWords_WordEncodingToSingleEosToken_IsDroppedSoOnlyOtherWordsGetMasked()
    {
        // FakeTokenizer.EosTokenId is 0 and encodes text as one id per UTF-16 character, so "\0"
        // is the one string that encodes to a single EOS-id token — the case FromWords must drop.
        var tokenizer = new FakeTokenizer(new Dictionary<string, int>());
        var processor = BannedSequenceLogitsProcessor.FromWords(tokenizer, ["\0", "A"]);
        float[] logits = new float[128];

        processor.Apply(logits, []);

        Assert.Equal(float.NegativeInfinity, logits['A']);
        Assert.Equal(0f, logits[0]); // EOS itself: never masked, and the dropped word wouldn't have banned it anyway
        Assert.Equal(0f, logits['B']); // nothing else was ever banned
    }
}
