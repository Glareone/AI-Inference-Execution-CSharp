using InferenceEngine.Core;
using InferenceEngine.Engine.Config;
using InferenceEngine.Engine.Prompting;
using InferenceEngine.Engine.Tests.Prompting;

namespace InferenceEngine.Engine.Tests;

/// <summary>
/// Business case: bad generation input — a negative token budget, a prompt that encodes to zero
/// tokens, or a requested sequence longer than the model's context — must be rejected before any
/// expensive or crash-prone work (KV-cache allocation, prefill) begins, and rejected as soon as
/// <see cref="InferenceSession.Generate"/> is called rather than only once the caller starts
/// enumerating the result (a classic C# iterator-method pitfall: code before a <c>yield</c>
/// doesn't run until the first <c>MoveNext</c>).
/// </summary>
public class InferenceSessionValidationTests
{
    private static InferenceSession CreateSession(int maxSeqLen = 2048)
    {
        var config = new ModelConfig(
            Architecture: "llama",
            VocabSize: 8,
            HiddenSize: 4,
            NumLayers: 1,
            NumAttentionHeads: 1,
            NumKvHeads: 1,
            HeadDim: 4,
            FfnHiddenSize: 4,
            MaxSeqLen: maxSeqLen,
            RmsNormEps: 1e-5f,
            RopeFreqBase: 10000f);

        var model = new FakeModel(config);
        var tokenizer = new FakeTokenizer(new Dictionary<string, int>
        {
            ["<|im_start|>"] = 1,
            ["<|im_end|>"] = 2,
        });
        var chatTemplate = new ChatMlTemplate("system prompt");

        return new InferenceSession(model, tokenizer, chatTemplate);
    }

    [Fact]
    public void Generate_NegativeMaxNewTokens_ThrowsImmediatelyWithoutEnumerating()
    {
        var session = CreateSession();

        var exception = Record.Exception(() => session.Generate("hi", new GenerationOptions(MaxNewTokens: -1)));

        Assert.IsType<ArgumentException>(exception);
    }

    [Fact]
    public void Generate_RawEmptyPrompt_ThrowsImmediatelyWithoutEnumerating()
    {
        var session = CreateSession();

        var exception = Record.Exception(() => session.Generate(string.Empty, new GenerationOptions(Raw: true)));

        Assert.IsType<ArgumentException>(exception);
    }

    [Fact]
    public void Generate_SequenceLongerThanModelMaxContext_ThrowsImmediatelyWithoutEnumerating()
    {
        var session = CreateSession(maxSeqLen: 5);

        var exception = Record.Exception(() => session.Generate("hi", new GenerationOptions(MaxNewTokens: 100)));

        Assert.IsType<ArgumentException>(exception);
    }

    [Fact]
    public void Generate_ValidInput_DoesNotThrow()
    {
        var session = CreateSession();

        var exception = Record.Exception(() => session.Generate("hi", new GenerationOptions(MaxNewTokens: 0)));

        Assert.Null(exception);
    }

    [Fact]
    public void Generate_MaxNewTokensNearIntMax_ThrowsWithoutIntegerOverflow()
    {
        // promptIds.Count + int.MaxValue would overflow a naive addition-based bound check and
        // could wrap past the limit instead of failing it — asserting the exception type (rather
        // than just "it throws") catches a regression back to that unchecked comparison.
        var session = CreateSession(maxSeqLen: 10);

        var exception = Record.Exception(() => session.Generate("hi", new GenerationOptions(MaxNewTokens: int.MaxValue)));

        Assert.IsType<ArgumentException>(exception);
    }

    [Fact]
    public void Generate_AtMaxNewTokens_DoesNotRunAWastedFinalForwardPass()
    {
        // The step that yields the MaxNewTokens-th token is also the last step: no further
        // sampling will ever read a forward pass run after it, so Generate must not run one.
        var config = new ModelConfig(
            Architecture: "llama", VocabSize: 8, HiddenSize: 4, NumLayers: 1,
            NumAttentionHeads: 1, NumKvHeads: 1, HeadDim: 4, FfnHiddenSize: 4,
            MaxSeqLen: 2048, RmsNormEps: 1e-5f, RopeFreqBase: 10000f);

        // Token id 5 is neither 0 (FakeTokenizer's hardcoded EosTokenId) nor out of the 8-slot
        // vocab, so sampling reaches the "yield a real token" path instead of stopping at EOS.
        var model = new FakeModel(config, preferredTokenId: 5);
        var tokenizer = new FakeTokenizer(new Dictionary<string, int> { ["<|im_start|>"] = 1, ["<|im_end|>"] = 2 });
        var session = new InferenceSession(model, tokenizer, new ChatMlTemplate("system prompt"));
        var promptTokenCount = session.Tokenize("hi", raw: false).Count;

        var generated = session.Generate("hi", new GenerationOptions(MaxNewTokens: 1)).ToList();

        Assert.Single(generated);
        Assert.Equal(promptTokenCount, model.ForwardCallCount); // prefill only — no forward pass for the step that never samples again
    }

    [Fact]
    public void PrefillTopLogits_PromptLongerThanMaxContext_ThrowsBeforeCacheAllocation()
    {
        // Without this check, LlamaModel.Forward's fixed-size scratch buffers (sized to
        // MaxSeqLen) would be exceeded mid-prefill, throwing an unattributed exception far from
        // the actual cause (an over-long prompt).
        var session = CreateSession(maxSeqLen: 2);

        var exception = Record.Exception(() => session.PrefillTopLogits([1, 2, 3, 4], topN: 5));

        Assert.IsType<ArgumentException>(exception);
    }

    [Fact]
    public void PrefillTopLogits_PromptWithinMaxContext_DoesNotThrow()
    {
        var session = CreateSession(maxSeqLen: 10);

        var exception = Record.Exception(() => session.PrefillTopLogits([1, 2, 3], topN: 5));

        Assert.Null(exception);
    }

    /// <summary>
    /// Business case: the model's vocab size (<c>llama.vocab_size</c>) and the tokenizer's
    /// vocabulary (<c>tokenizer.ggml.tokens</c> length) are read independently from the same
    /// GGUF file. If a malformed or mismatched file has them disagree, a sampled token id could
    /// be out of range for the tokenizer — this must be caught at load time with a clear cause,
    /// not as an obscure IndexOutOfRangeException mid-generation.
    /// </summary>
    [Fact]
    public void ValidateVocabSizesMatch_MismatchedSizes_ThrowsInvalidDataException()
    {
        var exception = Record.Exception(() => InferenceSession.ValidateVocabSizesMatch(modelVocabSize: 8, tokenizerVocabCount: 7));

        Assert.IsType<InvalidDataException>(exception);
    }

    [Fact]
    public void ValidateVocabSizesMatch_MatchingSizes_DoesNotThrow()
    {
        var exception = Record.Exception(() => InferenceSession.ValidateVocabSizesMatch(modelVocabSize: 8, tokenizerVocabCount: 8));

        Assert.Null(exception);
    }
}
