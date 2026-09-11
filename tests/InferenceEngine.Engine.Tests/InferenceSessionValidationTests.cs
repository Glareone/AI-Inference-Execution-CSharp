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
}
