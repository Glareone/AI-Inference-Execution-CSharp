using InferenceEngine.Engine.Prompting;

namespace InferenceEngine.Engine.Tests.Prompting;

/// <summary>
/// Business case: a chat-tuned model expects its prompt assembled in an exact turn order
/// (system, user, assistant-priming) using the model's own special tokens — get the order or the
/// special tokens wrong and the model responds to garbage it was never fine-tuned to parse. This
/// exercises <see cref="ChatMlTemplate"/> together with its collaborators
/// (<see cref="ChatPromptBuilder"/>, the <see cref="IChatPromptStep"/> chain) as the single unit a
/// caller actually depends on, using a hand-written fake <see cref="Core.ITokenizer"/>.
/// </summary>
public class ChatMlTemplateTests
{
    private static readonly Dictionary<string, int> SpecialTokens = new()
    {
        ["<|im_start|>"] = 1,
        ["<|im_end|>"] = 2,
    };

    [Fact]
    public void Build_AssemblesSystemThenUserThenAssistantTurns_UsingModelSpecialTokens()
    {
        var tokenizer = new FakeTokenizer(SpecialTokens);
        var template = new ChatMlTemplate("You are helpful.");

        var ids = template.Build(tokenizer, "Hi there");

        List<int> expected =
        [
            1, // <|im_start|>
            .. Encode("system\n"),
            .. Encode("You are helpful."),
            2, // <|im_end|>
            .. Encode("\n"),
            1, // <|im_start|>
            .. Encode("user\n"),
            .. Encode("Hi there"),
            2, // <|im_end|>
            .. Encode("\n"),
            1, // <|im_start|>
            .. Encode("assistant\n"),
        ];

        Assert.Equal(expected, ids);
    }

    [Fact]
    public void Build_WithMissingSpecialTokenInVocabulary_ThrowsKeyNotFoundException()
    {
        // The fake tokenizer's vocabulary has no special tokens at all — building the prompt must
        // fail loudly rather than silently drop the turn markers and produce a malformed prompt.
        var tokenizer = new FakeTokenizer(new Dictionary<string, int>());
        var template = new ChatMlTemplate("You are helpful.");

        Assert.Throws<KeyNotFoundException>(() => template.Build(tokenizer, "Hi there"));
    }

    private static IEnumerable<int> Encode(string text) => text.Select(c => (int)c);
}
