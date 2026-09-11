using InferenceEngine.Core;

namespace InferenceEngine.Engine.Prompting;

/// <summary>
/// Hard-coded ChatML wrapping matching SmolLM2's <c>tokenizer.chat_template</c> metadata (a
/// Jinja2 template this POC doesn't evaluate — see the POC plan's "explicitly mocked" list).
/// The predefined system text and turn structure live here, as a chain of
/// <see cref="IChatPromptStep"/>s, rather than inline in <see cref="InferenceSession"/> — so
/// swapping to a different model family's template later is a new chain, not a rewrite of the
/// session facade.
/// </summary>
internal sealed class ChatMlTemplate(string systemPrompt)
{
    public const string DefaultSystemPrompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

    public List<int> Build(ITokenizer tokenizer, string userPrompt)
    {
        var builder = new ChatPromptBuilder(tokenizer);
        IChatPromptStep[] chain =
        [
            new SystemMessageStep(systemPrompt),
            new UserMessageStep(userPrompt),
            new AssistantPrimingStep(),
        ];

        foreach (var step in chain)
        {
            step.Apply(builder);
        }

        return builder.Build();
    }
}
