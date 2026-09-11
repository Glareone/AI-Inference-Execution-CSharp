using InferenceEngine.Core;

namespace InferenceEngine.Engine.Prompting;

/// <summary>Accumulates token ids as a chain of <see cref="IChatPromptStep"/>s contributes them.</summary>
internal sealed class ChatPromptBuilder(ITokenizer tokenizer)
{
    private readonly List<int> _ids = [];

    public void AddSpecial(string token)
    {
        if (!tokenizer.TryGetId(token, out var id))
        {
            throw new KeyNotFoundException($"Special token '{token}' is not in the model's vocabulary.");
        }

        _ids.Add(id);
    }

    public void AddText(string text) => _ids.AddRange(tokenizer.Encode(text));

    public List<int> Build() => _ids;
}
