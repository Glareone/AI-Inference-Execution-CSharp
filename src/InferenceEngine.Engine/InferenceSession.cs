using InferenceEngine.Core;
using InferenceEngine.Engine.Config;
using InferenceEngine.Engine.Prompting;
using InferenceEngine.Engine.Sampling;
using InferenceEngine.Models.Gguf;
using InferenceEngine.Models.Llama;
using InferenceEngine.Tokenizers;

namespace InferenceEngine.Engine;

public sealed record GeneratedToken(int Id, string Text);

/// <summary>
/// The facade the CLI drives: load a model + tokenizer from a single GGUF file, then stream
/// generated tokens for a prompt. Owns the generation loop, sampling, and KV-cache — the only
/// place that knows about both <c>Models</c> and <c>Tokenizers</c> at once, per the
/// solution-layout ADR.
/// </summary>
public sealed class InferenceSession
{
    private readonly IModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly ChatMlTemplate _chatTemplate;

    private InferenceSession(IModel model, ITokenizer tokenizer, ChatMlTemplate chatTemplate)
    {
        _model = model;
        _tokenizer = tokenizer;
        _chatTemplate = chatTemplate;
    }

    public ModelConfig Config => _model.Config;

    public ITokenizer Tokenizer => _tokenizer;

    public static InferenceSession Load(string modelPath)
    {
        var model = LlamaModel.LoadFromGguf(modelPath);
        var tokenizerData = GgufTokenizerReader.Read(modelPath);
        var tokenizer = GgufBpeTokenizer.Create(tokenizerData);
        var chatTemplate = new ChatMlTemplate(ChatMlTemplate.DefaultSystemPrompt);
        return new InferenceSession(model, tokenizer, chatTemplate);
    }

    /// <summary>Token ids for <paramref name="prompt"/> — ChatML-wrapped unless <paramref name="raw"/>. For <c>--debug-tokenize</c>.</summary>
    public IReadOnlyList<int> Tokenize(string prompt, bool raw) =>
        raw ? _tokenizer.Encode(prompt) : _chatTemplate.Build(_tokenizer, prompt);

    public string Decode(IEnumerable<int> ids) => _tokenizer.Decode(ids);

    /// <summary>Runs prefill only and returns the top-<paramref name="topN"/> next-token logits. For <c>--debug-logits</c>.</summary>
    public (int Id, string Text, float Logit)[] PrefillTopLogits(IReadOnlyList<int> promptIds, int topN)
    {
        var kv = new SimpleKvCache(Config.NumLayers, promptIds.Count, Config.NumKvHeads * Config.HeadDim);
        var logits = default(ReadOnlySpan<float>);
        for (var i = 0; i < promptIds.Count; i++)
        {
            logits = _model.Forward(promptIds[i], i, kv, needLogits: i == promptIds.Count - 1);
        }

        var logitsCopy = logits.ToArray(); // capture before the span's backing buffer is reused
        var negatedKeys = logitsCopy.Select(v => -v).ToArray();
        var indices = Enumerable.Range(0, logitsCopy.Length).ToArray();
        Array.Sort(negatedKeys, indices); // ascending on negated == descending on original
        return indices.Take(topN).Select(i => (i, _tokenizer.DecodeToken(i), logitsCopy[i])).ToArray();
    }

    public IEnumerable<GeneratedToken> Generate(string prompt, GenerationOptions options)
    {
        var promptIds = Tokenize(prompt, options.Raw);
        var kv = new SimpleKvCache(Config.NumLayers, promptIds.Count + options.MaxNewTokens, Config.NumKvHeads * Config.HeadDim);
        var sampler = new SamplingPipeline(options, Config.VocabSize);

        var position = 0;
        var logits = default(ReadOnlySpan<float>);
        for (; position < promptIds.Count; position++)
        {
            logits = _model.Forward(promptIds[position], position, kv, needLogits: position == promptIds.Count - 1);
        }

        for (var step = 0; step < options.MaxNewTokens; step++)
        {
            var nextId = sampler.Sample(logits);
            if (nextId == _tokenizer.EosTokenId)
            {
                yield break;
            }

            yield return new GeneratedToken(nextId, _tokenizer.DecodeToken(nextId));

            logits = _model.Forward(nextId, position, kv, needLogits: true);
            position++;
        }
    }
}
