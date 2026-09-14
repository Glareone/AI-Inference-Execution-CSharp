using System.IO;
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
    /// <summary>Tokens per physical KV-cache block — see the kv-cache ADR's block-size rationale.</summary>
    private const int KvBlockSize = 32;

    private readonly IModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly ChatMlTemplate _chatTemplate;

    /// <remarks>Internal (not private) so tests can construct a session from fakes without a real GGUF file.</remarks>
    internal InferenceSession(IModel model, ITokenizer tokenizer, ChatMlTemplate chatTemplate)
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
        ValidateVocabSizesMatch(model.Config.VocabSize, tokenizerData.Tokens.Length);

        var tokenizer = GgufBpeTokenizer.Create(tokenizerData);
        var chatTemplate = new ChatMlTemplate(ChatMlTemplate.DefaultSystemPrompt);
        return new InferenceSession(model, tokenizer, chatTemplate);
    }

    /// <summary>
    /// The model's vocab size (<c>llama.vocab_size</c>) and the tokenizer's vocabulary
    /// (<c>tokenizer.ggml.tokens</c>) are read independently from the same GGUF file. If they
    /// disagree, a sampled token id could be out of range for the tokenizer's vocabulary —
    /// caught here, at load time, with a clear cause, instead of an obscure exception mid-generation.
    /// </summary>
    internal static void ValidateVocabSizesMatch(int modelVocabSize, int tokenizerVocabCount)
    {
        if (modelVocabSize != tokenizerVocabCount)
        {
            throw new InvalidDataException(
                $"Model vocab size ({modelVocabSize}) does not match the tokenizer vocabulary " +
                $"({tokenizerVocabCount} tokens) — the GGUF file's llama.vocab_size and " +
                "tokenizer.ggml.tokens metadata disagree.");
        }
    }

    /// <summary>Token ids for <paramref name="prompt"/> — ChatML-wrapped unless <paramref name="raw"/>. For <c>--debug-tokenize</c>.</summary>
    public IReadOnlyList<int> Tokenize(string prompt, bool raw) =>
        raw ? _tokenizer.Encode(prompt) : _chatTemplate.Build(_tokenizer, prompt);

    public string Decode(IEnumerable<int> ids) => _tokenizer.Decode(ids);

    /// <summary>Runs prefill only and returns the top-<paramref name="topN"/> next-token logits. For <c>--debug-logits</c>.</summary>
    public (int Id, string Text, float Logit)[] PrefillTopLogits(IReadOnlyList<int> promptIds, int topN)
    {
        if (promptIds.Count > Config.MaxSeqLen)
        {
            throw new ArgumentException(
                $"Prompt length {promptIds.Count} exceeds the model's max context ({Config.MaxSeqLen}).", nameof(promptIds));
        }

        var kv = CreateKvCache();
        var logits = default(ReadOnlySpan<float>);
        for (var i = 0; i < promptIds.Count; i++)
        {
            kv.Reserve(i);
            logits = _model.Forward(promptIds[i], i, kv, needLogits: i == promptIds.Count - 1);
        }

        var logitsCopy = logits.ToArray(); // capture before the span's backing buffer is reused
        var negatedKeys = logitsCopy.Select(v => -v).ToArray();
        var indices = Enumerable.Range(0, logitsCopy.Length).ToArray();
        Array.Sort(negatedKeys, indices); // ascending on negated == descending on original
        return indices.Take(topN).Select(i => (i, _tokenizer.DecodeToken(i), logitsCopy[i])).ToArray();
    }

    /// <summary>
    /// Validates eagerly (before returning) and generates lazily. Splitting these matters: this
    /// method is not itself an iterator, so a bad <paramref name="options"/> or empty prompt
    /// throws immediately when called, rather than only once the caller starts enumerating.
    /// </summary>
    public IEnumerable<GeneratedToken> Generate(string prompt, GenerationOptions options)
    {
        if (options.MaxNewTokens < 0)
        {
            throw new ArgumentException("MaxNewTokens must be >= 0.", nameof(options));
        }

        var promptIds = Tokenize(prompt, options.Raw);
        if (promptIds.Count == 0)
        {
            throw new ArgumentException("The prompt encoded to zero tokens; nothing to generate from.", nameof(prompt));
        }

        // This is no longer a KV-cache sizing check — PagedKvCache grows lazily up to
        // Config.MaxSeqLen regardless of promptIds.Count/MaxNewTokens, so it costs nothing to
        // let a too-long request past a cache-capacity check. It stays because Config.MaxSeqLen
        // is also the model's trained context window: RoPE's positional encoding (RopeFreqBase)
        // wasn't fit beyond it, and LlamaModel.Forward has its own `position < Config.MaxSeqLen`
        // scratch-buffer bound check that would throw mid-generation instead of before starting.
        // Compared via subtraction, not promptIds.Count + options.MaxNewTokens > Config.MaxSeqLen —
        // that addition can overflow for a large MaxNewTokens (e.g. int.MaxValue) and wrap past
        // the check instead of failing it.
        if (options.MaxNewTokens > Config.MaxSeqLen - promptIds.Count)
        {
            throw new ArgumentException(
                $"Requested sequence length ({promptIds.Count} prompt + {options.MaxNewTokens} new) " +
                $"exceeds the model's max context ({Config.MaxSeqLen}).");
        }

        return GenerateCore(promptIds, options);
    }

    private IEnumerable<GeneratedToken> GenerateCore(IReadOnlyList<int> promptIds, GenerationOptions options)
    {
        var kv = CreateKvCache();
        var sampler = new SamplingPipeline(options, Config.VocabSize);
        var decoder = new IncrementalUtf8Decoder();

        var position = 0;
        var logits = default(ReadOnlySpan<float>);
        for (; position < promptIds.Count; position++)
        {
            kv.Reserve(position);
            logits = _model.Forward(promptIds[position], position, kv, needLogits: position == promptIds.Count - 1);
        }

        for (var step = 0; step < options.MaxNewTokens; step++)
        {
            var nextId = sampler.Sample(logits);
            if (nextId == _tokenizer.EosTokenId)
            {
                yield break;
            }

            yield return new GeneratedToken(nextId, decoder.DecodeNext(_tokenizer.GetTokenBytes(nextId)));

            if (step == options.MaxNewTokens - 1)
            {
                // No further sampling step will run, so no step would ever read the logits (or
                // the cache write) a forward pass here would produce — skip the most expensive
                // op in the loop instead of computing and discarding it.
                yield break;
            }

            kv.Reserve(position);
            logits = _model.Forward(nextId, position, kv, needLogits: true);
            position++;
        }
    }

    /// <summary>
    /// A block pool and <see cref="PagedKvCache"/> sized to the model's full
    /// <see cref="ModelConfig.MaxSeqLen"/>, not the actual prompt/generation length — physical
    /// blocks are allocated lazily by <see cref="PagedKvCache.Reserve"/> as <c>position</c>
    /// advances, so passing the max capacity here costs an <c>int[]</c> block table and a
    /// nullable-array free list, not the KV data itself.
    /// </summary>
    private PagedKvCache CreateKvCache()
    {
        var capacity = Config.MaxSeqLen;
        var maxBlocks = (capacity + KvBlockSize - 1) / KvBlockSize;
        var pool = new KvBlockPool(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, KvBlockSize, maxBlocks);
        return new PagedKvCache(pool, Config.HeadDim, KvBlockSize, capacity);
    }
}
