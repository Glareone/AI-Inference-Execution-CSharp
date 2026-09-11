using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using InferenceEngine.Core;

namespace InferenceEngine.Tokenizers;

/// <summary>
/// Hand-rolled byte-level BPE tokenizer (the GPT-2 family; GGUF's <c>tokenizer.ggml.model =
/// "gpt2"</c>) built directly from a model's embedded vocab/merges — see the tokenization ADR
/// for why this is hand-rolled rather than built on a NuGet tokenizer library.
/// </summary>
/// <remarks>
/// GGUF embeds the vocabulary and merge ranks but not the pre-tokenizer's splitting regex
/// (see <c>investigation/huggingface-ecosystem.md</c>). Pre-tokenizer variants are looked up by
/// the <c>tokenizer.ggml.pre</c> name; only the one this POC's test model uses ("smollm",
/// exact pattern confirmed against llama.cpp's <c>unicode.cpp</c>) is implemented.
/// </remarks>
public sealed class GgufBpeTokenizer : ITokenizer
{
    private static readonly IReadOnlyDictionary<string, string[]> PreTokenizerPatternsByName =
        new Dictionary<string, string[]>
        {
            // llama.cpp LLAMA_VOCAB_PRE_TYPE_SMOLLM regex_exprs (src/llama-vocab.cpp): a bare
            // "\p{N}" pass first isolates every digit into its own single-character segment,
            // then the main GPT-2-style pattern splits everything else.
            ["smollm"] =
            [
                @"\p{N}",
                @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            ],
        };

    private readonly string[] _idToToken;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<(string Left, string Right), int> _mergeRanks;
    private readonly Regex[] _preTokenizerPatterns;

    public int BosTokenId { get; }

    public int EosTokenId { get; }

    private GgufBpeTokenizer(TokenizerData data)
    {
        _idToToken = data.Tokens;
        _tokenToId = new Dictionary<string, int>(data.Tokens.Length);
        for (var id = 0; id < data.Tokens.Length; id++)
        {
            _tokenToId[data.Tokens[id]] = id;
        }

        _mergeRanks = new Dictionary<(string, string), int>(data.Merges.Length);
        for (var rank = 0; rank < data.Merges.Length; rank++)
        {
            var parts = data.Merges[rank].Split(' ', 2);
            if (parts.Length != 2)
            {
                throw new InvalidDataException($"GGUF merge entry {rank} ('{data.Merges[rank]}') is not a space-separated pair.");
            }

            _mergeRanks[(parts[0], parts[1])] = rank;
        }

        if (!PreTokenizerPatternsByName.TryGetValue(data.PreTokenizerName, out var patterns))
        {
            throw new NotSupportedException(
                $"GGUF pre-tokenizer '{data.PreTokenizerName}' is not implemented yet (this POC only supports 'smollm').");
        }

        _preTokenizerPatterns = patterns
            .Select(p => new Regex(p, RegexOptions.Compiled | RegexOptions.CultureInvariant))
            .ToArray();

        BosTokenId = data.BosTokenId;
        EosTokenId = data.EosTokenId;
    }

    public static GgufBpeTokenizer Create(TokenizerData data) => new(data);

    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        foreach (var (start, length) in PreTokenize(text))
        {
            var piece = text.Substring(start, length);
            var byteLevelPiece = ByteLevelEncode(piece);
            foreach (var symbol in Bpe(byteLevelPiece))
            {
                if (!_tokenToId.TryGetValue(symbol, out var id))
                {
                    throw new KeyNotFoundException($"BPE produced symbol '{symbol}' (from piece '{piece}') that is not in the vocabulary.");
                }

                ids.Add(id);
            }
        }

        return ids;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var bytes = new List<byte>();
        foreach (var id in ids)
        {
            foreach (var ch in _idToToken[id])
            {
                bytes.Add(ByteLevelAlphabet.CharToByte[ch]);
            }
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    public string DecodeToken(int id) => Encoding.UTF8.GetString(GetTokenBytes(id));

    public byte[] GetTokenBytes(int id)
    {
        var token = _idToToken[id];
        var bytes = new byte[token.Length];
        for (var i = 0; i < token.Length; i++)
        {
            bytes[i] = ByteLevelAlphabet.CharToByte[token[i]];
        }

        return bytes;
    }

    public bool TryGetId(string token, out int id) => _tokenToId.TryGetValue(token, out id);

    /// <summary>
    /// Splits <paramref name="text"/> into pre-tokenizer segments, replicating llama.cpp's
    /// <c>unicode_regex_split</c>: each pattern in turn re-splits every existing segment,
    /// keeping unmatched runs as their own segments so no input character is dropped.
    /// </summary>
    private List<(int Start, int Length)> PreTokenize(string text)
    {
        var segments = new List<(int Start, int Length)> { (0, text.Length) };
        foreach (var pattern in _preTokenizerPatterns)
        {
            segments = SplitOnce(text, pattern, segments);
        }

        return segments;
    }

    private static List<(int Start, int Length)> SplitOnce(string text, Regex pattern, List<(int Start, int Length)> segments)
    {
        var result = new List<(int, int)>();
        foreach (var (start, length) in segments)
        {
            var cursor = 0;
            foreach (var match in pattern.EnumerateMatches(text.AsSpan(start, length)))
            {
                if (match.Index > cursor)
                {
                    result.Add((start + cursor, match.Index - cursor));
                }

                result.Add((start + match.Index, match.Length));
                cursor = match.Index + match.Length;
            }

            if (cursor < length)
            {
                result.Add((start + cursor, length - cursor));
            }
        }

        return result;
    }

    private static string ByteLevelEncode(string piece)
    {
        var utf8 = Encoding.UTF8.GetBytes(piece);
        var chars = new char[utf8.Length];
        for (var i = 0; i < utf8.Length; i++)
        {
            chars[i] = ByteLevelAlphabet.ByteToChar[utf8[i]];
        }

        return new string(chars);
    }

    /// <summary>Greedy lowest-rank-pair merging, the standard BPE encode loop.</summary>
    private List<string> Bpe(string byteLevelPiece)
    {
        var symbols = new List<string>(byteLevelPiece.Length);
        foreach (var ch in byteLevelPiece)
        {
            symbols.Add(ch.ToString());
        }

        while (symbols.Count > 1)
        {
            var bestRank = int.MaxValue;
            var bestIndex = -1;
            for (var i = 0; i < symbols.Count - 1; i++)
            {
                if (_mergeRanks.TryGetValue((symbols[i], symbols[i + 1]), out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
            {
                break;
            }

            symbols[bestIndex] += symbols[bestIndex + 1];
            symbols.RemoveAt(bestIndex + 1);
        }

        return symbols;
    }
}
