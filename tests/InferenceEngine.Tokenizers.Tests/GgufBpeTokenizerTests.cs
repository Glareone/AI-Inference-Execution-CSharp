using InferenceEngine.Core;

namespace InferenceEngine.Tokenizers.Tests;

/// <summary>
/// <see cref="GgufBpeTokenizer"/> is a hand-rolled byte-level BPE tokenizer built directly from a
/// model's embedded vocab/merges (see the tokenization ADR for why this is hand-rolled rather
/// than a NuGet tokenizer library). Business case: text must survive an encode/decode round trip,
/// the BPE merge loop must always pick the globally lowest-rank merge rather than the first one
/// it happens to scan, digits must be isolated into their own tokens by the "smollm"
/// pre-tokenizer, special tokens must be reachable by their exact text, and an unrecognized
/// pre-tokenizer variant must fail at construction rather than mis-tokenize silently.
/// </summary>
public class GgufBpeTokenizerTests
{
    private static TokenizerData MakeData(string[] tokens, string[] merges, string preTokenizerName = "smollm") =>
        new(tokens, merges, BosTokenId: 0, EosTokenId: 0, preTokenizerName);

    [Fact]
    public void Encode_ThenDecode_ReturnsOriginalText()
    {
        var tokenizer = GgufBpeTokenizer.Create(MakeData(tokens: ["a", "b"], merges: []));

        var ids = tokenizer.Encode("ab");
        var text = tokenizer.Decode(ids);

        Assert.Equal("ab", text);
    }

    [Fact]
    public void Bpe_AlwaysAppliesLowestRankMerge_NotFirstAdjacentPairFound()
    {
        // Merge ranks: "b c" is rank 0 (highest priority), "a b" is rank 1 — even though the
        // scan visits the (a,b) pair before the (b,c) pair, the lower-rank (b,c) merge must win.
        var tokenizer = GgufBpeTokenizer.Create(MakeData(
            tokens: ["a", "b", "c", "bc"],
            merges: ["b c", "a b"]));

        var ids = tokenizer.Encode("abc");

        // "a" (id 0) stays a single symbol; "b"+"c" merge into "bc" (id 3) — NOT "a"+"b".
        Assert.Equal([0, 3], ids);
    }

    [Fact]
    public void Encode_PreTokenizesDigitsIndividually_NotAsPartOfTheSurroundingLetters()
    {
        // The "smollm" pre-tokenizer's first regex pass isolates every \p{N} character into its
        // own segment before the main letter/punctuation pattern runs, so "a1b" must become three
        // separate pre-tokenizer segments (and therefore three separate BPE calls/tokens), not
        // one segment "a1b" or two segments "a1"/"b".
        var tokenizer = GgufBpeTokenizer.Create(MakeData(
            tokens: ["a", "1", "b"],
            merges: []));

        var ids = tokenizer.Encode("a1b");

        Assert.Equal([0, 1, 2], ids);
    }

    [Fact]
    public void TryGetId_FindsSpecialToken_ByExactText()
    {
        var tokenizer = GgufBpeTokenizer.Create(MakeData(
            tokens: ["<|im_start|>", "<|im_end|>", "hello"],
            merges: []));

        Assert.True(tokenizer.TryGetId("<|im_start|>", out var id));
        Assert.Equal(0, id);
    }

    [Fact]
    public void TryGetId_ForUnknownText_ReturnsFalse()
    {
        var tokenizer = GgufBpeTokenizer.Create(MakeData(
            tokens: ["<|im_start|>", "hello"],
            merges: []));

        Assert.False(tokenizer.TryGetId("<|does_not_exist|>", out _));
    }

    [Fact]
    public void Create_WithUnrecognizedPreTokenizer_ThrowsAtConstruction()
    {
        var data = MakeData(tokens: ["a"], merges: [], preTokenizerName: "some-unimplemented-variant");

        Assert.Throws<NotSupportedException>(() => GgufBpeTokenizer.Create(data));
    }

    [Fact]
    public void Create_WithMalformedMergeEntry_ThrowsInvalidDataExceptionNamingTheEntry()
    {
        // A merge entry is expected to be "left right" (single space). One with no space at all
        // (or more than two space-separated parts) is a malformed/corrupted GGUF file, not a
        // case that should surface as an unexplained IndexOutOfRangeException from Split.
        var data = MakeData(tokens: ["a", "b"], merges: ["ab"]);

        var ex = Assert.Throws<InvalidDataException>(() => GgufBpeTokenizer.Create(data));
        Assert.Contains("ab", ex.Message);
    }
}
