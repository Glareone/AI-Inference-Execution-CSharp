namespace InferenceEngine.Core;

/// <summary>
/// Encodes text to token ids and back. Owned by <c>InferenceEngine.Tokenizers</c>;
/// <c>InferenceEngine.Engine</c> calls only these members.
/// </summary>
public interface ITokenizer
{
    IReadOnlyList<int> Encode(string text);

    string Decode(IEnumerable<int> ids);

    /// <summary>Decodes a single token id to its text piece (may be a partial UTF-8 fragment).</summary>
    string DecodeToken(int id);

    /// <summary>
    /// Raw UTF-8 bytes for a single token id, before any decoding — a token may represent only
    /// part of a multi-byte character. For a caller that must accumulate bytes across several
    /// tokens (e.g. streaming generation) before decoding, so a character split across two
    /// tokens doesn't decode to a replacement character for each half.
    /// </summary>
    byte[] GetTokenBytes(int id);

    int BosTokenId { get; }

    /// <summary>The id that ends a generation turn — the generation loop stops when a sampled id equals this.</summary>
    int EosTokenId { get; }

    /// <summary>Looks up a special/control token (e.g. a chat-template marker) by its literal text, without running it through BPE.</summary>
    bool TryGetId(string token, out int id);
}
