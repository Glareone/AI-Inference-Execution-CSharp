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
    /// Raw UTF-8 bytes for a token, before decoding — a token can be only part of a multi-byte
    /// character, so a caller accumulating bytes across tokens (streaming) avoids a replacement
    /// character at the split point.
    /// </summary>
    byte[] GetTokenBytes(int id);

    int BosTokenId { get; }

    /// <summary>The id that ends a generation turn — the generation loop stops when a sampled id equals this.</summary>
    int EosTokenId { get; }

    /// <summary>Looks up a special/control token (e.g. a chat-template marker) by its literal text, without running it through BPE.</summary>
    bool TryGetId(string token, out int id);
}
