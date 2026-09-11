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

    int BosTokenId { get; }

    int EosTokenId { get; }

    bool TryGetId(string token, out int id);
}
