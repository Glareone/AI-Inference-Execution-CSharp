using System.Text;
using InferenceEngine.Core;

namespace InferenceEngine.Engine.Tests.Prompting;

/// <summary>
/// Hand-written <see cref="ITokenizer"/> test double (per test-writer-runner guidance: this
/// three-ish-method interface doesn't warrant a mocking library). "Encodes" plain text as one id
/// per UTF-16 character, so a test can compute the exact expected id sequence for any text
/// without depending on real BPE vocabulary/merge data.
/// </summary>
internal sealed class FakeTokenizer(IReadOnlyDictionary<string, int> specialTokens) : ITokenizer
{
    public int BosTokenId => 0;

    public int EosTokenId => 0;

    public IReadOnlyList<int> Encode(string text) => text.Select(c => (int)c).ToList();

    public string Decode(IEnumerable<int> ids) => new(ids.Select(id => (char)id).ToArray());

    public string DecodeToken(int id) => ((char)id).ToString();

    public byte[] GetTokenBytes(int id) => Encoding.UTF8.GetBytes(((char)id).ToString());

    public bool TryGetId(string token, out int id) => specialTokens.TryGetValue(token, out id);
}
