namespace InferenceEngine.Tokenizers.Tests;

/// <summary>
/// <see cref="ByteLevelAlphabet"/> is the GPT-2 byte&lt;-&gt;printable-char bijection that lets
/// BPE merge rules be expressed as plain text. Business case: every one of the 256 byte values
/// must round-trip through <c>ByteToChar</c> then <c>CharToByte</c> back to itself, or some byte
/// sequences would be silently unrepresentable/corrupted during tokenization.
/// </summary>
public class ByteLevelAlphabetTests
{
    [Fact]
    public void EveryByteValue_RoundTripsThroughCharAndBack()
    {
        for (var b = 0; b < 256; b++)
        {
            var ch = ByteLevelAlphabet.ByteToChar[b];
            var roundTripped = ByteLevelAlphabet.CharToByte[ch];

            Assert.Equal((byte)b, roundTripped);
        }
    }
}
