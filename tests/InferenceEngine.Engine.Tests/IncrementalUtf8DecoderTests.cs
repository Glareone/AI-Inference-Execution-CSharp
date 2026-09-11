namespace InferenceEngine.Engine.Tests;

/// <summary>
/// Business case: streamed generation decodes one token's bytes at a time, but a multi-byte
/// UTF-8 character can be split across two tokens. The decoder must buffer an incomplete
/// character across calls and only emit it once complete, instead of turning each half into a
/// replacement character.
/// </summary>
public class IncrementalUtf8DecoderTests
{
    [Fact]
    public void DecodeNext_CompleteAsciiBytes_DecodesImmediately()
    {
        var decoder = new IncrementalUtf8Decoder();

        var result = decoder.DecodeNext("hi"u8.ToArray());

        Assert.Equal("hi", result);
    }

    [Fact]
    public void DecodeNext_MultiByteCharacterSplitAcrossTwoTokens_DecodesOnlyOnceComplete()
    {
        // 'é' (U+00E9) is UTF-8 bytes 0xC3 0xA9. Feed them as two separate "tokens".
        var decoder = new IncrementalUtf8Decoder();

        var afterFirstByte = decoder.DecodeNext([0xC3]);
        var afterSecondByte = decoder.DecodeNext([0xA9]);

        Assert.Equal(string.Empty, afterFirstByte);
        Assert.Equal("é", afterSecondByte);
    }

    [Fact]
    public void DecodeNext_CharacterFollowedByMoreAscii_BothDecodeCorrectly()
    {
        var decoder = new IncrementalUtf8Decoder();

        decoder.DecodeNext([0xC3]);
        var completedChar = decoder.DecodeNext([0xA9]);
        var nextAscii = decoder.DecodeNext("!"u8.ToArray());

        Assert.Equal("é", completedChar);
        Assert.Equal("!", nextAscii);
    }
}
