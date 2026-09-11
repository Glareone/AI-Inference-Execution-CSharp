using System.Text;

namespace InferenceEngine.Engine;

/// <summary>
/// Decodes UTF-8 bytes arriving one token at a time, buffering an incomplete multi-byte
/// character across calls instead of emitting a replacement character for each half. A single
/// BPE token can represent only part of a multi-byte character, so decoding each token's bytes
/// independently (as <see cref="Core.ITokenizer.DecodeToken"/> does) corrupts streamed text at
/// character boundaries that happen to fall between two tokens.
/// </summary>
/// <remarks>Not thread-safe; use one instance per generation stream.</remarks>
internal sealed class IncrementalUtf8Decoder
{
    private readonly Decoder _decoder = Encoding.UTF8.GetDecoder();

    /// <summary>
    /// Decodes as much of <paramref name="utf8Bytes"/> as forms complete characters, given
    /// whatever incomplete bytes are already buffered from a previous call. Returns an empty
    /// string when the new bytes only complete part of a character — the rest is still pending.
    /// Never flushes: a token that leaves a genuinely incomplete tail at the very end of a
    /// stream is silently dropped rather than replaced with a placeholder character.
    /// </summary>
    public string DecodeNext(byte[] utf8Bytes)
    {
        var chars = new char[utf8Bytes.Length]; // decoded char count never exceeds the byte count
        var charCount = _decoder.GetChars(utf8Bytes, 0, utf8Bytes.Length, chars, 0, flush: false);
        return new string(chars, 0, charCount);
    }
}
