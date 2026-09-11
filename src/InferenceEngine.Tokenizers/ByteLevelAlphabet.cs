namespace InferenceEngine.Tokenizers;

/// <summary>
/// GPT-2's byte-level alphabet: a bijection between the 256 byte values and 256 printable
/// characters, so every byte string can be represented as ordinary text before BPE merging
/// (this is why merge rules can be plain UTF-8 text like <c>"Ġ t"</c> for "space + t").
/// Printable ASCII (<c>'!'..'~'</c>) and Latin-1 punctuation/letters map to themselves;
/// everything else (control bytes, space, DEL) maps into the U+0100+ range.
/// </summary>
internal static class ByteLevelAlphabet
{
    public static readonly char[] ByteToChar = Build();
    public static readonly Dictionary<char, byte> CharToByte = BuildReverse();

    private static char[] Build()
    {
        var printable = new List<int>();
        for (var b = '!'; b <= '~'; b++) printable.Add(b);
        for (var b = 0xA1; b <= 0xAC; b++) printable.Add(b);
        for (var b = 0xAE; b <= 0xFF; b++) printable.Add(b);

        var printableSet = new HashSet<int>(printable);
        var result = new char[256];
        foreach (var b in printable)
        {
            result[b] = (char)b;
        }

        var n = 0;
        for (var b = 0; b < 256; b++)
        {
            if (!printableSet.Contains(b))
            {
                result[b] = (char)(256 + n);
                n++;
            }
        }

        return result;
    }

    private static Dictionary<char, byte> BuildReverse()
    {
        var reverse = new Dictionary<char, byte>(256);
        for (var b = 0; b < 256; b++)
        {
            reverse[ByteToChar[b]] = (byte)b;
        }

        return reverse;
    }
}
