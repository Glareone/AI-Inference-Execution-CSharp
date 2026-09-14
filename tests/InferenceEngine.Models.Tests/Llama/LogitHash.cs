namespace InferenceEngine.Models.Tests.Llama;

/// <summary>
/// Bit-exact FNV-1a hash of a logits vector, used to freeze a "golden" forward-pass output for
/// later regression comparison. Hashes the raw IEEE-754 bit pattern of every float
/// (<see cref="BitConverter.SingleToInt32Bits(float)"/>), not its decimal text representation —
/// two floats that print identically can still differ in their low mantissa bits, and this hash
/// is deliberately sensitive to that, because the whole point is catching numeric drift a
/// KV-cache rewrite must not introduce. Kept as a small reusable static method (rather than
/// inlined in one test) because a later end-to-end golden test, after the KV-cache rewrite, needs
/// to compute the exact same hash to compare against.
/// </summary>
internal static class LogitHash
{
    private const ulong Fnv64OffsetBasis = 14695981039346656037UL;
    private const ulong Fnv64Prime = 1099511628211UL;

    public static ulong Fnv1a(ReadOnlySpan<float> logits)
    {
        var hash = Fnv64OffsetBasis;
        foreach (var value in logits)
        {
            var bits = BitConverter.SingleToInt32Bits(value);
            hash = HashByte(hash, (byte)bits);
            hash = HashByte(hash, (byte)(bits >> 8));
            hash = HashByte(hash, (byte)(bits >> 16));
            hash = HashByte(hash, (byte)(bits >> 24));
        }

        return hash;
    }

    private static ulong HashByte(ulong hash, byte b)
    {
        hash ^= b;
        hash *= Fnv64Prime;
        return hash;
    }
}
