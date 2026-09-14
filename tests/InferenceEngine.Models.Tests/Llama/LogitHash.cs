namespace InferenceEngine.Models.Tests.Llama;

/// <summary>
/// Hashes raw bit patterns, not decimal text, because two floats that print identically can
/// still differ in their low mantissa bits — the whole point is catching numeric drift a
/// KV-cache rewrite must not introduce. A reusable static method, not inlined in one test,
/// because a later end-to-end golden test needs to compute the exact same hash to compare against.
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

    /// <summary>
    /// One FNV-1a mixing step: XOR the byte into the hash, then multiply by the FNV prime — in
    /// that order. The order is what makes this FNV-1a rather than the original FNV-1 (which
    /// multiplies first and XORs after); the two produce different, non-interchangeable hashes,
    /// so swapping the order here would silently invalidate every hash already on record
    /// (<see cref="GoldenLogitBaselineTests"/>'s hardcoded constants).
    /// </summary>
    private static ulong HashByte(ulong hash, byte b)
    {
        hash ^= b;
        hash *= Fnv64Prime;
        return hash;
    }
}
