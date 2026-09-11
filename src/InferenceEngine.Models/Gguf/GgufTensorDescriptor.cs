namespace InferenceEngine.Models.Gguf;

/// <summary>
/// One tensor's location and shape within a GGUF file. <see cref="Dims"/> is in GGUF's
/// <c>ne[]</c> order — <c>Dims[0]</c> is the fastest-varying (contiguous row) dimension.
/// </summary>
internal sealed record GgufTensorDescriptor(string Name, long[] Dims, GgmlType Type, long Offset)
{
    // checked: dims come straight from the file, so a malformed one must overflow loudly here
    // rather than silently wrap into a smaller, plausible-but-wrong element count.
    public long ElementCount => Dims.Aggregate(1L, (acc, d) => checked(acc * d));
}
