namespace InferenceEngine.Models.Gguf;

/// <summary>
/// One tensor's location and shape within a GGUF file. <see cref="Dims"/> is in GGUF's
/// <c>ne[]</c> order — <c>Dims[0]</c> is the fastest-varying (contiguous row) dimension.
/// </summary>
internal sealed record GgufTensorDescriptor(string Name, long[] Dims, GgmlType Type, long Offset)
{
    public long ElementCount => Dims.Aggregate(1L, (acc, d) => acc * d);
}
