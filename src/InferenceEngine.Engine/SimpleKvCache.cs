using InferenceEngine.Core;

namespace InferenceEngine.Engine;

/// <summary>
/// Head-major (HND) KV-cache, exact-fit allocated to <c>capacity</c> tokens per the kv-cache ADR
/// (<c>docs/architecture/260914-kv-cache.md</c>). This step lands the HND layout and the
/// block-shaped tile accessors <see cref="IKvCache"/> now requires, but keeps one contiguous
/// <c>float[]</c> allocation per layer rather than paging — paging (lazy per-block allocation via
/// a <c>KvBlockPool</c>) is a later step; this type is deleted once <c>PagedKvCache</c> lands.
/// Layout per layer: <c>[kvHead][position][headDim]</c>, so one KV head's data across positions
/// is contiguous (what makes the block tile accessors and the loop-reordered attention possible)
/// at the cost of a position's data across heads no longer being contiguous.
/// </summary>
internal sealed class SimpleKvCache : IKvCache
{
    private readonly float[][] _keys;
    private readonly float[][] _values;
    private readonly int _capacity;

    public int BlockSize { get; }

    public int HeadDim { get; }

    public int Capacity => _capacity;

    public int Length { get; private set; }

    public SimpleKvCache(int numLayers, int capacity, int numKvHeads, int headDim, int blockSize = 32)
    {
        _capacity = capacity;
        HeadDim = headDim;
        BlockSize = blockSize;

        _keys = new float[numLayers][];
        _values = new float[numLayers][];
        var headStride = capacity * headDim;
        for (var layer = 0; layer < numLayers; layer++)
        {
            _keys[layer] = new float[numKvHeads * headStride];
            _values[layer] = new float[numKvHeads * headStride];
        }
    }

    public void Reserve(int position)
    {
        if (position < 0 || position >= _capacity)
        {
            throw new ArgumentOutOfRangeException(
                nameof(position), position, $"Position must be within [0, {_capacity}).");
        }

        if (position + 1 > Length)
        {
            Length = position + 1;
        }
    }

    public Span<float> KeySlot(int layer, int kvHead, int position) =>
        _keys[layer].AsSpan(SlotOffset(kvHead, position), HeadDim);

    public Span<float> ValueSlot(int layer, int kvHead, int position) =>
        _values[layer].AsSpan(SlotOffset(kvHead, position), HeadDim);

    public ReadOnlySpan<float> KeyBlockForHead(int layer, int kvHead, int logicalBlock, int count) =>
        _keys[layer].AsSpan(TileOffset(kvHead, logicalBlock), count * HeadDim);

    public ReadOnlySpan<float> ValueBlockForHead(int layer, int kvHead, int logicalBlock, int count) =>
        _values[layer].AsSpan(TileOffset(kvHead, logicalBlock), count * HeadDim);

    public void Reset() => Length = 0;

    public void Rollback(int toPosition) => Length = toPosition;

    private int SlotOffset(int kvHead, int position) => ((kvHead * _capacity) + position) * HeadDim;

    private int TileOffset(int kvHead, int logicalBlock) => SlotOffset(kvHead, logicalBlock * BlockSize);
}
