namespace InferenceEngine.Engine;

/// <summary>
/// Owns physical KV-cache block storage for one session, per the paged/head-major kv-cache ADR
/// (<c>docs/architecture/260914-kv-cache.md</c>). A "physical block" is one <c>float[]</c> (K)
/// plus one <c>float[]</c> (V), each shaped <c>[layer][kvHead][slot][headDim]</c> (head-major, all
/// layers), holding <see cref="BlockSize"/> tokens' worth of KV data across every layer at once.
/// Arrays are allocated lazily on first use and then retained across <see cref="Free"/>/
/// <see cref="Allocate"/> cycles — not reallocated — so freeing and re-allocating a block is
/// allocation-free, which is what makes <c>Reset()</c>/<c>Rollback()</c> cheap on the
/// <see cref="PagedKvCache"/> side.
/// </summary>
internal sealed class KvBlockPool
{
    private readonly int _numLayers;
    private readonly int _headDim;
    private readonly int _blockSize;
    private readonly int _maxBlocks;
    private readonly int _headStride;
    private readonly int _layerStride;
    private readonly int _blockLength;

    private readonly float[]?[] _keyStores;
    private readonly float[]?[] _valueStores;
    private readonly Stack<int> _freeIds;

    public KvBlockPool(int numLayers, int numKvHeads, int headDim, int blockSize, int maxBlocks)
    {
        _numLayers = numLayers;
        _headDim = headDim;
        _blockSize = blockSize;
        _maxBlocks = maxBlocks;
        _headStride = blockSize * headDim;
        _layerStride = blockSize * numKvHeads * headDim;
        _blockLength = numLayers * _layerStride;

        _keyStores = new float[maxBlocks][];
        _valueStores = new float[maxBlocks][];

        // Seeded so Allocate() returns ascending ids (0, 1, 2, ...) for a freshly constructed
        // pool — a single sequence's physical block order then matches a contiguous cache's byte
        // order, per the ADR's sizing note.
        _freeIds = new Stack<int>(maxBlocks);
        for (var id = maxBlocks - 1; id >= 0; id--)
        {
            _freeIds.Push(id);
        }
    }

    public int MaxBlocks => _maxBlocks;

    /// <summary>Pops a physical block id off the free list, allocating its backing arrays on first use.</summary>
    public int Allocate()
    {
        if (_freeIds.Count == 0)
        {
            throw new InvalidOperationException(
                $"KV block pool exhausted: all {_maxBlocks} physical blocks (blockSize={_blockSize}) are in use.");
        }

        var blockId = _freeIds.Pop();
        _keyStores[blockId] ??= new float[_blockLength];
        _valueStores[blockId] ??= new float[_blockLength];
        return blockId;
    }

    /// <summary>Returns a physical block id to the free list. Does not zero its contents.</summary>
    public void Free(int blockId) => _freeIds.Push(blockId);

    public float[] KeyStore(int blockId) => _keyStores[blockId]!;

    public float[] ValueStore(int blockId) => _valueStores[blockId]!;

    /// <summary>Offset (in floats) of <c>(layer, kvHead, slot)</c>'s <see cref="HeadDim"/>-wide row within one physical block's K or V array.</summary>
    public int Offset(int layer, int kvHead, int slot) =>
        (layer * _layerStride) + (kvHead * _headStride) + (slot * _headDim);

    public int HeadDim => _headDim;

    public int NumLayers => _numLayers;
}
