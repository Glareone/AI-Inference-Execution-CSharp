using System.Numerics;
using InferenceEngine.Core;

namespace InferenceEngine.Engine;

/// <summary>
/// Block-paged KV cache: a logical-block-index -> physical-block-id table over a shared
/// <see cref="KvBlockPool"/>, per the paged/head-major kv-cache ADR
/// (<c>docs/architecture/260914-kv-cache.md</c>). Physical blocks are assigned lazily via
/// <see cref="Reserve"/> as <c>position</c> advances, rather than up front — the whole point of
/// paging over <c>SimpleKvCache</c>'s exact-fit sizing. Single sequence, single block table; no
/// sharing/ref-counting/copy-on-write (see the ADR's Consequences for the named-but-unbuilt seams).
/// </summary>
internal sealed class PagedKvCache : IKvCache
{
    private readonly KvBlockPool _pool;
    private readonly int[] _blockTable; // logical block index -> physical block id, -1 = unassigned
    private readonly int _blockSizeLog2;
    private readonly int _blockSizeMask;

    public int BlockSize { get; }

    public int HeadDim { get; }

    public int Capacity { get; }

    public int Length { get; private set; }

    public PagedKvCache(KvBlockPool pool, int headDim, int blockSize, int capacity)
    {
        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0)
        {
            throw new ArgumentException($"blockSize must be a power of two (got {blockSize}).", nameof(blockSize));
        }

        _pool = pool;
        BlockSize = blockSize;
        HeadDim = headDim;
        Capacity = capacity;
        _blockSizeLog2 = BitOperations.Log2((uint)blockSize);
        _blockSizeMask = blockSize - 1;

        var numLogicalBlocks = (capacity + blockSize - 1) / blockSize;
        _blockTable = new int[numLogicalBlocks];
        Array.Fill(_blockTable, -1);
    }

    public void Reserve(int position)
    {
        if (position < 0 || position >= Capacity)
        {
            throw new ArgumentOutOfRangeException(
                nameof(position), position, $"Position must be within [0, {Capacity}).");
        }

        var logicalBlock = position >> _blockSizeLog2;
        if (_blockTable[logicalBlock] < 0)
        {
            _blockTable[logicalBlock] = _pool.Allocate();
        }

        if (position + 1 > Length)
        {
            Length = position + 1;
        }
    }

    public Span<float> KeySlot(int layer, int kvHead, int position)
    {
        var (blockId, slot) = Locate(position);
        return _pool.KeyStore(blockId).AsSpan(_pool.Offset(layer, kvHead, slot), HeadDim);
    }

    public Span<float> ValueSlot(int layer, int kvHead, int position)
    {
        var (blockId, slot) = Locate(position);
        return _pool.ValueStore(blockId).AsSpan(_pool.Offset(layer, kvHead, slot), HeadDim);
    }

    public ReadOnlySpan<float> KeyBlockForHead(int layer, int kvHead, int logicalBlock, int count)
    {
        var blockId = BlockIdOf(logicalBlock);
        return _pool.KeyStore(blockId).AsSpan(_pool.Offset(layer, kvHead, 0), count * HeadDim);
    }

    public ReadOnlySpan<float> ValueBlockForHead(int layer, int kvHead, int logicalBlock, int count)
    {
        var blockId = BlockIdOf(logicalBlock);
        return _pool.ValueStore(blockId).AsSpan(_pool.Offset(layer, kvHead, 0), count * HeadDim);
    }

    public void Reset() => Rollback(0);

    public void Rollback(int toPosition)
    {
        var keepBlocks = toPosition == 0 ? 0 : ((toPosition - 1) >> _blockSizeLog2) + 1;
        for (var i = keepBlocks; i < _blockTable.Length; i++)
        {
            if (_blockTable[i] >= 0)
            {
                _pool.Free(_blockTable[i]);
                _blockTable[i] = -1;
            }
        }

        Length = toPosition;
    }

    private (int BlockId, int Slot) Locate(int position)
    {
        var logicalBlock = position >> _blockSizeLog2;
        var blockId = BlockIdOf(logicalBlock);
        return (blockId, position & _blockSizeMask);
    }

    private int BlockIdOf(int logicalBlock)
    {
        var blockId = _blockTable[logicalBlock];
        if (blockId < 0)
        {
            throw new InvalidOperationException(
                $"Logical block {logicalBlock} has not been reserved yet — call Reserve(position) before reading or writing it.");
        }

        return blockId;
    }
}
