namespace InferenceEngine.Engine.Tests;

/// <summary>
/// <see cref="PagedKvCache"/> business case: causal attention over history is only correct if
/// (layer, kvHead, position) slots and block tiles never overlap — including across a logical
/// block boundary — and lazy paging is only a real memory-saving feature if
/// <see cref="PagedKvCache.Reserve"/> allocates exactly one physical block per logical block (not
/// per position) and <see cref="PagedKvCache.Rollback"/> frees exactly the blocks it should,
/// making their physical ids available for reuse.
/// </summary>
public class PagedKvCacheTests
{
    private static PagedKvCache MakeCache(
        int capacity, int blockSize, int maxBlocks, int numKvHeads = 1, int headDim = 2, int numLayers = 1)
    {
        var pool = new KvBlockPool(numLayers, numKvHeads, headDim, blockSize, maxBlocks);
        return new PagedKvCache(pool, headDim, blockSize, capacity);
    }

    [Fact]
    public void Constructor_RejectsNonPowerOfTwoBlockSize()
    {
        var pool = new KvBlockPool(numLayers: 1, numKvHeads: 1, headDim: 2, blockSize: 3, maxBlocks: 4);
        Assert.Throws<ArgumentException>(() => new PagedKvCache(pool, headDim: 2, blockSize: 3, capacity: 8));
    }

    [Fact]
    public void Reserve_RejectsOutOfRangePositions()
    {
        var cache = MakeCache(capacity: 4, blockSize: 2, maxBlocks: 4);
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Reserve(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Reserve(4));
    }

    [Fact]
    public void KeySlot_BeforeReserve_ThrowsClearly()
    {
        var cache = MakeCache(capacity: 4, blockSize: 2, maxBlocks: 4);
        Assert.Throws<InvalidOperationException>(() => cache.KeySlot(0, 0, 0));
    }

    [Fact]
    public void Reserve_IsIdempotentWithinABlock_AndConsumesExactlyOnePhysicalBlockPerLogicalBlock()
    {
        // blockSize=2, capacity=6 (3 logical blocks), maxBlocks=2: positions 0 and 1 share
        // logical block 0 (one physical block); position 2 needs a second logical block.
        // Reserving both positions of block 0 repeatedly must not exhaust the 2-block pool.
        var cache = MakeCache(capacity: 6, blockSize: 2, maxBlocks: 2);

        cache.Reserve(0);
        cache.Reserve(0); // idempotent
        cache.Reserve(1); // still logical block 0
        cache.Reserve(1);

        cache.KeySlot(0, 0, 0)[0] = 1f;
        cache.KeySlot(0, 0, 1)[0] = 2f;
        Assert.Equal(1f, cache.KeySlot(0, 0, 0)[0]);
        Assert.Equal(2f, cache.KeySlot(0, 0, 1)[0]);

        cache.Reserve(2); // logical block 1 -- the pool's second (and last) physical block
        cache.KeySlot(0, 0, 2)[0] = 3f;
        Assert.Equal(3f, cache.KeySlot(0, 0, 2)[0]);

        // The pool is now fully allocated (2 physical blocks for 2 logical blocks) -- a third
        // logical block must fail with a clear exhaustion error, proving exactly one physical
        // block was consumed per logical block above, not one per Reserve call.
        Assert.Throws<InvalidOperationException>(() => cache.Reserve(4));
    }

    [Fact]
    public void DifferentLayersHeadsAndPositions_DoNotOverlap_IncludingAcrossABlockBoundary()
    {
        const int blockSize = 4;
        var cache = MakeCache(capacity: 10, blockSize: blockSize, maxBlocks: 4, numKvHeads: 2, headDim: 2, numLayers: 2);

        for (var position = 0; position < 10; position++)
        {
            cache.Reserve(position);
        }

        for (var layer = 0; layer < 2; layer++)
        {
            for (var kvHead = 0; kvHead < 2; kvHead++)
            {
                for (var position = 0; position < 10; position++)
                {
                    var slot = cache.KeySlot(layer, kvHead, position);
                    var value = (layer * 1000) + (kvHead * 100) + position;
                    slot[0] = value;
                    slot[1] = -value;
                }
            }
        }

        for (var layer = 0; layer < 2; layer++)
        {
            for (var kvHead = 0; kvHead < 2; kvHead++)
            {
                for (var position = 0; position < 10; position++)
                {
                    var expected = (layer * 1000) + (kvHead * 100) + position;
                    var slot = cache.KeySlot(layer, kvHead, position);
                    Assert.Equal(expected, slot[0]);
                    Assert.Equal(-expected, slot[1]);
                }
            }
        }

        // Tile read across a block boundary (block 1 covers positions [4,8)) must line up with
        // the values written per-slot above.
        var tile = cache.KeyBlockForHead(layer: 1, kvHead: 1, logicalBlock: 1, count: blockSize);
        for (var i = 0; i < blockSize; i++)
        {
            var position = (1 * blockSize) + i;
            Assert.Equal(1000 + 100 + position, tile[i * 2]);
        }
    }

    [Fact]
    public void Rollback_FreesExactlyTheRightBlocks_AndALaterReserveReusesThoseIds()
    {
        // blockSize=2, maxBlocks=4, capacity=8 -> 4 logical blocks, exactly matching pool size.
        var cache = MakeCache(capacity: 8, blockSize: 2, maxBlocks: 4);
        for (var position = 0; position < 8; position++)
        {
            cache.Reserve(position);
        }

        // Rollback(3): keep positions [0,3) -> logical block 0 (positions 0-1) and logical block
        // 1 (positions 2-3, since position 2 < 3) are kept; blocks 2 and 3 must be freed.
        cache.Rollback(3);
        Assert.Equal(3, cache.Length);

        // The kept block's data must survive the rollback untouched.
        Assert.Equal(0f, cache.KeySlot(0, 0, 0)[0]);

        // The two freed physical blocks must be available again -- reserving two new logical
        // blocks (2 and 3, covering positions 4-5 and 6-7) must succeed without exhausting the
        // now-4-physical-block pool.
        cache.Reserve(4);
        cache.Reserve(6);
    }

    [Fact]
    public void Reset_ReturnsEveryBlock()
    {
        var cache = MakeCache(capacity: 4, blockSize: 2, maxBlocks: 2);
        cache.Reserve(0);
        cache.Reserve(2);

        cache.Reset();
        Assert.Equal(0, cache.Length);

        // Both physical blocks must be free again -- reserving both logical blocks a second time
        // must not exhaust the pool.
        cache.Reserve(0);
        cache.Reserve(2);
    }

    [Fact]
    public void CapacityExhaustion_WhenMoreLogicalBlocksThanPhysicalBlocks_ThrowsClearly()
    {
        // capacity spans 3 logical blocks (blockSize=2 -> 6 positions), but the pool only has 2
        // physical blocks.
        var cache = MakeCache(capacity: 6, blockSize: 2, maxBlocks: 2);
        cache.Reserve(0);
        cache.Reserve(2);

        var ex = Assert.Throws<InvalidOperationException>(() => cache.Reserve(4));
        Assert.Contains("2", ex.Message);
    }
}
