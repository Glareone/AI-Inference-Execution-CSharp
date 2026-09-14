namespace InferenceEngine.Engine.Tests;

/// <summary>
/// <see cref="KvBlockPool"/> business case: it is the physical block allocator every
/// <see cref="PagedKvCache"/> session shares, so its contract has to hold regardless of what
/// logical addressing sits on top of it — ascending allocation order for a fresh pool (so a
/// single sequence's physical bytes land where a contiguous cache would), array retention across
/// Free/Allocate (so Reset/Rollback are allocation-free), non-overlapping (layer, kvHead, slot)
/// offsets, and a clear failure (not a silent wraparound) when the pool is exhausted.
/// </summary>
public class KvBlockPoolTests
{
    [Fact]
    public void Allocate_OnFreshPool_ReturnsAscendingIds()
    {
        var pool = new KvBlockPool(numLayers: 1, numKvHeads: 1, headDim: 4, blockSize: 2, maxBlocks: 3);

        Assert.Equal(0, pool.Allocate());
        Assert.Equal(1, pool.Allocate());
        Assert.Equal(2, pool.Allocate());
    }

    [Fact]
    public void Allocate_AfterExhaustion_ThrowsNamingMaxBlocks()
    {
        var pool = new KvBlockPool(numLayers: 1, numKvHeads: 1, headDim: 4, blockSize: 2, maxBlocks: 2);
        pool.Allocate();
        pool.Allocate();

        var ex = Assert.Throws<InvalidOperationException>(() => pool.Allocate());
        Assert.Contains("2", ex.Message);
    }

    [Fact]
    public void Free_ThenAllocate_ReusesTheFreedId_AndRetainsItsArrays()
    {
        var pool = new KvBlockPool(numLayers: 1, numKvHeads: 1, headDim: 4, blockSize: 2, maxBlocks: 2);
        var first = pool.Allocate();
        pool.KeyStore(first)[0] = 42f;

        pool.Free(first);
        var reused = pool.Allocate();

        Assert.Equal(first, reused);
        Assert.Equal(42f, pool.KeyStore(reused)[0]); // arrays retained, not zeroed, across Free/Allocate
    }

    [Fact]
    public void Offset_DoesNotOverlap_AcrossLayersHeadsOrSlots()
    {
        const int numLayers = 2;
        const int numKvHeads = 2;
        const int headDim = 3;
        const int blockSize = 4;
        var pool = new KvBlockPool(numLayers, numKvHeads, headDim, blockSize, maxBlocks: 1);
        var blockId = pool.Allocate();
        var store = pool.KeyStore(blockId);

        var seenRanges = new List<(int Start, int End)>();
        for (var layer = 0; layer < numLayers; layer++)
        {
            for (var kvHead = 0; kvHead < numKvHeads; kvHead++)
            {
                for (var slot = 0; slot < blockSize; slot++)
                {
                    var start = pool.Offset(layer, kvHead, slot);
                    var end = start + headDim;
                    Assert.True(end <= store.Length, "Offset + headDim must stay within the block's array.");

                    foreach (var (existingStart, existingEnd) in seenRanges)
                    {
                        var overlaps = start < existingEnd && existingStart < end;
                        Assert.False(overlaps, $"(layer={layer}, kvHead={kvHead}, slot={slot}) range [{start},{end}) overlaps an earlier one.");
                    }

                    seenRanges.Add((start, end));
                }
            }
        }
    }
}
