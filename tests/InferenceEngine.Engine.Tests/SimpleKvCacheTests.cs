namespace InferenceEngine.Engine.Tests;

/// <summary>
/// <see cref="SimpleKvCache"/> business case: causal attention over history is only correct if
/// every (layer, kvHead, position) slot's key/value data is independent of every other — writing
/// to one slot must never leak into or overwrite another, and a block-tile read must return
/// exactly the slots it claims to, including across a logical block boundary.
/// </summary>
public class SimpleKvCacheTests
{
    [Fact]
    public void WrittenKeyAndValue_AreReadBackUnchanged()
    {
        var cache = new SimpleKvCache(numLayers: 2, capacity: 4, numKvHeads: 2, headDim: 3);

        var keySlot = cache.KeySlot(layer: 1, kvHead: 1, position: 2);
        keySlot[0] = 1f;
        keySlot[1] = 2f;
        keySlot[2] = 3f;

        var valueSlot = cache.ValueSlot(layer: 1, kvHead: 1, position: 2);
        valueSlot[0] = 10f;
        valueSlot[1] = 20f;
        valueSlot[2] = 30f;

        Assert.Equal([1f, 2f, 3f], cache.KeySlot(1, 1, 2).ToArray());
        Assert.Equal([10f, 20f, 30f], cache.ValueSlot(1, 1, 2).ToArray());
    }

    [Fact]
    public void DifferentLayersHeadsAndPositions_DoNotOverlap()
    {
        var cache = new SimpleKvCache(numLayers: 2, capacity: 2, numKvHeads: 2, headDim: 2);

        cache.KeySlot(0, 0, 0)[0] = 1f;
        cache.KeySlot(0, 0, 0)[1] = 2f;
        cache.KeySlot(0, 0, 1)[0] = 3f;
        cache.KeySlot(0, 0, 1)[1] = 4f;
        cache.KeySlot(0, 1, 0)[0] = 5f;
        cache.KeySlot(0, 1, 0)[1] = 6f;
        cache.KeySlot(1, 0, 0)[0] = 7f;
        cache.KeySlot(1, 0, 0)[1] = 8f;

        Assert.Equal([1f, 2f], cache.KeySlot(0, 0, 0).ToArray());
        Assert.Equal([3f, 4f], cache.KeySlot(0, 0, 1).ToArray());
        Assert.Equal([5f, 6f], cache.KeySlot(0, 1, 0).ToArray());
        Assert.Equal([7f, 8f], cache.KeySlot(1, 0, 0).ToArray());
    }

    [Fact]
    public void KeyBlockForHead_ReturnsExactlyTheRequestedTile_AcrossABlockBoundary()
    {
        // blockSize=4, capacity=10 -> two logical blocks for kvHead 0: [0,4) and [4,8).
        var cache = new SimpleKvCache(numLayers: 1, capacity: 10, numKvHeads: 1, headDim: 2, blockSize: 4);
        for (var position = 0; position < 10; position++)
        {
            var slot = cache.KeySlot(0, 0, position);
            slot[0] = position;
            slot[1] = position + 100;
        }

        var firstBlock = cache.KeyBlockForHead(layer: 0, kvHead: 0, logicalBlock: 0, count: 4);
        Assert.Equal([0f, 100f, 1f, 101f, 2f, 102f, 3f, 103f], firstBlock.ToArray());

        var secondBlockPartial = cache.KeyBlockForHead(layer: 0, kvHead: 0, logicalBlock: 1, count: 2);
        Assert.Equal([4f, 104f, 5f, 105f], secondBlockPartial.ToArray());
    }

    [Fact]
    public void Reserve_GrowsLengthToPositionPlusOne_AndRejectsOutOfRangePositions()
    {
        var cache = new SimpleKvCache(numLayers: 1, capacity: 4, numKvHeads: 1, headDim: 2);

        Assert.Equal(0, cache.Length);

        cache.Reserve(2);
        Assert.Equal(3, cache.Length);

        cache.Reserve(0); // reserving an earlier position must not shrink Length
        Assert.Equal(3, cache.Length);

        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Reserve(4));
    }
}
