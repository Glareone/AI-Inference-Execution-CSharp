namespace InferenceEngine.Engine.Tests;

/// <summary>
/// <see cref="SimpleKvCache"/> business case: causal attention over history is only correct if
/// every (layer, position) slot's key/value data is independent of every other — writing to one
/// slot must never leak into or overwrite another.
/// </summary>
public class SimpleKvCacheTests
{
    [Fact]
    public void WrittenKeyAndValue_AreReadBackUnchanged()
    {
        var cache = new SimpleKvCache(numLayers: 2, length: 4, kvDim: 3);

        var keySlot = cache.KeySlot(layer: 1, position: 2);
        keySlot[0] = 1f;
        keySlot[1] = 2f;
        keySlot[2] = 3f;

        var valueSlot = cache.ValueSlot(layer: 1, position: 2);
        valueSlot[0] = 10f;
        valueSlot[1] = 20f;
        valueSlot[2] = 30f;

        Assert.Equal([1f, 2f, 3f], cache.Key(1, 2).ToArray());
        Assert.Equal([10f, 20f, 30f], cache.Value(1, 2).ToArray());
    }

    [Fact]
    public void DifferentLayersAndPositions_DoNotOverlap()
    {
        var cache = new SimpleKvCache(numLayers: 2, length: 2, kvDim: 2);

        cache.KeySlot(0, 0)[0] = 1f;
        cache.KeySlot(0, 0)[1] = 2f;
        cache.KeySlot(0, 1)[0] = 3f;
        cache.KeySlot(0, 1)[1] = 4f;
        cache.KeySlot(1, 0)[0] = 5f;
        cache.KeySlot(1, 0)[1] = 6f;
        cache.KeySlot(1, 1)[0] = 7f;
        cache.KeySlot(1, 1)[1] = 8f;

        Assert.Equal([1f, 2f], cache.Key(0, 0).ToArray());
        Assert.Equal([3f, 4f], cache.Key(0, 1).ToArray());
        Assert.Equal([5f, 6f], cache.Key(1, 0).ToArray());
        Assert.Equal([7f, 8f], cache.Key(1, 1).ToArray());
    }
}
