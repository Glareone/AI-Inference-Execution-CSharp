using InferenceEngine.Models.Gguf;

namespace InferenceEngine.Models.Tests.Gguf;

/// <summary>
/// Business case: tensor dimensions come from the (possibly malformed) GGUF file. Multiplying
/// them to get the element count must not silently wrap into a smaller-looking but wrong
/// number — a corrupted or adversarial file with huge declared dimensions must fail loudly
/// instead of producing an incorrectly-sized tensor read.
/// </summary>
public class GgufTensorDescriptorTests
{
    [Fact]
    public void ElementCount_NormalDimensions_MultipliesCorrectly()
    {
        var descriptor = new GgufTensorDescriptor("w", [576, 1536], GgmlType.F16, Offset: 0);

        Assert.Equal(576L * 1536L, descriptor.ElementCount);
    }

    [Fact]
    public void ElementCount_DimensionsThatWouldOverflowLong_ThrowsOverflowException()
    {
        var descriptor = new GgufTensorDescriptor("w", [long.MaxValue, 2], GgmlType.F16, Offset: 0);

        Assert.Throws<OverflowException>(() => descriptor.ElementCount);
    }
}
