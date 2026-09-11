namespace InferenceEngine.Core.Tests;

/// <summary>
/// <c>InferenceEngine.Core</c> is intentionally thin: data contracts (<see cref="ModelConfig"/>,
/// <see cref="TokenizerData"/>) and interfaces (<see cref="IModel"/>, <see cref="ITokenizer"/>,
/// <see cref="IKvCache"/>) with no logic of their own. The one guarantee worth protecting here is
/// that the data contracts are C# records with value semantics — other layers (e.g. a future model
/// cache keyed by config) may rely on two descriptions of the same model/vocab comparing equal
/// without a custom comparer.
/// </summary>
public class ValueSemanticsTests
{
    [Fact]
    public void ModelConfig_WithSameValues_AreEqual()
    {
        var a = new ModelConfig(
            Architecture: "llama",
            VocabSize: 49152,
            HiddenSize: 576,
            NumLayers: 30,
            NumAttentionHeads: 9,
            NumKvHeads: 3,
            HeadDim: 64,
            FfnHiddenSize: 1536,
            MaxSeqLen: 2048,
            RmsNormEps: 1e-5f,
            RopeFreqBase: 10000f);

        var b = new ModelConfig(
            Architecture: "llama",
            VocabSize: 49152,
            HiddenSize: 576,
            NumLayers: 30,
            NumAttentionHeads: 9,
            NumKvHeads: 3,
            HeadDim: 64,
            FfnHiddenSize: 1536,
            MaxSeqLen: 2048,
            RmsNormEps: 1e-5f,
            RopeFreqBase: 10000f);

        Assert.Equal(a, b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void ModelConfig_WithDifferentValues_AreNotEqual()
    {
        var a = new ModelConfig("llama", 49152, 576, 30, 9, 3, 64, 1536, 2048, 1e-5f, 10000f);
        var b = a with { NumLayers = 31 };

        Assert.NotEqual(a, b);
    }

    [Fact]
    public void TokenizerData_WithSameArrayInstances_AreEqual()
    {
        var tokens = new[] { "<s>", "a", "b" };
        var merges = new[] { "a b" };
        var a = new TokenizerData(tokens, merges, 0, 0, 0, "smollm");
        var b = new TokenizerData(tokens, merges, 0, 0, 0, "smollm");

        Assert.Equal(a, b);
    }

    /// <summary>
    /// A real gap, documented rather than silently "fixed": C#'s synthesized record equality
    /// compares each field with <c>EqualityComparer&lt;T&gt;.Default</c>, and for array types that
    /// is reference equality, not element-wise <c>SequenceEqual</c>. So two
    /// <see cref="TokenizerData"/> instances built from separately-parsed but byte-identical GGUF
    /// metadata (e.g. by two independent loads of the same model) do NOT compare equal today,
    /// unlike <see cref="ModelConfig"/> whose fields are all scalars. A future caching layer that
    /// assumes "same vocab -> equal TokenizerData" would need either a custom
    /// <see cref="IEquatable{T}"/> implementation on <see cref="TokenizerData"/> or a
    /// value-equatable collection type (e.g. <c>ImmutableArray&lt;string&gt;</c>) in place of
    /// <c>string[]</c> — a production-code decision, not a test fix, so it is only surfaced here.
    /// </summary>
    [Fact]
    public void TokenizerData_WithEqualButDistinctArrayInstances_AreNotEqual()
    {
        var a = new TokenizerData(["<s>", "a", "b"], ["a b"], 0, 0, 0, "smollm");
        var b = new TokenizerData(["<s>", "a", "b"], ["a b"], 0, 0, 0, "smollm");

        Assert.NotEqual(a, b);
    }
}
