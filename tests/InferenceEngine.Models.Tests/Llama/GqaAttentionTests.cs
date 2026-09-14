using System.Numerics.Tensors;
using InferenceEngine.Core;
using InferenceEngine.Models.Llama;

namespace InferenceEngine.Models.Tests.Llama;

/// <summary>
/// Business case: <see cref="GqaAttention.Attend"/> was extracted verbatim from
/// <c>LlamaModel.Forward</c>'s attention block (see <c>docs/architecture/260914-kv-cache.md</c>)
/// and must keep computing exactly the same floating-point values in exactly the same
/// accumulation order as the original — any later loop-reorder or layout change in
/// <c>GqaAttention</c> has to keep matching this independent transcription of the original
/// attention math, not just "look equivalent". The reference implementation below is
/// transcribed directly from the pre-extraction <c>LlamaModel.cs</c> attention block, not by
/// calling <c>GqaAttention</c>, so a bug introduced during extraction can't hide from both sides
/// of the comparison at once.
/// </summary>
public class GqaAttentionTests
{
    private const int NumKvHeads = 3;
    private const int GroupSize = 3;
    private const int NumQHeads = NumKvHeads * GroupSize;
    private const int HeadDim = 64;
    private const int KvDim = NumKvHeads * HeadDim;
    private const int QDim = NumQHeads * HeadDim;
    private const int Layer = 0;
    private const float Scale = 0.125f;

    [Theory]
    [InlineData(1)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(96)]
    [InlineData(129)]
    public void Attend_MatchesReferenceTranscription_ForContextLength(int contextLength)
    {
        var position = contextLength - 1;
        var seed = 1000 + contextLength;
        var rng = new Random(seed);

        var q = RandomVector(rng, QDim);
        var cache = BuildFakeCache(rng, contextLength);

        var expected = new float[QDim];
        ReferenceAttend(cache, contextLength, q, expected);

        var actual = new float[QDim];
        var scores = new float[contextLength];
        var probs = new float[contextLength];
        GqaAttention.Attend(
            cache, Layer, position, q, NumQHeads, NumKvHeads, HeadDim, Scale, scores, probs, actual);

        for (var i = 0; i < QDim; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    private static float[] RandomVector(Random rng, int length)
    {
        var vec = new float[length];
        for (var i = 0; i < length; i++)
        {
            vec[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        return vec;
    }

    private static TokenMajorFakeKvCache BuildFakeCache(Random rng, int contextLength)
    {
        var cache = new TokenMajorFakeKvCache(contextLength, KvDim);
        for (var t = 0; t < contextLength; t++)
        {
            var k = RandomVector(rng, KvDim);
            var v = RandomVector(rng, KvDim);
            k.CopyTo(cache.KeySlot(Layer, t));
            v.CopyTo(cache.ValueSlot(Layer, t));
        }

        return cache;
    }

    /// <summary>
    /// Verbatim transcription of the attention block that lived in <c>LlamaModel.Forward</c>
    /// before extraction (query-head-outermost loop, per-position row accessors) — deliberately
    /// NOT calling <see cref="GqaAttention"/>, so this is an independent oracle.
    /// </summary>
    private static void ReferenceAttend(
        IKvCache kvCache, int contextLength, ReadOnlySpan<float> q, Span<float> attnOut)
    {
        var scores = new float[contextLength];
        var probs = new float[contextLength];

        for (var qh = 0; qh < NumQHeads; qh++)
        {
            var kvh = qh / GroupSize;
            var qHead = q.Slice(qh * HeadDim, HeadDim);

            for (var t = 0; t < contextLength; t++)
            {
                var kHead = kvCache.Key(Layer, t).Slice(kvh * HeadDim, HeadDim);
                scores[t] = TensorPrimitives.Dot(qHead, kHead) * Scale;
            }

            var scoresSpan = scores.AsSpan(0, contextLength);
            var probsSpan = probs.AsSpan(0, contextLength);
            TensorPrimitives.SoftMax(scoresSpan, probsSpan);

            var outHead = attnOut.Slice(qh * HeadDim, HeadDim);
            outHead.Clear();
            for (var t = 0; t < contextLength; t++)
            {
                var vHead = kvCache.Value(Layer, t).Slice(kvh * HeadDim, HeadDim);
                TensorPrimitives.MultiplyAdd(vHead, probsSpan[t], outHead, outHead);
            }
        }
    }

    /// <summary>
    /// Local, minimal <see cref="IKvCache"/> test double matching what <c>SimpleKvCache</c> does
    /// today (token-major, one flat array per layer) — this test project can't see
    /// <c>InferenceEngine.Engine</c>'s internal <c>SimpleKvCache</c> type, and this step predates
    /// the paged/head-major cache the later steps introduce.
    /// </summary>
    private sealed class TokenMajorFakeKvCache : IKvCache
    {
        private readonly float[] _keys;
        private readonly float[] _values;
        private readonly int _kvDim;

        public int Length { get; }

        public TokenMajorFakeKvCache(int numTokens, int kvDim)
        {
            Length = numTokens;
            _kvDim = kvDim;
            _keys = new float[numTokens * kvDim];
            _values = new float[numTokens * kvDim];
        }

        public Span<float> KeySlot(int layer, int position) => _keys.AsSpan(position * _kvDim, _kvDim);

        public Span<float> ValueSlot(int layer, int position) => _values.AsSpan(position * _kvDim, _kvDim);

        public ReadOnlySpan<float> Key(int layer, int position) => KeySlot(layer, position);

        public ReadOnlySpan<float> Value(int layer, int position) => ValueSlot(layer, position);
    }
}
