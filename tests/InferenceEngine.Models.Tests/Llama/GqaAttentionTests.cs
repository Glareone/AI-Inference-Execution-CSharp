using System.Numerics.Tensors;
using InferenceEngine.Core;
using InferenceEngine.Models.Llama;

namespace InferenceEngine.Models.Tests.Llama;

/// <summary>
/// Business case: <see cref="GqaAttention.Attend"/> was extracted from
/// <c>LlamaModel.Forward</c>'s attention block (see <c>docs/architecture/260914-kv-cache.md</c>)
/// and must keep computing exactly the same floating-point values in exactly the same
/// accumulation order as the original — any loop-reorder or layout change in
/// <c>GqaAttention</c> has to keep matching this independent transcription of the original
/// attention math, not just "look equivalent". The reference implementation below is
/// transcribed directly from the pre-extraction <c>LlamaModel.cs</c> attention block (adapted
/// only to address the head-major cache per-KV-head, per-position, rather than a since-deleted
/// per-layer packed row), not by calling <c>GqaAttention</c>, so a bug introduced during
/// extraction can't hide from both sides of the comparison at once.
/// </summary>
public class GqaAttentionTests
{
    private const int NumKvHeads = 3;
    private const int GroupSize = 3;
    private const int NumQHeads = NumKvHeads * GroupSize;
    private const int HeadDim = 64;
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
        var scores = new float[GroupSize * contextLength];
        var probs = new float[GroupSize * contextLength];
        GqaAttention.Attend(
            cache, Layer, position, q, NumQHeads, NumKvHeads, HeadDim, Scale, contextLength,
            scores, probs, actual);

        for (var i = 0; i < QDim; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void Attend_WithScoreStrideLargerThanContextLength_MatchesReference()
    {
        // Production sizes scores/probs to groupSize * MaxSeqLen and passes MaxSeqLen as the
        // stride regardless of the actual (shorter) context length — this test exercises that
        // "stride > contextLength" shape directly, since the per-lane theory test above always
        // used stride == contextLength.
        const int contextLength = 40;
        const int scoreStride = 128;
        var position = contextLength - 1;
        var rng = new Random(42);

        var q = RandomVector(rng, QDim);
        var cache = BuildFakeCache(rng, contextLength);

        var expected = new float[QDim];
        ReferenceAttend(cache, contextLength, q, expected);

        var actual = new float[QDim];
        var scores = new float[GroupSize * scoreStride];
        var probs = new float[GroupSize * scoreStride];
        GqaAttention.Attend(
            cache, Layer, position, q, NumQHeads, NumKvHeads, HeadDim, Scale, scoreStride,
            scores, probs, actual);

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

    private static HeadMajorFakeKvCache BuildFakeCache(Random rng, int contextLength)
    {
        var cache = new HeadMajorFakeKvCache(contextLength, NumKvHeads, HeadDim);
        for (var t = 0; t < contextLength; t++)
        {
            for (var kvh = 0; kvh < NumKvHeads; kvh++)
            {
                var k = RandomVector(rng, HeadDim);
                var v = RandomVector(rng, HeadDim);
                k.CopyTo(cache.KeySlot(Layer, kvh, t));
                v.CopyTo(cache.ValueSlot(Layer, kvh, t));
            }
        }

        return cache;
    }

    /// <summary>
    /// Transcription of the attention block's math (query-head-outermost loop, per-(kvHead,
    /// position) slot accessors) — deliberately NOT calling <see cref="GqaAttention"/>, so this
    /// is an independent oracle.
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
                var kHead = kvCache.KeySlot(Layer, kvh, t);
                scores[t] = TensorPrimitives.Dot(qHead, kHead) * Scale;
            }

            var scoresSpan = scores.AsSpan(0, contextLength);
            var probsSpan = probs.AsSpan(0, contextLength);
            TensorPrimitives.SoftMax(scoresSpan, probsSpan);

            var outHead = attnOut.Slice(qh * HeadDim, HeadDim);
            outHead.Clear();
            for (var t = 0; t < contextLength; t++)
            {
                var vHead = kvCache.ValueSlot(Layer, kvh, t);
                TensorPrimitives.MultiplyAdd(vHead, probsSpan[t], outHead, outHead);
            }
        }
    }

    /// <summary>
    /// Local, minimal <see cref="IKvCache"/> test double matching <c>SimpleKvCache</c>'s
    /// head-major shape (<c>[kvHead][position][headDim]</c> per layer) — this test project can't
    /// see <c>InferenceEngine.Engine</c>'s internal <c>SimpleKvCache</c> type. Block size fixed at
    /// 32 to match the ADR; <see cref="Reserve"/>/<see cref="Reset"/>/<see cref="Rollback"/> are
    /// unused by these tests (GqaAttention never calls them) and implemented minimally.
    /// </summary>
    private sealed class HeadMajorFakeKvCache : IKvCache
    {
        private readonly float[] _keys;
        private readonly float[] _values;
        private readonly int _capacity;

        public int BlockSize => 32;

        public int HeadDim { get; }

        public int Capacity => _capacity;

        public int Length { get; private set; }

        public HeadMajorFakeKvCache(int capacity, int numKvHeads, int headDim)
        {
            _capacity = capacity;
            HeadDim = headDim;
            Length = capacity;
            var headStride = capacity * headDim;
            _keys = new float[numKvHeads * headStride];
            _values = new float[numKvHeads * headStride];
        }

        public void Reserve(int position) => Length = System.Math.Max(Length, position + 1);

        public Span<float> KeySlot(int layer, int kvHead, int position) =>
            _keys.AsSpan(SlotOffset(kvHead, position), HeadDim);

        public Span<float> ValueSlot(int layer, int kvHead, int position) =>
            _values.AsSpan(SlotOffset(kvHead, position), HeadDim);

        public ReadOnlySpan<float> KeyBlockForHead(int layer, int kvHead, int logicalBlock, int count) =>
            _keys.AsSpan(TileOffset(kvHead, logicalBlock), count * HeadDim);

        public ReadOnlySpan<float> ValueBlockForHead(int layer, int kvHead, int logicalBlock, int count) =>
            _values.AsSpan(TileOffset(kvHead, logicalBlock), count * HeadDim);

        public void Reset() => Length = 0;

        public void Rollback(int toPosition) => Length = toPosition;

        private int SlotOffset(int kvHead, int position) => ((kvHead * _capacity) + position) * HeadDim;

        private int TileOffset(int kvHead, int logicalBlock) => SlotOffset(kvHead, logicalBlock * BlockSize);
    }
}
