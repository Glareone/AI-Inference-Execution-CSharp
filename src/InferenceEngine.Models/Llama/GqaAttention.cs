using System.Numerics.Tensors;
using InferenceEngine.Core;
using InferenceEngine.Models.Math;

namespace InferenceEngine.Models.Llama;

/// <summary>
/// Grouped-query attention over the KV cache for a single layer at a single decode position.
/// Extracted verbatim from <c>LlamaModel.Forward</c>'s attention block so it has its own,
/// independently testable surface — see the attention-and-transformer and kv-cache ADRs. This
/// step is a pure relocation (no behavior change); the loop order changes in a later step.
/// </summary>
internal static class GqaAttention
{
    /// <summary>
    /// Computes attention output for every query head into <paramref name="attnOut"/>. <paramref
    /// name="q"/> is <c>numQHeads * headDim</c> floats (already RoPE'd); the KV cache is assumed
    /// already RoPE'd and written for <paramref name="position"/>. <paramref name="scores"/>/
    /// <paramref name="probs"/> are scratch spans at least <c>groupSize * scoreStride</c> floats
    /// long, where <paramref name="scoreStride"/> is at least <c>position + 1</c> — one
    /// contiguous lane per query head sharing a KV head, so all <c>groupSize</c> heads' scores
    /// for one KV-head tile can be computed before moving to the next KV head (see the kv-cache
    /// ADR's loop-reorder rationale: this outer-KV-head-inner-query-heads nesting reads each
    /// KV-head row once and reuses it across the <c>groupSize</c> query heads that share it,
    /// instead of re-reading it once per query head).
    /// </summary>
    public static void Attend(
        IKvCache kvCache,
        int layer,
        int position,
        ReadOnlySpan<float> q,
        int numQHeads,
        int numKvHeads,
        int headDim,
        float scale,
        int scoreStride,
        Span<float> scores,
        Span<float> probs,
        Span<float> attnOut)
    {
        var groupSize = numQHeads / numKvHeads;
        var contextLength = position + 1;

        for (var kvh = 0; kvh < numKvHeads; kvh++)
        {
            var qBase = kvh * groupSize;

            // pass 1: scores for all groupSize query heads sharing this KV head.
            for (var t = 0; t < contextLength; t++)
            {
                var kHead = kvCache.Key(layer, t).Slice(kvh * headDim, headDim);
                for (var g = 0; g < groupSize; g++)
                {
                    var qHead = q.Slice((qBase + g) * headDim, headDim);
                    scores[(g * scoreStride) + t] = TensorPrimitives.Dot(qHead, kHead) * scale;
                }
            }

            for (var g = 0; g < groupSize; g++)
            {
                Ops.Softmax(
                    scores.Slice(g * scoreStride, contextLength),
                    probs.Slice(g * scoreStride, contextLength));
                attnOut.Slice((qBase + g) * headDim, headDim).Clear();
            }

            // pass 2: V-weighted accumulation, t ascending — float addition isn't associative, so
            // this order must match the original query-head-outermost loop exactly.
            for (var t = 0; t < contextLength; t++)
            {
                var vHead = kvCache.Value(layer, t).Slice(kvh * headDim, headDim);
                for (var g = 0; g < groupSize; g++)
                {
                    var outHead = attnOut.Slice((qBase + g) * headDim, headDim);
                    TensorPrimitives.MultiplyAdd(vHead, probs[(g * scoreStride) + t], outHead, outHead);
                }
            }
        }
    }
}
