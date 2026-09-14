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
    /// <paramref name="probs"/> are scratch spans at least <c>position + 1</c> floats long.
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
        Span<float> scores,
        Span<float> probs,
        Span<float> attnOut)
    {
        var groupSize = numQHeads / numKvHeads;
        var contextLength = position + 1;

        for (var qh = 0; qh < numQHeads; qh++)
        {
            var kvh = qh / groupSize;
            var qHead = q.Slice(qh * headDim, headDim);

            for (var t = 0; t < contextLength; t++)
            {
                var kHead = kvCache.Key(layer, t).Slice(kvh * headDim, headDim);
                scores[t] = TensorPrimitives.Dot(qHead, kHead) * scale;
            }

            var scoresSlice = scores[..contextLength];
            var probsSlice = probs[..contextLength];
            Ops.Softmax(scoresSlice, probsSlice);

            var outHead = attnOut.Slice(qh * headDim, headDim);
            outHead.Clear();
            for (var t = 0; t < contextLength; t++)
            {
                var vHead = kvCache.Value(layer, t).Slice(kvh * headDim, headDim);
                TensorPrimitives.MultiplyAdd(vHead, probsSlice[t], outHead, outHead);
            }
        }
    }
}
