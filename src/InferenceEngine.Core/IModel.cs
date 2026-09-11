namespace InferenceEngine.Core;

/// <summary>
/// A loaded model capable of running a single-token forward pass. Owned and produced by
/// <c>InferenceEngine.Models</c>; consumed by <c>InferenceEngine.Engine</c>, which never
/// touches tensor internals beyond this contract.
/// </summary>
public interface IModel
{
    ModelConfig Config { get; }

    /// <summary>
    /// Runs the model forward for a single token at the given sequence position, updating
    /// <paramref name="kvCache"/> in place.
    /// </summary>
    /// <param name="tokenId">The input token id.</param>
    /// <param name="position">The token's position in the sequence (0-based).</param>
    /// <param name="kvCache">The KV-cache to read prior context from and write this step into.</param>
    /// <param name="needLogits">
    /// When <c>false</c>, the LM head projection is skipped and an empty span is returned —
    /// used while prefilling every prompt token except the last, where only the KV-cache
    /// update matters.
    /// </param>
    /// <returns>Logits over the vocabulary (length <see cref="ModelConfig.VocabSize"/>), or
    /// empty when <paramref name="needLogits"/> is <c>false</c>.</returns>
    ReadOnlySpan<float> Forward(int tokenId, int position, IKvCache kvCache, bool needLogits);
}
