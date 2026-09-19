using InferenceEngine.Core;

namespace InferenceEngine.Engine.Sampling;

/// <summary>
/// Bans literal token sequences from ever being generated — the same sequence-prefix-matching
/// algorithm as HuggingFace's <c>NoBadWordsLogitsProcessor</c>. See the logits-processing ADR
/// (<c>docs/architecture/260917-logits-processing.md</c>) for why sequence matching, not
/// single-token banning, is required to ban a multi-token word.
/// </summary>
internal sealed class BannedSequenceLogitsProcessor(int eosTokenId, IReadOnlyList<int[]> bannedSequences) : ILogitsProcessor
{
    /// <inheritdoc/>
    public void Apply(Span<float> logits, ReadOnlySpan<int> generatedTokenIds)
    {
        foreach (var sequence in bannedSequences)
        {
            if (sequence.Length == 1)
            {
                if (sequence[0] != eosTokenId)
                {
                    logits[sequence[0]] = float.NegativeInfinity;
                }

                continue;
            }

            var prefixLength = sequence.Length - 1;
            if (generatedTokenIds.Length < prefixLength)
            {
                continue;
            }

            var tail = generatedTokenIds[^prefixLength..];
            if (!tail.SequenceEqual(sequence.AsSpan(0, prefixLength)))
            {
                continue;
            }

            var completingToken = sequence[^1];
            if (completingToken != eosTokenId)
            {
                logits[completingToken] = float.NegativeInfinity;
            }
        }
    }

    /// <summary>
    /// Tokenizes each word once, at session setup, rather than per generated token. Drops a word
    /// that encodes to zero tokens (an edge case <see cref="Apply"/>'s indexing doesn't need to
    /// handle) or to a single EOS token — <see cref="ILogitsProcessor.Apply"/>'s own EOS-never-
    /// masked rule would ignore it anyway, so storing it would only waste a comparison every call.
    /// </summary>
    public static BannedSequenceLogitsProcessor FromWords(ITokenizer tokenizer, IReadOnlyList<string> words)
    {
        var sequences = new List<int[]>();
        foreach (var word in words)
        {
            var encoded = tokenizer.Encode(word);
            if (encoded.Count == 0)
            {
                continue;
            }

            if (encoded.Count == 1 && encoded[0] == tokenizer.EosTokenId)
            {
                continue;
            }

            sequences.Add(encoded.ToArray());
        }

        return new BannedSequenceLogitsProcessor(tokenizer.EosTokenId, sequences);
    }
}
