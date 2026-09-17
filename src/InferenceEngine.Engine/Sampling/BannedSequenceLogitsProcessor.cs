using InferenceEngine.Core;

namespace InferenceEngine.Engine.Sampling;

internal sealed class BannedSequenceLogitsProcessor(int eosTokenId, IReadOnlyList<int[]> bannedSequences) : ILogitsProcessor
{
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
