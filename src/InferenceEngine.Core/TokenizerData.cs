namespace InferenceEngine.Core;

/// <summary>
/// Raw vocabulary/merge data needed to construct a byte-level BPE tokenizer, extracted from a
/// model file's embedded metadata (e.g. GGUF's <c>tokenizer.ggml.*</c> keys). A plain data
/// contract between <c>InferenceEngine.Models</c> (which reads it) and
/// <c>InferenceEngine.Tokenizers</c> (which builds an <see cref="ITokenizer"/> from it) — this
/// keeps Tokenizers from depending on Models' format-parsing internals, per the solution-layout
/// ADR. Lives in Core because it's exchanged between two sibling layers, neither of which
/// depends on the other.
/// </summary>
public sealed record TokenizerData(
    string[] Tokens,
    string[] Merges,
    int BosTokenId,
    int EosTokenId,
    string PreTokenizerName);
