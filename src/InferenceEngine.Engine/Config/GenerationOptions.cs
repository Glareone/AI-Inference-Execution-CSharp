namespace InferenceEngine.Engine.Config;

/// <summary>
/// Options for <see cref="InferenceSession.Generate"/>. <see cref="Temperature"/> of 0 (the
/// default) means greedy decoding — <see cref="TopK"/>/<see cref="TopP"/> are then ignored.
/// </summary>
public sealed record GenerationOptions(
    int MaxNewTokens = 64,
    float Temperature = 0f,
    int TopK = 0,
    float TopP = 0f,
    int? Seed = null,
    /// <summary>Skip the ChatML wrapping and encode the prompt text as-is.</summary>
    bool Raw = false,
    /// <summary>
    /// Literal strings to ban from generated output; each is tokenized once at session setup.
    /// A word that BPE-splits into more than one token requires that exact token sequence to
    /// appear consecutively in the generated output to be blocked. The caller supplies exact
    /// variants (e.g. <c>"EPAM"</c>, <c>"EPAM "</c>) — there is no automatic case/whitespace
    /// derivation.
    /// </summary>
    IReadOnlyList<string>? BannedWords = null);
