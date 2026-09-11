namespace InferenceEngine.Engine;

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
    bool Raw = false);
