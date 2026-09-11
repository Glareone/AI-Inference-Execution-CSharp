namespace InferenceEngine.Core;

/// <summary>
/// Architecture-level configuration for a loaded model, extracted from its metadata
/// (e.g. GGUF <c>{arch}.*</c> keys). Immutable — one instance per loaded model.
/// </summary>
public sealed record ModelConfig(
    string Architecture,
    int VocabSize,
    int HiddenSize,
    int NumLayers,
    int NumAttentionHeads,
    int NumKvHeads,
    int HeadDim,
    int FfnHiddenSize,
    int MaxSeqLen,
    float RmsNormEps,
    float RopeFreqBase);
