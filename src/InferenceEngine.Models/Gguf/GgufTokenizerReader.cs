using InferenceEngine.Core;

namespace InferenceEngine.Models.Gguf;

/// <summary>
/// Reads the <c>tokenizer.ggml.*</c> metadata embedded in a GGUF file into a plain
/// <see cref="TokenizerData"/>, for <c>InferenceEngine.Engine</c> to hand to
/// <c>InferenceEngine.Tokenizers</c>. Opens the file independently of
/// <see cref="Llama.LlamaModel.LoadFromGguf"/> — the header/metadata section is a few KB, so
/// re-reading it is simpler than threading a shared <see cref="GgufFile"/> across layers that
/// deliberately don't depend on each other.
/// </summary>
public static class GgufTokenizerReader
{
    public static TokenizerData Read(string path)
    {
        using var gguf = GgufFile.Open(path);

        var model = gguf.Metadata.GetString("tokenizer.ggml.model");
        if (model != "gpt2")
        {
            throw new NotSupportedException(
                $"GGUF tokenizer model '{model}' is not implemented yet (this POC supports 'gpt2' byte-level BPE only).");
        }

        return new TokenizerData(
            Tokens: gguf.Metadata.GetStringArray("tokenizer.ggml.tokens"),
            Merges: gguf.Metadata.GetStringArray("tokenizer.ggml.merges"),
            BosTokenId: (int)gguf.Metadata.GetU32("tokenizer.ggml.bos_token_id"),
            EosTokenId: (int)gguf.Metadata.GetU32("tokenizer.ggml.eos_token_id"),
            UnknownTokenId: (int)gguf.Metadata.GetU32OrDefault("tokenizer.ggml.unknown_token_id", 0),
            PreTokenizerName: gguf.Metadata.GetStringOrDefault("tokenizer.ggml.pre") ?? "default");
    }
}
