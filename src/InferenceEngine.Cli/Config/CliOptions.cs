namespace InferenceEngine.Cli.Config;

/// <summary>
/// Resolved CLI configuration: a <c>.env</c> file (working directory, then next to the
/// executable) and the real process environment provide defaults under an
/// <c>INFERENCE_*</c> prefix; explicit command-line flags override them. Collected once here
/// so <c>Program.cs</c> orchestrates rather than parsing.
/// </summary>
internal sealed record CliOptions(
    string ModelPath,
    string Prompt,
    int MaxTokens,
    float Temperature,
    int TopK,
    float TopP,
    int? Seed,
    bool Raw,
    bool Stats,
    bool DebugTokenize,
    bool DebugLogits)
{
    private const string EnvPrefix = "INFERENCE_";

    public const string UsageText =
        "Usage: InferenceEngine.Cli --model <path.gguf> --prompt \"...\" " +
        "[--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--seed N] " +
        "[--raw] [--stats] [--debug-tokenize] [--debug-logits]\n" +
        "Any option may instead be set via a .env file or environment variable, " +
        "e.g. INFERENCE_MODEL, INFERENCE_PROMPT, INFERENCE_MAX_TOKENS.";

    public static CliOptions Load(string[] args)
    {
        DotEnvLoader.Load(".env");
        DotEnvLoader.Load(Path.Combine(AppContext.BaseDirectory, ".env"));

        string? modelPath = GetEnv("MODEL");
        var prompt = GetEnv("PROMPT") ?? "";
        var maxTokens = ParseInt(GetEnv("MAX_TOKENS")) ?? 64;
        var temperature = ParseFloat(GetEnv("TEMPERATURE")) ?? 0f;
        var topK = ParseInt(GetEnv("TOP_K")) ?? 0;
        var topP = ParseFloat(GetEnv("TOP_P")) ?? 0f;
        var seed = ParseInt(GetEnv("SEED"));
        var raw = ParseBool(GetEnv("RAW")) ?? false;
        var stats = ParseBool(GetEnv("STATS")) ?? false;
        var debugTokenize = false;
        var debugLogits = false;

        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model": modelPath = args[++i]; break;
                case "--prompt": prompt = args[++i]; break;
                case "--max-tokens": maxTokens = int.Parse(args[++i]); break;
                case "--temperature": temperature = float.Parse(args[++i]); break;
                case "--top-k": topK = int.Parse(args[++i]); break;
                case "--top-p": topP = float.Parse(args[++i]); break;
                case "--seed": seed = int.Parse(args[++i]); break;
                case "--raw": raw = true; break;
                case "--stats": stats = true; break;
                case "--debug-tokenize": debugTokenize = true; break;
                case "--debug-logits": debugLogits = true; break;
                default: throw new ArgumentException($"Unknown argument: {args[i]}");
            }
        }

        if (string.IsNullOrEmpty(modelPath))
        {
            throw new ArgumentException("--model (or INFERENCE_MODEL) is required.");
        }

        return new CliOptions(modelPath, prompt, maxTokens, temperature, topK, topP, seed, raw, stats, debugTokenize, debugLogits);
    }

    private static string? GetEnv(string suffix) => Environment.GetEnvironmentVariable(EnvPrefix + suffix);

    private static int? ParseInt(string? value) => value is null ? null : int.Parse(value);

    private static float? ParseFloat(string? value) => value is null ? null : float.Parse(value);

    private static bool? ParseBool(string? value) => value is null ? null : bool.Parse(value);
}
