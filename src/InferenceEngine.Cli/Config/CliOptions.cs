using System.Globalization;

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

    /// <param name="loadDotEnv">
    /// When <c>false</c>, skips reading <c>.env</c> files entirely, so tests can exercise the
    /// flags/environment-merge logic in isolation from real file I/O and a stray root
    /// <c>.env</c>. Always <c>true</c> in production.
    /// </param>
    public static CliOptions Load(string[] args, bool loadDotEnv = true)
    {
        if (loadDotEnv)
        {
            DotEnvLoader.Load(".env");
            DotEnvLoader.Load(Path.Combine(AppContext.BaseDirectory, ".env"));
        }

        string? modelPath = GetEnv("MODEL");
        var prompt = GetEnv("PROMPT") ?? "";
        var maxTokens = ParseInt(GetEnv("MAX_TOKENS"), EnvPrefix + "MAX_TOKENS") ?? 64;
        var temperature = ParseFloat(GetEnv("TEMPERATURE"), EnvPrefix + "TEMPERATURE") ?? 0f;
        var topK = ParseInt(GetEnv("TOP_K"), EnvPrefix + "TOP_K") ?? 0;
        var topP = ParseFloat(GetEnv("TOP_P"), EnvPrefix + "TOP_P") ?? 0f;
        var seed = ParseInt(GetEnv("SEED"), EnvPrefix + "SEED");
        var raw = ParseBool(GetEnv("RAW"), EnvPrefix + "RAW") ?? false;
        var stats = ParseBool(GetEnv("STATS"), EnvPrefix + "STATS") ?? false;
        var debugTokenize = false;
        var debugLogits = false;

        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model": modelPath = NextValue(args, ref i); break;
                case "--prompt": prompt = NextValue(args, ref i); break;
                case "--max-tokens": maxTokens = ParseRequiredInt(args, ref i); break;
                case "--temperature": temperature = ParseRequiredFloat(args, ref i); break;
                case "--top-k": topK = ParseRequiredInt(args, ref i); break;
                case "--top-p": topP = ParseRequiredFloat(args, ref i); break;
                case "--seed": seed = ParseRequiredInt(args, ref i); break;
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

    private static string NextValue(string[] args, ref int i)
    {
        if (i + 1 >= args.Length)
        {
            throw new ArgumentException($"'{args[i]}' requires a value.");
        }

        return args[++i];
    }

    private static int ParseRequiredInt(string[] args, ref int i)
    {
        var flag = args[i];
        var value = NextValue(args, ref i);
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var result))
        {
            throw new ArgumentException($"'{flag}' expects an integer, got '{value}'.");
        }

        return result;
    }

    private static float ParseRequiredFloat(string[] args, ref int i)
    {
        var flag = args[i];
        var value = NextValue(args, ref i);
        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var result))
        {
            throw new ArgumentException($"'{flag}' expects a number, got '{value}'.");
        }

        return result;
    }

    private static int? ParseInt(string? value, string name)
    {
        if (value is null)
        {
            return null;
        }

        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var result))
        {
            throw new ArgumentException($"{name} expects an integer, got '{value}'.");
        }

        return result;
    }

    private static float? ParseFloat(string? value, string name)
    {
        if (value is null)
        {
            return null;
        }

        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var result))
        {
            throw new ArgumentException($"{name} expects a number, got '{value}'.");
        }

        return result;
    }

    private static bool? ParseBool(string? value, string name)
    {
        if (value is null)
        {
            return null;
        }

        if (!bool.TryParse(value, out var result))
        {
            throw new ArgumentException($"{name} expects true/false, got '{value}'.");
        }

        return result;
    }
}
