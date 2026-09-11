namespace InferenceEngine.Cli.Config;

/// <summary>
/// Loads a <c>.env</c> file (simple <c>KEY=VALUE</c> lines, <c>#</c> comments, blank lines
/// ignored) into the process environment. Hand-rolled rather than a NuGet package — the format
/// is a handful of lines to parse and this is CLI convenience, not an inference concern.
/// </summary>
internal static class DotEnvLoader
{
    public static void Load(string path)
    {
        if (!File.Exists(path))
        {
            return;
        }

        foreach (var line in File.ReadLines(path))
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#'))
            {
                continue;
            }

            var separatorIndex = trimmed.IndexOf('=');
            if (separatorIndex < 0)
            {
                continue;
            }

            var key = trimmed[..separatorIndex].Trim();
            var value = trimmed[(separatorIndex + 1)..].Trim().Trim('"');

            // Don't override a value already set in the real environment.
            if (Environment.GetEnvironmentVariable(key) is null)
            {
                Environment.SetEnvironmentVariable(key, value);
            }
        }
    }
}
