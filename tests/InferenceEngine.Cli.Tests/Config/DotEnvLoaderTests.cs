using InferenceEngine.Cli.Config;

namespace InferenceEngine.Cli.Tests.Config;

/// <summary>
/// Business case: <c>.env</c> parsing tolerates comments, blank lines, and quoted values, and
/// never overrides a variable already present in the real process environment. Each test uses a
/// unique temp file path and a unique, GUID-suffixed environment variable name so nothing
/// collides with a developer's real environment or with other tests running in the same process.
/// </summary>
public class DotEnvLoaderTests
{
    [Fact]
    public void Load_SetsEnvironmentVariablesFromKeyValueLines()
    {
        var key = UniqueKey();
        var path = WriteTempEnvFile([$"{key}=hello"]);

        try
        {
            DotEnvLoader.Load(path);

            Assert.Equal("hello", Environment.GetEnvironmentVariable(key));
        }
        finally
        {
            Cleanup(path, key);
        }
    }

    [Fact]
    public void Load_IgnoresCommentsAndBlankLines()
    {
        var key = UniqueKey();
        var path = WriteTempEnvFile(
        [
            "# a comment line",
            "",
            "   ",
            $"{key}=value",
        ]);

        try
        {
            DotEnvLoader.Load(path);

            Assert.Equal("value", Environment.GetEnvironmentVariable(key));
        }
        finally
        {
            Cleanup(path, key);
        }
    }

    [Fact]
    public void Load_StripsSurroundingQuotesFromValues()
    {
        var key = UniqueKey();
        var path = WriteTempEnvFile([$"{key}=\"quoted value\""]);

        try
        {
            DotEnvLoader.Load(path);

            Assert.Equal("quoted value", Environment.GetEnvironmentVariable(key));
        }
        finally
        {
            Cleanup(path, key);
        }
    }

    [Fact]
    public void Load_NeverOverridesAnAlreadySetEnvironmentVariable()
    {
        var key = UniqueKey();
        Environment.SetEnvironmentVariable(key, "already-set");
        var path = WriteTempEnvFile([$"{key}=from-dotenv"]);

        try
        {
            DotEnvLoader.Load(path);

            Assert.Equal("already-set", Environment.GetEnvironmentVariable(key));
        }
        finally
        {
            Cleanup(path, key);
        }
    }

    [Fact]
    public void Load_WithMissingFile_DoesNothingAndDoesNotThrow()
    {
        var missingPath = Path.Combine(Path.GetTempPath(), $"does-not-exist-{Guid.NewGuid():N}.env");

        var exception = Record.Exception(() => DotEnvLoader.Load(missingPath));

        Assert.Null(exception);
    }

    private static string UniqueKey() => $"INFERENCE_TEST_{Guid.NewGuid():N}";

    private static string WriteTempEnvFile(string[] lines)
    {
        var path = Path.Combine(Path.GetTempPath(), $"dotenv-test-{Guid.NewGuid():N}.env");
        File.WriteAllLines(path, lines);
        return path;
    }

    private static void Cleanup(string path, string key)
    {
        File.Delete(path);
        Environment.SetEnvironmentVariable(key, null);
    }
}
