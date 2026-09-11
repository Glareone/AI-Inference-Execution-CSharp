using InferenceEngine.Cli.Config;

namespace InferenceEngine.Cli.Tests.Config;

/// <summary>
/// Business case: CLI configuration can come from a <c>.env</c> file, the real process
/// environment, or command-line flags — command-line flags always win over the environment,
/// which always wins over <c>.env</c> defaults. Running without a model path (from any source)
/// is rejected clearly before any model-loading work would start.
/// </summary>
/// <remarks>
/// <see cref="CliOptions.Load"/> touches real process environment variables (and, unless
/// <c>loadDotEnv: false</c> is passed, the filesystem/CWD). Every test here saves and restores
/// the exact <c>INFERENCE_*</c> keys <see cref="CliOptions"/> reads — a real <c>.env</c> or shell
/// environment could exist on the machine running these tests, so none of them may assume the
/// keys start out unset.
/// </remarks>
public class CliOptionsTests : IDisposable
{
    private static readonly string[] EnvSuffixes =
    [
        "MODEL", "PROMPT", "MAX_TOKENS", "TEMPERATURE", "TOP_K", "TOP_P", "SEED", "RAW", "STATS",
    ];

    private readonly Dictionary<string, string?> _originalEnv = new();

    public CliOptionsTests()
    {
        foreach (var suffix in EnvSuffixes)
        {
            var key = "INFERENCE_" + suffix;
            _originalEnv[key] = Environment.GetEnvironmentVariable(key);
            Environment.SetEnvironmentVariable(key, null);
        }
    }

    public void Dispose()
    {
        foreach (var (key, value) in _originalEnv)
        {
            Environment.SetEnvironmentVariable(key, value);
        }
    }

    [Fact]
    public void Load_WithoutModelFlagOrEnvironmentVariable_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() => CliOptions.Load([], loadDotEnv: false));

        Assert.Contains("--model", ex.Message);
    }

    [Fact]
    public void Load_UsesEnvironmentVariable_WhenNoFlagIsGiven()
    {
        Environment.SetEnvironmentVariable("INFERENCE_MODEL", "from-env.gguf");

        var options = CliOptions.Load([], loadDotEnv: false);

        Assert.Equal("from-env.gguf", options.ModelPath);
    }

    [Fact]
    public void Load_FlagOverridesEnvironmentVariable()
    {
        Environment.SetEnvironmentVariable("INFERENCE_MODEL", "from-env.gguf");

        var options = CliOptions.Load(["--model", "from-flag.gguf"], loadDotEnv: false);

        Assert.Equal("from-flag.gguf", options.ModelPath);
    }

    [Fact]
    public void Load_Precedence_FlagsBeatEnvironment_WhichBeatsDotEnvDefaults()
    {
        var tempDir = Directory.CreateTempSubdirectory("inference-cli-test-");
        var originalCwd = Environment.CurrentDirectory;
        try
        {
            File.WriteAllLines(Path.Combine(tempDir.FullName, ".env"),
            [
                "INFERENCE_MODEL=from-dotenv.gguf",
                "INFERENCE_PROMPT=from-dotenv-prompt",
                "INFERENCE_MAX_TOKENS=7",
            ]);
            Environment.CurrentDirectory = tempDir.FullName;

            // A real environment variable set before Load runs beats whatever the .env file says.
            Environment.SetEnvironmentVariable("INFERENCE_PROMPT", "from-env-prompt");

            var options = CliOptions.Load(["--model", "from-flag.gguf"], loadDotEnv: true);

            Assert.Equal("from-flag.gguf", options.ModelPath); // flag beat both env and .env
            Assert.Equal("from-env-prompt", options.Prompt); // env beat .env
            Assert.Equal(7, options.MaxTokens); // .env supplied the default; nothing else did
        }
        finally
        {
            Environment.CurrentDirectory = originalCwd;
            tempDir.Delete(recursive: true);
        }
    }

    [Fact]
    public void Load_WithLoadDotEnvFalse_NeverTouchesTheFilesystem()
    {
        // Regression guard for the test seam itself: a .env file sitting in the current directory
        // (e.g. a developer's real one) must not leak into a test that explicitly opts out.
        var tempDir = Directory.CreateTempSubdirectory("inference-cli-test-");
        var originalCwd = Environment.CurrentDirectory;
        try
        {
            File.WriteAllLines(Path.Combine(tempDir.FullName, ".env"), ["INFERENCE_MODEL=should-not-be-used.gguf"]);
            Environment.CurrentDirectory = tempDir.FullName;

            var ex = Assert.Throws<ArgumentException>(() => CliOptions.Load([], loadDotEnv: false));

            Assert.Contains("--model", ex.Message);
        }
        finally
        {
            Environment.CurrentDirectory = originalCwd;
            tempDir.Delete(recursive: true);
        }
    }
}
