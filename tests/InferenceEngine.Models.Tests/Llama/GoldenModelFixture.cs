using InferenceEngine.Engine;

namespace InferenceEngine.Models.Tests.Llama;

/// <summary>
/// Loads the real GGUF model named by <c>INFERENCE_MODEL</c> once per test class run (loading a
/// 270 MB F16 GGUF is by far the most expensive part of <see cref="GoldenLogitBaselineTests"/>,
/// and both tests in that class need the same loaded model). If the environment variable is
/// unset, empty, or doesn't point at an existing file — the case on any machine that hasn't
/// downloaded the model — <see cref="Session"/> is <c>null</c> and <see cref="SkipReason"/>
/// explains why, so tests can skip cleanly via <c>Assert.SkipUnless</c> instead of failing.
/// </summary>
public sealed class GoldenModelFixture
{
    public const string ModelPathEnvVar = "INFERENCE_MODEL";

    public InferenceSession? Session { get; }

    public string SkipReason { get; }

    public GoldenModelFixture()
    {
        var modelPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
        {
            Session = null;
            SkipReason =
                $"{ModelPathEnvVar} is not set to an existing GGUF file " +
                $"(got '{modelPath}') — skipping the golden logit baseline capture.";
            return;
        }

        Session = InferenceSession.Load(modelPath);
        SkipReason = "";
    }
}
