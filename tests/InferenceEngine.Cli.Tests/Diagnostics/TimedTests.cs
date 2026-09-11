using InferenceEngine.Cli.Diagnostics;

namespace InferenceEngine.Cli.Tests.Diagnostics;

/// <summary>
/// Business case: <see cref="Timed"/> reports elapsed wall-clock time around an action and
/// returns the action's result unchanged. Minimal by nature — it's a thin
/// <see cref="System.Diagnostics.Stopwatch"/> wrapper.
/// </summary>
public class TimedTests
{
    [Fact]
    public void Run_WithResult_ReturnsTheActionsResultAndNonNegativeElapsedTime()
    {
        var (result, elapsed) = Timed.Run(() => 42);

        Assert.Equal(42, result);
        Assert.True(elapsed >= TimeSpan.Zero);
    }

    [Fact]
    public void Run_WithoutResult_ReturnsNonNegativeElapsedTimeAndRunsTheAction()
    {
        var ran = false;

        var elapsed = Timed.Run(() => { ran = true; });

        Assert.True(ran);
        Assert.True(elapsed >= TimeSpan.Zero);
    }
}
