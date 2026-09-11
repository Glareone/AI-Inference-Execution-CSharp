using System.Diagnostics;

namespace InferenceEngine.Cli.Diagnostics;

internal static class Timed
{
    public static (T Result, TimeSpan Elapsed) Run<T>(Func<T> action)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = action();
        stopwatch.Stop();
        return (result, stopwatch.Elapsed);
    }

    public static TimeSpan Run(Action action)
    {
        var stopwatch = Stopwatch.StartNew();
        action();
        stopwatch.Stop();
        return stopwatch.Elapsed;
    }
}
