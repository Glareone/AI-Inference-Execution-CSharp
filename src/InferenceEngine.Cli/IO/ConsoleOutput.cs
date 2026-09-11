namespace InferenceEngine.Cli.IO;

/// <summary>Writes directly to the process's stdout/stderr text streams, unbuffered by any formatting layer.</summary>
internal sealed class ConsoleOutput : IOutput
{
    public void Write(string text) => Console.Out.Write(text);

    public void WriteLine(string text) => Console.Out.WriteLine(text);

    public void Error(string text) => Console.Error.WriteLine(text);
}
