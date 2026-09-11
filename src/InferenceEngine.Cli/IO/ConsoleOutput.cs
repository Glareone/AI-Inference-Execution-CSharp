namespace InferenceEngine.Cli.IO;

internal sealed class ConsoleOutput : IOutput
{
    public void Write(string text) => Console.Out.Write(text);

    public void WriteLine(string text) => Console.Out.WriteLine(text);

    public void Error(string text) => Console.Error.WriteLine(text);
}
