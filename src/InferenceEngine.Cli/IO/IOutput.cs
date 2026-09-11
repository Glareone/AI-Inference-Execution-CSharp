namespace InferenceEngine.Cli.IO;

/// <summary>
/// The CLI's only output boundary. <c>Program.cs</c> decides *what* to report; it never touches
/// <see cref="Console"/> directly, so the same orchestration works whether the process is run
/// interactively, from a script, or hosted by something else that wants the raw text stream.
/// </summary>
internal interface IOutput
{
    void Write(string text);

    void WriteLine(string text);

    void Error(string text);
}
