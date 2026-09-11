using InferenceEngine.Cli.IO;

namespace InferenceEngine.Cli.Tests.IO;

/// <summary>
/// Business case: <see cref="ConsoleOutput"/> is the CLI's only output boundary — everything it
/// writes must actually reach the real stdout/stderr text streams. Thin by nature (it's a direct
/// pass-through to <see cref="Console"/>), so a minimal redirection check is enough.
/// </summary>
public class ConsoleOutputTests
{
    [Fact]
    public void WriteAndWriteLine_GoToStandardOutput()
    {
        var originalOut = Console.Out;
        var writer = new StringWriter();
        Console.SetOut(writer);
        try
        {
            var output = new ConsoleOutput();

            output.Write("hello ");
            output.WriteLine("world");

            Assert.Equal("hello world" + Environment.NewLine, writer.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
        }
    }

    [Fact]
    public void Error_GoesToStandardError()
    {
        var originalError = Console.Error;
        var writer = new StringWriter();
        Console.SetError(writer);
        try
        {
            var output = new ConsoleOutput();

            output.Error("something went wrong");

            Assert.Equal("something went wrong" + Environment.NewLine, writer.ToString());
        }
        finally
        {
            Console.SetError(originalError);
        }
    }
}
