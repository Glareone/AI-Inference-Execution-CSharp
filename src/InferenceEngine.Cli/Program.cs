using InferenceEngine.Cli.Config;
using InferenceEngine.Cli.Diagnostics;
using InferenceEngine.Cli.IO;
using InferenceEngine.Engine;
using InferenceEngine.Engine.Config;

IOutput output = new ConsoleOutput();

CliOptions options;
try
{
    options = CliOptions.Load(args);
}
catch (ArgumentException ex)
{
    output.Error(ex.Message);
    output.Error(CliOptions.UsageText);
    return 1;
}

try
{
    var (session, loadElapsed) = Timed.Run(() => InferenceSession.Load(options.ModelPath));

    if (options.DebugTokenize)
    {
        var config = session.Config;
        output.WriteLine(
            $"architecture={config.Architecture} layers={config.NumLayers} hidden={config.HiddenSize} " +
            $"heads={config.NumAttentionHeads}/{config.NumKvHeads} ffn={config.FfnHiddenSize} " +
            $"vocab={config.VocabSize} ropeFreqBase={config.RopeFreqBase} maxSeqLen={config.MaxSeqLen}");

        var ids = session.Tokenize(options.Prompt, options.Raw);
        output.WriteLine($"tokens ({ids.Count}): [{string.Join(", ", ids)}]");
        output.WriteLine($"decode: {session.Decode(ids)}");
    }

    if (options.DebugLogits)
    {
        var ids = session.Tokenize(options.Prompt, options.Raw);
        output.WriteLine("top-5 next-token logits after prefill:");
        foreach (var (id, text, logit) in session.PrefillTopLogits(ids, 5))
        {
            output.WriteLine($"  {id,6}  {logit,8:F3}  {text.Replace("\n", "\\n")}");
        }
    }

    if (options.DebugTokenize || options.DebugLogits)
    {
        return 0;
    }

    var generationOptions = new GenerationOptions(options.MaxTokens, options.Temperature, options.TopK, options.TopP, options.Seed, options.Raw);

    var tokenCount = 0;
    var genElapsed = Timed.Run(() =>
    {
        foreach (var token in session.Generate(options.Prompt, generationOptions))
        {
            output.Write(token.Text);
            tokenCount++;
        }
    });
    output.WriteLine("");

    if (options.Stats)
    {
        var tokensPerSecond = tokenCount / genElapsed.TotalSeconds;
        output.WriteLine(
            $"load {loadElapsed.TotalMilliseconds:F0} ms | generated {tokenCount} tok in " +
            $"{genElapsed.TotalSeconds:F2}s ({tokensPerSecond:F1} tok/s)");
    }

    return 0;
}
catch (ArgumentException ex)
{
    output.Error(ex.Message);
    return 1;
}
