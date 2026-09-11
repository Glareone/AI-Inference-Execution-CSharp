using System.Diagnostics;
using InferenceEngine.Engine;

string? modelPath = null;
var prompt = "";
var maxTokens = 64;
var temperature = 0f;
var topK = 0;
var topP = 0f;
int? seed = null;
var raw = false;
var stats = false;
var debugTokenize = false;
var debugLogits = false;

for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model": modelPath = args[++i]; break;
        case "--prompt": prompt = args[++i]; break;
        case "--max-tokens": maxTokens = int.Parse(args[++i]); break;
        case "--temperature": temperature = float.Parse(args[++i]); break;
        case "--top-k": topK = int.Parse(args[++i]); break;
        case "--top-p": topP = float.Parse(args[++i]); break;
        case "--seed": seed = int.Parse(args[++i]); break;
        case "--raw": raw = true; break;
        case "--stats": stats = true; break;
        case "--debug-tokenize": debugTokenize = true; break;
        case "--debug-logits": debugLogits = true; break;
        default:
            Console.Error.WriteLine($"Unknown argument: {args[i]}");
            return 1;
    }
}

if (modelPath is null)
{
    Console.Error.WriteLine(
        "Usage: InferenceEngine.Cli --model <path.gguf> --prompt \"...\" " +
        "[--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--seed N] " +
        "[--raw] [--stats] [--debug-tokenize] [--debug-logits]");
    return 1;
}

var loadStopwatch = Stopwatch.StartNew();
var session = InferenceSession.Load(modelPath);
loadStopwatch.Stop();

if (debugTokenize)
{
    var config = session.Config;
    Console.WriteLine(
        $"architecture={config.Architecture} layers={config.NumLayers} hidden={config.HiddenSize} " +
        $"heads={config.NumAttentionHeads}/{config.NumKvHeads} ffn={config.FfnHiddenSize} " +
        $"vocab={config.VocabSize} ropeFreqBase={config.RopeFreqBase} maxSeqLen={config.MaxSeqLen}");

    var ids = session.Tokenize(prompt, raw);
    Console.WriteLine($"tokens ({ids.Count}): [{string.Join(", ", ids)}]");
    Console.WriteLine($"decode: {session.Decode(ids)}");
}

if (debugLogits)
{
    var ids = session.Tokenize(prompt, raw);
    Console.WriteLine("top-5 next-token logits after prefill:");
    foreach (var (id, text, logit) in session.PrefillTopLogits(ids, 5))
    {
        Console.WriteLine($"  {id,6}  {logit,8:F3}  {text.Replace("\n", "\\n")}");
    }
}

if (debugTokenize || debugLogits)
{
    return 0;
}

var options = new GenerationOptions(maxTokens, temperature, topK, topP, seed, raw);

var genStopwatch = Stopwatch.StartNew();
var tokenCount = 0;
foreach (var token in session.Generate(prompt, options))
{
    Console.Write(token.Text);
    tokenCount++;
}

genStopwatch.Stop();
Console.WriteLine();

if (stats)
{
    var tokensPerSecond = tokenCount / genStopwatch.Elapsed.TotalSeconds;
    Console.WriteLine(
        $"load {loadStopwatch.ElapsedMilliseconds} ms | generated {tokenCount} tok in " +
        $"{genStopwatch.Elapsed.TotalSeconds:F2}s ({tokensPerSecond:F1} tok/s)");
}

return 0;
