namespace InferenceEngine.Models.Gguf;

/// <summary>
/// Typed accessors over the raw GGUF metadata KV dictionary. Scalars are boxed primitives;
/// arrays are <see cref="object"/>[] (or <see cref="string"/>[] for string arrays, read directly
/// as such since every consumer needs the concrete type).
/// </summary>
internal sealed class GgufMetadata(Dictionary<string, object> values)
{
    public bool TryGet(string key, out object value) => values.TryGetValue(key, out value!);

    public string GetString(string key) => (string)values[key];

    public string? GetStringOrDefault(string key) => values.TryGetValue(key, out var v) ? (string)v : null;

    public uint GetU32(string key) => Convert.ToUInt32(values[key]);

    public uint GetU32OrDefault(string key, uint fallback) =>
        values.TryGetValue(key, out var v) ? Convert.ToUInt32(v) : fallback;

    public float GetF32(string key) => Convert.ToSingle(values[key]);

    public float GetF32OrDefault(string key, float fallback) =>
        values.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;

    public string[] GetStringArray(string key) => (string[])values[key];

    public int[] GetI32Array(string key) => ((object[])values[key]).Select(Convert.ToInt32).ToArray();
}
