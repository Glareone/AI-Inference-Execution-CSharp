using System.Text;
using InferenceEngine.Models.Gguf;

namespace InferenceEngine.Models.Tests.Gguf;

/// <summary>
/// Writes a synthetic GGUF v3 file byte-for-byte using <see cref="BinaryWriter"/>, mirroring
/// <c>GgufFile.Open</c>'s exact read layout, so tests can exercise the real parser against small,
/// fully-controlled fixtures instead of a downloaded model. Test-only; not part of the production
/// reader.
/// </summary>
internal sealed class GgufFileBuilder
{
    private const uint MagicGguf = 0x46554747; // "GGUF" little-endian

    private readonly List<(string Key, GgufValueType Type, object Value)> _metadata = [];
    private readonly List<TensorEntry> _tensors = [];

    private readonly record struct TensorEntry(string Name, long[] Dims, GgmlType Type, long Offset, byte[] Data);

    public GgufFileBuilder AddMetadata(string key, GgufValueType type, object value)
    {
        _metadata.Add((key, type, value));
        return this;
    }

    public GgufFileBuilder AddArrayMetadata(string key, GgufValueType elementType, object[] values)
    {
        _metadata.Add((key, GgufValueType.Array, (elementType, values)));
        return this;
    }

    /// <param name="offset">Byte offset relative to the (aligned) data section start.</param>
    public GgufFileBuilder AddTensor(string name, long[] dims, GgmlType type, long offset, byte[] data)
    {
        _tensors.Add(new TensorEntry(name, dims, type, offset, data));
        return this;
    }

    /// <summary>Writes the file to a fresh temp path and returns it.</summary>
    public string WriteToTempFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gguf-test-{Guid.NewGuid():N}.gguf");
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(MagicGguf);
            writer.Write(3u); // version
            writer.Write((ulong)_tensors.Count);
            writer.Write((ulong)_metadata.Count);

            foreach (var (key, type, value) in _metadata)
            {
                WriteGgufString(writer, key);
                writer.Write((uint)type);
                WriteValue(writer, type, value);
            }

            foreach (var tensor in _tensors)
            {
                WriteGgufString(writer, tensor.Name);
                writer.Write((uint)tensor.Dims.Length);
                foreach (var d in tensor.Dims)
                {
                    writer.Write((ulong)d);
                }

                writer.Write((uint)tensor.Type);
                writer.Write((ulong)tensor.Offset);
            }

            writer.Flush();
        }

        var headerEnd = stream.Position;
        var alignment = GetAlignmentOrDefault();
        var dataOffset = AlignUp(headerEnd, alignment);

        var maxExtent = _tensors.Count == 0 ? 0 : _tensors.Max(t => t.Offset + t.Data.Length);
        stream.SetLength(dataOffset + maxExtent);

        foreach (var tensor in _tensors)
        {
            stream.Position = dataOffset + tensor.Offset;
            stream.Write(tensor.Data, 0, tensor.Data.Length);
        }

        return path;
    }

    private uint GetAlignmentOrDefault()
    {
        foreach (var (key, _, value) in _metadata)
        {
            if (key == "general.alignment")
            {
                return Convert.ToUInt32(value);
            }
        }

        return 32;
    }

    private static long AlignUp(long value, long alignment) => (value + alignment - 1) / alignment * alignment;

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static void WriteValue(BinaryWriter writer, GgufValueType type, object value)
    {
        switch (type)
        {
            case GgufValueType.UInt8: writer.Write((byte)value); break;
            case GgufValueType.Int8: writer.Write((sbyte)value); break;
            case GgufValueType.UInt16: writer.Write((ushort)value); break;
            case GgufValueType.Int16: writer.Write((short)value); break;
            case GgufValueType.UInt32: writer.Write((uint)value); break;
            case GgufValueType.Int32: writer.Write((int)value); break;
            case GgufValueType.Float32: writer.Write((float)value); break;
            case GgufValueType.Bool: writer.Write((bool)value); break;
            case GgufValueType.String: WriteGgufString(writer, (string)value); break;
            case GgufValueType.UInt64: writer.Write((ulong)value); break;
            case GgufValueType.Int64: writer.Write((long)value); break;
            case GgufValueType.Float64: writer.Write((double)value); break;
            case GgufValueType.Array:
                var (elementType, values) = ((GgufValueType ElementType, object[] Values))value;
                writer.Write((uint)elementType);
                writer.Write((ulong)values.Length);
                foreach (var v in values)
                {
                    WriteValue(writer, elementType, v);
                }

                break;
            default:
                throw new NotSupportedException($"Unsupported GGUF value type {type} in test builder.");
        }
    }
}
