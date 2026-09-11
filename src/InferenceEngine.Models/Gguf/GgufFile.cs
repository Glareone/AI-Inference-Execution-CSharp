using System.IO.MemoryMappedFiles;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;

namespace InferenceEngine.Models.Gguf;

/// <summary>
/// Minimal GGUF (v3) reader: memory-maps the file, parses the header/metadata/tensor-info
/// sections eagerly, and materializes individual tensors to <c>float[]</c> on demand.
/// </summary>
/// <remarks>
/// There is no maintained GGUF-parsing library on NuGet (see the format-loading ADR), so this
/// reader is hand-written — kept intentionally small: only <see cref="GgmlType.F32"/> and
/// <see cref="GgmlType.F16"/> tensor data can be materialized; every other quantized type
/// parses correctly as metadata/shape but throws on <see cref="ReadTensorAsF32"/>.
/// </remarks>
internal sealed class GgufFile : IDisposable
{
    private const uint MagicGguf = 0x46554747; // "GGUF" little-endian
    private const uint SupportedVersion = 3;
    private const long DefaultAlignment = 32;

    private readonly MemoryMappedFile _mmf;
    private readonly long _dataOffset;
    private readonly Dictionary<string, GgufTensorDescriptor> _tensors;

    public GgufMetadata Metadata { get; }

    private GgufFile(MemoryMappedFile mmf, long dataOffset, GgufMetadata metadata, Dictionary<string, GgufTensorDescriptor> tensors)
    {
        _mmf = mmf;
        _dataOffset = dataOffset;
        Metadata = metadata;
        _tensors = tensors;
    }

    public static GgufFile Open(string path)
    {
        var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        try
        {
            using var headerStream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
            using var reader = new BinaryReader(headerStream, Encoding.UTF8, leaveOpen: true);

            if (reader.ReadUInt32() != MagicGguf)
            {
                throw new InvalidDataException($"'{path}' is not a GGUF file (bad magic).");
            }

            var version = reader.ReadUInt32();
            if (version != SupportedVersion)
            {
                throw new NotSupportedException($"GGUF version {version} is not supported (only v{SupportedVersion}).");
            }

            var tensorCount = ReadCount(reader, "tensor_count");
            var kvCount = ReadCount(reader, "metadata_kv_count");

            var rawMetadata = new Dictionary<string, object>(kvCount);
            for (var i = 0; i < kvCount; i++)
            {
                var key = ReadGgufString(reader);
                var type = (GgufValueType)reader.ReadUInt32();
                rawMetadata[key] = ReadValue(reader, type);
            }

            var metadata = new GgufMetadata(rawMetadata);

            var tensors = new Dictionary<string, GgufTensorDescriptor>(tensorCount);
            for (var i = 0; i < tensorCount; i++)
            {
                var name = ReadGgufString(reader);
                var nDims = reader.ReadUInt32();
                if (nDims > (uint)headerStream.Length)
                {
                    throw new InvalidDataException($"GGUF tensor '{name}' has an implausible dimension count ({nDims}).");
                }

                var dims = new long[nDims];
                for (var d = 0; d < nDims; d++)
                {
                    dims[d] = checked((long)reader.ReadUInt64());
                }

                var ggmlType = (GgmlType)reader.ReadUInt32();
                var offset = checked((long)reader.ReadUInt64());
                tensors[name] = new GgufTensorDescriptor(name, dims, ggmlType, offset);
            }

            var alignment = metadata.GetU32OrDefault("general.alignment", (uint)DefaultAlignment);
            var dataOffset = AlignUp(headerStream.Position, alignment);

            return new GgufFile(mmf, dataOffset, metadata, tensors);
        }
        catch
        {
            mmf.Dispose();
            throw;
        }
    }

    public bool HasTensor(string name) => _tensors.ContainsKey(name);

    /// <summary>Row-major shape as <c>[out, in]</c> for a 2-D weight, or the single length for a 1-D vector.</summary>
    public long[] TensorShape(string name) => GetDescriptor(name).Dims;

    public float[] ReadTensorAsF32(string name)
    {
        var descriptor = GetDescriptor(name);
        var count = checked((int)descriptor.ElementCount);
        var absoluteOffset = _dataOffset + descriptor.Offset;

        switch (descriptor.Type)
        {
            case GgmlType.F32:
            {
                var buffer = new byte[count * sizeof(float)];
                ReadRawBytes(absoluteOffset, buffer);
                return MemoryMarshal.Cast<byte, float>(buffer).ToArray();
            }
            case GgmlType.F16:
            {
                var buffer = new byte[count * 2];
                ReadRawBytes(absoluteOffset, buffer);
                var halves = MemoryMarshal.Cast<byte, Half>(buffer);
                var result = new float[count];
                TensorPrimitives.ConvertToSingle(halves, result);
                return result;
            }
            default:
                throw new NotSupportedException(
                    $"GGUF tensor type {descriptor.Type} ('{name}') is not implemented yet (this POC supports F32/F16 only).");
        }
    }

    private GgufTensorDescriptor GetDescriptor(string name) =>
        _tensors.TryGetValue(name, out var descriptor)
            ? descriptor
            : throw new KeyNotFoundException($"GGUF file has no tensor named '{name}'.");

    private void ReadRawBytes(long absoluteOffset, byte[] buffer)
    {
        using var stream = _mmf.CreateViewStream(absoluteOffset, buffer.Length, MemoryMappedFileAccess.Read);
        stream.ReadExactly(buffer);
    }

    private static string ReadGgufString(BinaryReader reader)
    {
        var length = ReadCount(reader, "string length");
        var bytes = new byte[length];
        reader.BaseStream.ReadExactly(bytes); // BinaryReader.ReadBytes would silently truncate on a short/corrupt file
        return Encoding.UTF8.GetString(bytes);
    }

    /// <summary>
    /// Reads a file-declared count (a string length, array length, or the top-level
    /// tensor/metadata counts) and rejects one that exceeds the file's own size — such a count
    /// can never be valid, and letting it through risks an excessive allocation from a small
    /// malformed or truncated file.
    /// </summary>
    private static int ReadCount(BinaryReader reader, string what)
    {
        var raw = reader.ReadUInt64();
        if (raw > (ulong)reader.BaseStream.Length)
        {
            throw new InvalidDataException($"GGUF {what} ({raw}) exceeds the file size and cannot be valid.");
        }

        return checked((int)raw);
    }

    private static object ReadValue(BinaryReader reader, GgufValueType type) => type switch
    {
        GgufValueType.UInt8 => reader.ReadByte(),
        GgufValueType.Int8 => reader.ReadSByte(),
        GgufValueType.UInt16 => reader.ReadUInt16(),
        GgufValueType.Int16 => reader.ReadInt16(),
        GgufValueType.UInt32 => reader.ReadUInt32(),
        GgufValueType.Int32 => reader.ReadInt32(),
        GgufValueType.Float32 => reader.ReadSingle(),
        GgufValueType.Bool => reader.ReadBoolean(),
        GgufValueType.String => ReadGgufString(reader),
        GgufValueType.UInt64 => reader.ReadUInt64(),
        GgufValueType.Int64 => reader.ReadInt64(),
        GgufValueType.Float64 => reader.ReadDouble(),
        GgufValueType.Array => ReadArray(reader),
        _ => throw new NotSupportedException($"Unknown GGUF metadata value type {type}."),
    };

    private static object ReadArray(BinaryReader reader)
    {
        var elementType = (GgufValueType)reader.ReadUInt32();
        var count = ReadCount(reader, "array element count");

        if (elementType == GgufValueType.String)
        {
            var strings = new string[count];
            for (var i = 0; i < count; i++)
            {
                strings[i] = ReadGgufString(reader);
            }

            return strings;
        }

        var values = new object[count];
        for (var i = 0; i < count; i++)
        {
            values[i] = ReadValue(reader, elementType);
        }

        return values;
    }

    private static long AlignUp(long value, long alignment) => (value + alignment - 1) / alignment * alignment;

    public void Dispose() => _mmf.Dispose();
}
