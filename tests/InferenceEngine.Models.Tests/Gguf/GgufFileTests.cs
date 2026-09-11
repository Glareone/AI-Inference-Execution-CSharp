using InferenceEngine.Models.Gguf;

namespace InferenceEngine.Models.Tests.Gguf;

/// <summary>
/// <c>GgufFile</c> is a hand-rolled reader for the GGUF v3 model-file format (no maintained
/// NuGet parser exists — see the format-loading ADR). Business case: given a GGUF file's raw
/// bytes, model metadata and tensor data must come back exactly as they were written, and
/// unsupported tensor encodings must fail loudly rather than silently return wrong numbers.
/// Fixtures are built in-process with <see cref="GgufFileBuilder"/> rather than a downloaded
/// model, so every case is small and deterministic.
/// </summary>
public class GgufFileTests
{
    [Fact]
    public void ScalarMetadata_RoundTripsExactValues()
    {
        var path = new GgufFileBuilder()
            .AddMetadata("general.name", GgufValueType.String, "smollm-test")
            .AddMetadata("llama.context_length", GgufValueType.UInt32, 2048u)
            .AddMetadata("llama.rope.freq_base", GgufValueType.Float32, 10000f)
            .WriteToTempFile();

        try
        {
            using var file = GgufFile.Open(path);

            Assert.Equal("smollm-test", file.Metadata.GetString("general.name"));
            Assert.Equal(2048u, file.Metadata.GetU32("llama.context_length"));
            Assert.Equal(10000f, file.Metadata.GetF32("llama.rope.freq_base"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ScalarMetadata_MissingKey_FallsBackToOrDefaultValue()
    {
        var path = new GgufFileBuilder().WriteToTempFile();

        try
        {
            using var file = GgufFile.Open(path);

            Assert.Null(file.Metadata.GetStringOrDefault("missing.key"));
            Assert.Equal(99u, file.Metadata.GetU32OrDefault("missing.key", 99u));
            Assert.Equal(1.5f, file.Metadata.GetF32OrDefault("missing.key", 1.5f));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ArrayMetadata_RoundTripsExactValues()
    {
        var path = new GgufFileBuilder()
            .AddArrayMetadata("tokenizer.ggml.tokens", GgufValueType.String, ["<s>", "a", "b"])
            .AddArrayMetadata("tokenizer.ggml.token_type", GgufValueType.Int32, [1, 2, 3])
            .WriteToTempFile();

        try
        {
            using var file = GgufFile.Open(path);

            Assert.Equal(["<s>", "a", "b"], file.Metadata.GetStringArray("tokenizer.ggml.tokens"));
            Assert.Equal([1, 2, 3], file.Metadata.GetI32Array("tokenizer.ggml.token_type"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void F32Tensor_RoundTripsExactValues()
    {
        float[] expected = [1.5f, -2.25f, 3f];
        var path = new GgufFileBuilder()
            .AddTensor("weight.f32", [3], GgmlType.F32, offset: 0, data: FloatBytes(expected))
            .WriteToTempFile();

        try
        {
            using var file = GgufFile.Open(path);

            Assert.True(file.HasTensor("weight.f32"));
            Assert.Equal([3L], file.TensorShape("weight.f32"));
            Assert.Equal(expected, file.ReadTensorAsF32("weight.f32"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void F16Tensor_DequantizesToF32()
    {
        float[] expected = [1f, 2.5f, -3.75f];
        var path = new GgufFileBuilder()
            .AddTensor("weight.f16", [3], GgmlType.F16, offset: 0, data: HalfBytes(expected))
            .WriteToTempFile();

        try
        {
            using var file = GgufFile.Open(path);

            var actual = file.ReadTensorAsF32("weight.f16");
            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i], 0.001f);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void QuantizedTensor_ThrowsNotSupportedException_InsteadOfWrongData()
    {
        // Q4_0 (ggml type 2): parses fine as metadata/shape, but this POC never dequantizes it,
        // so materializing it must fail loudly rather than silently hand back garbage floats.
        var path = new GgufFileBuilder()
            .AddTensor("weight.q4_0", [32], GgmlType.Q4_0, offset: 0, data: new byte[18])
            .WriteToTempFile();

        try
        {
            using var file = GgufFile.Open(path);

            Assert.True(file.HasTensor("weight.q4_0"));
            Assert.Throws<NotSupportedException>(() => file.ReadTensorAsF32("weight.q4_0"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void UnknownTensorName_ThrowsKeyNotFoundException()
    {
        var path = new GgufFileBuilder().WriteToTempFile();

        try
        {
            using var file = GgufFile.Open(path);

            Assert.False(file.HasTensor("nope"));
            Assert.Throws<KeyNotFoundException>(() => file.ReadTensorAsF32("nope"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void MultipleTensors_DoNotOverlap_EvenWithDefaultAlignmentPadding()
    {
        // Two adjacent, tightly-packed tensors after a header whose length is not itself a
        // multiple of the default 32-byte alignment (a metadata string of odd length forces
        // real padding bytes before the data section) — if the reader computed the aligned data
        // offset incorrectly, one or both tensors would read back wrong or overlapping values.
        float[] first = [10f, 20f];
        float[] second = [30f, 40f];

        var path = new GgufFileBuilder()
            .AddMetadata("general.name", GgufValueType.String, "odd-length-name") // forces non-aligned header end
            .AddTensor("first", [2], GgmlType.F32, offset: 0, data: FloatBytes(first))
            .AddTensor("second", [2], GgmlType.F32, offset: 8, data: FloatBytes(second))
            .WriteToTempFile();

        try
        {
            using var file = GgufFile.Open(path);

            Assert.Equal(first, file.ReadTensorAsF32("first"));
            Assert.Equal(second, file.ReadTensorAsF32("second"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CustomAlignment_OverridesDefaultDataSectionOffset()
    {
        // general.alignment overrides the default 32-byte alignment the data section start is
        // rounded up to. Using a distinctly different alignment (64) and confirming the tensor
        // still reads back correctly proves the override is actually honored, not just parsed.
        float[] expected = [7f, 8f, 9f];
        var path = new GgufFileBuilder()
            .AddMetadata("general.alignment", GgufValueType.UInt32, 64u)
            .AddTensor("weight", [3], GgmlType.F32, offset: 0, data: FloatBytes(expected))
            .WriteToTempFile();

        try
        {
            using var file = GgufFile.Open(path);

            Assert.Equal(expected, file.ReadTensorAsF32("weight"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ImplausibleMetadataKvCount_ThrowsInvalidDataException()
    {
        // A count larger than the file itself can never be valid — reject it before attempting
        // to allocate a dictionary/array sized to it (a small malformed file could otherwise
        // trigger an excessive allocation).
        var path = WriteRawFile(writer =>
        {
            writer.Write(0UL);        // tensor_count
            writer.Write(1_000_000UL); // metadata_kv_count — wildly exceeds this tiny file
        });

        try
        {
            Assert.Throws<InvalidDataException>(() => GgufFile.Open(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ImplausibleTensorDimensionCount_ThrowsInvalidDataExceptionNamingTheTensor()
    {
        var path = WriteRawFile(writer =>
        {
            writer.Write(1UL); // tensor_count
            writer.Write(0UL); // metadata_kv_count
            writer.Write(1UL); // tensor name length
            writer.Write((byte)'w');
            writer.Write(1_000_000u); // implausible dimension count for this tiny file
        });

        try
        {
            var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(path));
            Assert.Contains("w", ex.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TruncatedStringValue_ThrowsEndOfStreamException_InsteadOfSilentlyReturningAShortString()
    {
        // The declared length (10) is well within this file's total size, so the coarse
        // "count can't exceed the file" guard doesn't catch it — only actually running out of
        // bytes mid-read does. BinaryReader.ReadBytes would silently return a truncated 5-byte
        // result instead; the fix reads via Stream.ReadExactly, which throws instead.
        var path = WriteRawFile(writer =>
        {
            writer.Write(0UL); // tensor_count
            writer.Write(1UL); // metadata_kv_count
            writer.Write(1UL); // key length
            writer.Write((byte)'k');
            writer.Write((uint)GgufValueType.String);
            writer.Write(10UL);        // claims a 10-byte string value...
            writer.Write(new byte[5]); // ...but only 5 bytes actually follow before EOF
        });

        try
        {
            Assert.Throws<EndOfStreamException>(() => GgufFile.Open(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>Writes just the GGUF magic/version header, then whatever the caller supplies, to a temp file.</summary>
    private static string WriteRawFile(Action<BinaryWriter> writeBody)
    {
        var path = Path.Combine(Path.GetTempPath(), $"gguf-raw-test-{Guid.NewGuid():N}.gguf");
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);          // version
        writeBody(writer);
        return path;
    }

    private static byte[] FloatBytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] HalfBytes(float[] values)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        foreach (var v in values)
        {
            writer.Write((Half)v);
        }

        writer.Flush();
        return stream.ToArray();
    }
}
