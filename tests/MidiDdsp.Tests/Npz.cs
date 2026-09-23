using System.IO.Compression;
using System.Text.RegularExpressions;
using MidiDdsp.Core.Nn;

namespace MidiDdsp.Tests;

/// <summary>Reads the little-endian float32 / int64 arrays in a NumPy .npz file.</summary>
internal sealed partial class Npz
{
    private readonly Dictionary<string, (string Dtype, int[] Shape, byte[] Data)> _arrays = [];

    public Npz(string path)
    {
        using var zip = ZipFile.OpenRead(path);
        foreach (var entry in zip.Entries)
        {
            using var stream = entry.Open();
            using var buffer = new MemoryStream();
            stream.CopyTo(buffer);
            _arrays[Path.GetFileNameWithoutExtension(entry.Name)] = ParseNpy(buffer.ToArray());
        }
    }

    public static Npz LoadReference(string name) =>
        new(Path.Combine(AppContext.BaseDirectory, "Reference", name));

    public float[] Floats(string name)
    {
        var (dtype, _, data) = Get(name, "<f4");
        var values = new float[data.Length / 4];
        Buffer.BlockCopy(data, 0, values, 0, data.Length);
        return values;
    }

    public int[] Ints(string name)
    {
        var (_, _, data) = Get(name, "<i8");
        return Enumerable.Range(0, data.Length / 8).Select(i => checked((int)BitConverter.ToInt64(data, i * 8))).ToArray();
    }

    /// <summary>A 2-D float32 array as a matrix.</summary>
    public Matrix Matrix(string name)
    {
        var (_, shape, _) = Get(name, "<f4");
        if (shape.Length != 2)
            throw new InvalidDataException($"'{name}' has rank {shape.Length}, expected 2.");
        return new Matrix(shape[0], shape[1], Floats(name));
    }

    private (string, int[], byte[]) Get(string name, string dtype)
    {
        var array = _arrays[name];
        if (array.Dtype != dtype)
            throw new InvalidDataException($"'{name}' is {array.Dtype}, expected {dtype}.");
        return array;
    }

    private static (string, int[], byte[]) ParseNpy(byte[] bytes)
    {
        if (bytes[0] != 0x93 || System.Text.Encoding.ASCII.GetString(bytes, 1, 5) != "NUMPY")
            throw new InvalidDataException("Not a .npy file.");
        int major = bytes[6];
        int headerLength = major == 1 ? BitConverter.ToUInt16(bytes, 8) : (int)BitConverter.ToUInt32(bytes, 8);
        int headerStart = major == 1 ? 10 : 12;
        var header = System.Text.Encoding.ASCII.GetString(bytes, headerStart, headerLength);

        var dtype = DescrRegex().Match(header).Groups[1].Value;
        if (header.Contains("'fortran_order': True"))
            throw new NotSupportedException("Fortran-ordered arrays are not supported.");
        var shape = ShapeRegex().Match(header).Groups[1].Value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(int.Parse).ToArray();
        return (dtype, shape, bytes[(headerStart + headerLength)..]);
    }

    [GeneratedRegex(@"'descr':\s*'([^']*)'")]
    private static partial Regex DescrRegex();

    [GeneratedRegex(@"'shape':\s*\(([^)]*)\)")]
    private static partial Regex ShapeRegex();
}

internal static class AssertClose
{
    /// <summary>Asserts |actual - expected| &lt;= atol + rtol·|expected| element-wise.</summary>
    public static void Equal(float[] expected, float[] actual, float atol = 1e-4f, float rtol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0;
        int worstIndex = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float excess = MathF.Abs(actual[i] - expected[i]) - (atol + rtol * MathF.Abs(expected[i]));
            if (excess > worst || float.IsNaN(actual[i]))
            {
                worst = float.IsNaN(actual[i]) ? float.PositiveInfinity : excess;
                worstIndex = i;
            }
        }
        Assert.True(worstIndex < 0,
            worstIndex < 0 ? "" : $"Mismatch at index {worstIndex}: expected {expected[worstIndex]}, got {actual[worstIndex]}.");
    }

    public static void Equal(Matrix expected, Matrix actual, float atol = 1e-4f, float rtol = 1e-4f)
    {
        Assert.Equal((expected.Rows, expected.Cols), (actual.Rows, actual.Cols));
        Equal(expected.Data, actual.Data, atol, rtol);
    }
}
