using System.Text.Json;
using MidiDdsp.Core.Checkpoint;

namespace MidiDdsp.Tests.Checkpoint;

/// <summary>
/// Compares the C# reader against Reference/checkpoint_reference.json, which
/// tools/reference/export_checkpoint_reference.py produced with TensorFlow's
/// own checkpoint reader on the same pretrained weights.
/// </summary>
public class TfCheckpointTests
{
    private static readonly JsonElement Reference = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Reference", "checkpoint_reference.json"))).RootElement;

    [RequiresWeightsFact]
    public void ExpressionGeneratorMatchesTensorFlow() =>
        AssertMatchesReference(PretrainedWeights.ExpressionGenerator, "expression_generator");

    [RequiresWeightsFact]
    public void SynthesisGeneratorMatchesTensorFlow() =>
        AssertMatchesReference(PretrainedWeights.SynthesisGenerator, "synthesis_generator");

    [RequiresWeightsFact]
    public void GetVariableLooksUpByObjectPath()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.ExpressionGenerator);
        var info = checkpoint.GetVariable("rnn1/cell/kernel");
        Assert.Equal([262L, 384L], info.Shape);
        Assert.Throws<KeyNotFoundException>(() => checkpoint.GetVariable("no/such/variable"));
    }

    [RequiresWeightsFact]
    public void CorruptedDataFailsChecksum()
    {
        var dir = Directory.CreateTempSubdirectory("midi-ddsp-ckpt-");
        try
        {
            var prefix = Path.Combine(dir.FullName, "5000");
            File.Copy(PretrainedWeights.ExpressionGenerator + ".index", prefix + ".index");
            File.Copy(PretrainedWeights.ExpressionGenerator + ".data-00000-of-00001", prefix + ".data-00000-of-00001");

            using (var original = TfCheckpoint.Open(prefix))
            {
                var info = original.GetVariable("rnn1/cell/kernel");
                using var data = File.Open(prefix + ".data-00000-of-00001", FileMode.Open);
                data.Seek(info.Offset + 100, SeekOrigin.Begin);
                int b = data.ReadByte();
                data.Seek(-1, SeekOrigin.Current);
                data.WriteByte((byte)(b ^ 0xFF));
            }

            using var corrupted = TfCheckpoint.Open(prefix);
            var ex = Assert.Throws<InvalidDataException>(() => corrupted.ReadFloats(corrupted.GetVariable("rnn1/cell/kernel")));
            Assert.Contains("Checksum", ex.Message);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void MissingIndexThrowsFileNotFound()
    {
        Assert.Throws<FileNotFoundException>(() =>
            TfCheckpoint.Open(Path.Combine(Path.GetTempPath(), "does-not-exist", "0")));
    }

    private static void AssertMatchesReference(string prefix, string key)
    {
        using var checkpoint = TfCheckpoint.Open(prefix);
        var expected = Reference.GetProperty("checkpoints").GetProperty(key).EnumerateArray().ToList();

        Assert.Equal(
            expected.Select(e => e.GetProperty("name").GetString()!).Order(StringComparer.Ordinal),
            checkpoint.Tensors.Keys.Order(StringComparer.Ordinal));

        foreach (var entry in expected)
        {
            var name = entry.GetProperty("name").GetString()!;
            var info = checkpoint.Tensors[name];
            Assert.Equal(entry.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()), info.Shape);

            if (entry.GetProperty("dtype").GetString() != "float32")
            {
                Assert.Equal(TfDataType.String, info.DataType);
                continue;
            }

            Assert.Equal(TfDataType.Float, info.DataType);
            var values = checkpoint.ReadFloats(info);
            var head = entry.GetProperty("head").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            Assert.Equal(head, values.Take(head.Length).Select(v => (double)v));

            // Same float32 values summed in float64 in a different order: allow rounding noise.
            double sumAbs = values.Sum(v => Math.Abs((double)v));
            double tolerance = 1e-9 * Math.Max(1.0, entry.GetProperty("sum_abs").GetDouble());
            Assert.Equal(entry.GetProperty("sum_abs").GetDouble(), sumAbs, tolerance);
            Assert.Equal(entry.GetProperty("sum").GetDouble(), values.Sum(v => (double)v), tolerance);
        }
    }
}
