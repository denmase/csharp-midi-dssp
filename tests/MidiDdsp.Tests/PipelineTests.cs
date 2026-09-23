using MidiDdsp.Core;
using MidiDdsp.Core.Midi;
using MidiDdsp.Core.Models;
using MidiDdsp.Core.Nn;

namespace MidiDdsp.Tests;

/// <summary>
/// End-to-end comparison with tools/reference/export_pipeline_reference.py,
/// which wrote Reference/pipeline_test.mid, read it with pretty_midi and ran
/// the original synthesize_midi on it with argmax f0 selection.
/// </summary>
public class PipelineTests
{
    private static readonly Npz Reference = Npz.LoadReference("pipeline_reference.npz");
    private static readonly string MidiPath = Path.Combine(AppContext.BaseDirectory, "Reference", "pipeline_test.mid");

    [Fact]
    public void MidiFileIsReadLikePrettyMidi()
    {
        var midi = MidiFile.Load(MidiPath);

        Assert.Equal(Reference.Ints("pm_instrument_count")[0], midi.Instruments.Count);
        for (int i = 0; i < midi.Instruments.Count; i++)
        {
            var part = midi.Instruments[i];
            Assert.Equal(Reference.Ints($"pm{i}_program")[0], part.Program);
            Assert.Equal(Reference.Ints($"pm{i}_pitch"), part.Notes.Select(n => n.Pitch));
            Assert.Equal(Reference.Doubles($"pm{i}_start"), part.Notes.Select(n => n.Start));
            Assert.Equal(Reference.Doubles($"pm{i}_end"), part.Notes.Select(n => n.End));
        }
    }

    [RequiresWeightsFact]
    public void SynthesisMatchesOriginalWithArgmaxF0()
    {
        var synthesizer = MidiDdspSynthesizer.Load(PretrainedWeights.Directory!);

        var result = synthesizer.Synthesize(
            MidiFile.Load(MidiPath), new SynthesisOptions { F0Sampling = F0SamplingMethod.Argmax, Seed = 1 });

        Assert.Equal(Reference.Ints("parts"), result.Parts.Select(p => p.PartNumber));
        Assert.Equal(2, result.SkippedParts.Count); // guitar and drums
        foreach (var part in result.Parts)
        {
            string prefix = $"part{part.PartNumber}";
            AssertClose.Equal(Reference.Floats($"{prefix}_f0_hz"), part.HarmonicControls.F0Hz, atol: 1e-3f, rtol: 1e-5f);
            AssertClose.Equal(Reference.Floats($"{prefix}_amplitudes"), ToDb(part.HarmonicControls.Amplitudes), atol: 1e-3f, rtol: 0);
            AssertClose.Equal(Reference.Matrix($"{prefix}_harmonic_distribution"), part.HarmonicControls.HarmonicDistribution, atol: 1e-5f, rtol: 1e-4f);
            AssertClose.Equal(Reference.Floats($"{prefix}_noise_magnitudes"), ToDb(part.NoiseMagnitudes.Data), atol: 1e-3f, rtol: 0);
        }

        // The noise is random in both, so compare the level of the mix only.
        var expectedMix = Reference.Floats("mix_audio");
        Assert.Equal(expectedMix.Length, result.Mix.Length);
        Assert.InRange(Rms(result.Mix) / Rms(expectedMix), 0.95, 1.05);
    }

    [RequiresWeightsFact]
    public void SeededTopPSamplingIsReproducible()
    {
        var synthesizer = MidiDdspSynthesizer.Load(PretrainedWeights.Directory!);
        var midi = MidiFile.Load(MidiPath);
        var options = new SynthesisOptions { Seed = 11 };

        var a = synthesizer.Synthesize(midi, options);
        var b = synthesizer.Synthesize(midi, options);

        Assert.Equal(a.Mix, b.Mix);
        Assert.InRange(Rms(a.Mix) / Rms(Reference.Floats("mix_audio")), 0.8, 1.25);
    }

    /// <summary>ddsp <c>amplitude_to_db</c>: power in dB, floored at −80 dB.</summary>
    private static float[] ToDb(float[] amplitude) =>
        amplitude.Select(a => MathF.Max(10f * MathF.Log10(MathF.Max(1e-8f, a * a)), -80f)).ToArray();

    private static double Rms(float[] x) => Math.Sqrt(x.Average(v => (double)v * v));
}
