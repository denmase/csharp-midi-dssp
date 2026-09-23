using MidiDdsp.Core.Checkpoint;
using MidiDdsp.Core.Dsp;
using MidiDdsp.Core.Nn;

namespace MidiDdsp.Tests.Dsp;

/// <summary>
/// Compares the synthesizers with tools/reference/export_ddsp_reference.py,
/// which ran ddsp 3.2.0 on the first frames of the Synthesis Generator's
/// reference output.
/// </summary>
public class DdspSynthesisTests
{
    private static readonly Npz Reference = Npz.LoadReference("ddsp_reference.npz");
    private static readonly Npz Parameters = Npz.LoadReference("synthesis_reference.npz");
    private static readonly int Frames = Reference.Ints("num_frames")[0];

    private static float[] FirstFrames(string name) => Parameters.Floats(name).AsSpan(0, Frames).ToArray();

    private static Matrix FirstFrameRows(string name)
    {
        var m = Parameters.Matrix(name);
        return new Matrix(Frames, m.Cols, m.Data.AsSpan(0, Frames * m.Cols).ToArray());
    }

    private static HarmonicControls HarmonicControls() =>
        new HarmonicSynthesizer().GetControls(
            FirstFrames("short_amplitudes"), FirstFrameRows("short_harmonic_distribution"), FirstFrames("short_f0_hz"));

    [Fact]
    public void HarmonicControlsMatch()
    {
        var controls = HarmonicControls();

        AssertClose.Equal(Reference.Floats("amplitudes"), controls.Amplitudes, atol: 1e-7f, rtol: 1e-5f);
        AssertClose.Equal(Reference.Matrix("harmonic_distribution"), controls.HarmonicDistribution, atol: 1e-7f, rtol: 1e-5f);
    }

    [Fact]
    public void HarmonicAudioMatches()
    {
        var audio = new HarmonicSynthesizer().Synthesize(HarmonicControls());

        AssertClose.Equal(Reference.Floats("harmonic_audio"), audio, atol: 1e-6f, rtol: 1e-3f);
    }

    [Fact]
    public void FilteredNoiseMatchesOnTheSameNoise()
    {
        var synth = new FilteredNoiseSynthesizer();
        var magnitudes = synth.GetControls(FirstFrameRows("short_noise_magnitudes"));
        AssertClose.Equal(Reference.Matrix("noise_magnitudes"), magnitudes, atol: 1e-8f, rtol: 1e-5f);

        var audio = synth.Synthesize(magnitudes, Reference.Floats("white_noise"));

        AssertClose.Equal(Reference.Floats("noise_audio"), audio, atol: 1e-7f, rtol: 1e-4f);
    }

    [Fact]
    public void FilteredNoiseWithRandomNoiseHasTheReferenceLevel()
    {
        var synth = new FilteredNoiseSynthesizer();
        var audio = synth.Synthesize(synth.GetControls(FirstFrameRows("short_noise_magnitudes")), new Random(3));

        static double Rms(float[] x) => Math.Sqrt(x.Average(v => (double)v * v));
        Assert.InRange(Rms(audio) / Rms(Reference.Floats("noise_audio")), 0.9, 1.1);
    }

    [RequiresWeightsFact]
    public void ReverbMatches()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var reverb = Reverb.Load(checkpoint);
        var harmonic = Reference.Floats("harmonic_audio");
        var noise = Reference.Floats("noise_audio");
        var dry = harmonic.Select((h, i) => h + noise[i]).ToArray();

        var wet = reverb.Apply(dry, Parameters.Ints("instrument_id")[0]);

        // TensorFlow convolves with a float32 FFT of 131,072 points, which is up to
        // 2.8e-5 away from the exact convolution here (checked against float64
        // numpy.convolve); the C# double-precision FFT matches the exact result.
        AssertClose.Equal(Reference.Floats("reverb_audio"), wet, atol: 5e-5f, rtol: 1e-4f);
    }
}

public class ReverbImpulseResponseTests
{
    [RequiresWeightsFact]
    public void ImpulseResponseMasksDryTapAndDecaysAfterOneSecond()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var reverb = Reverb.Load(checkpoint);
        var raw = checkpoint.ReadFloats(checkpoint.GetVariable("reverb_module/magnitudes_embedding/embeddings"));

        var ir = reverb.ImpulseResponse(2);
        var stored = raw.AsSpan(2 * reverb.Length, reverb.Length);

        Assert.Equal(48000, ir.Length);
        Assert.Equal(0f, ir[0]);
        Assert.Equal(stored[1..16000].ToArray(), ir.AsSpan(1, 15999).ToArray());
        Assert.Equal(stored[16000], ir[16000]);                               // exp(0)
        Assert.Equal(stored[^1] * MathF.Exp(-4f), ir[^1], 1e-9f);             // exp(-4) at the end
    }
}
