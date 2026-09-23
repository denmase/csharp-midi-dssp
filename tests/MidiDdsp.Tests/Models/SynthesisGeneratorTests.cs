using MidiDdsp.Core.Checkpoint;
using MidiDdsp.Core.Models;
using MidiDdsp.Core.Nn;

namespace MidiDdsp.Tests.Models;

/// <summary>
/// Compares the note sequence, frame conditioning and Synthesis Generator with
/// tools/reference/export_synthesis_reference.py, which ran the original
/// pipeline with argmax f0 selection.
/// </summary>
public class SynthesisGeneratorTests
{
    private static readonly Npz Reference = Npz.LoadReference("synthesis_reference.npz");
    private static readonly int InstrumentId = Reference.Ints("instrument_id")[0];

    private static List<MidiNote> Notes(string prefix)
    {
        var start = Reference.Doubles($"{prefix}_note_start");
        var end = Reference.Doubles($"{prefix}_note_end");
        var pitch = Reference.Ints($"{prefix}_note_pitch");
        return start.Select((s, i) => new MidiNote(s, end[i], pitch[i])).ToList();
    }

    [Fact]
    public void NoteSequenceMatches()
    {
        var sequence = NoteSequence.FromNotes(Notes("short"));

        Assert.Equal(Reference.Ints("short_seq_pitch"), sequence.Pitches);
        Assert.Equal(Reference.Floats("short_seq_length"), sequence.LengthSeconds);
    }

    [Fact]
    public void FrameConditioningMatches()
    {
        var sequence = NoteSequence.FromNotes(Notes("short"));
        var conditioning = FrameConditioning.Build(sequence, Reference.Matrix("short_expression"));

        AssertClose.Equal(Reference.Matrix("short_z_conditioning"), conditioning.Features, atol: 1e-6f, rtol: 1e-6f);
    }

    [Fact]
    public void FrameConditioningMatchesPastOneHundredNotes()
    {
        var sequence = NoteSequence.FromNotes(Notes("long"));
        Assert.True(sequence.Count > FrameConditioning.MaxRegions);

        var conditioning = FrameConditioning.Build(sequence, Reference.Matrix("long_expression"));

        var expected = Reference.Matrix("long_z_conditioning");
        AssertClose.Equal(expected, conditioning.Features, atol: 1e-6f, rtol: 1e-6f);
        // The original drops the position code after 100 note regions.
        int lastFrame = expected.Rows - 1 - NoteSequence.FrameRate;
        Assert.True(conditioning.QuantizedPitch[lastFrame] > 0);
        Assert.Equal(0f, conditioning.Features[lastFrame, FrameConditioning.FeatureSize - 1]);
    }

    [RequiresWeightsFact]
    public void ExpressionGeneratorMatchesOnNoteSequence()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.ExpressionGenerator);
        var sequence = NoteSequence.FromNotes(Notes("short"));

        var expression = ExpressionGenerator.Load(checkpoint)
            .Generate(sequence.Pitches, sequence.LengthSeconds, InstrumentId);

        AssertClose.Equal(Reference.Matrix("short_expression"), expression);
    }

    [RequiresWeightsFact]
    public void EncodedConditioningMatches()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var conditioning = FrameConditioning.Build(
            NoteSequence.FromNotes(Notes("short")), Reference.Matrix("short_expression"));

        var z = SynthesisGenerator.Load(checkpoint).EncodeConditioning(conditioning, InstrumentId);

        AssertClose.Equal(Reference.Matrix("short_z_midi_decoder"), z);
    }

    [RequiresWeightsFact]
    public void SynthesisParametersMatchWithArgmaxF0()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var conditioning = FrameConditioning.Build(
            NoteSequence.FromNotes(Notes("short")), Reference.Matrix("short_expression"));

        var parameters = SynthesisGenerator.Load(checkpoint)
            .Generate(conditioning, InstrumentId, new F0Sampler(F0SamplingMethod.Argmax));

        AssertClose.Equal(Reference.Floats("short_f0_midi"), parameters.F0Midi, atol: 1e-5f, rtol: 1e-6f);
        AssertClose.Equal(Reference.Floats("short_f0_hz"), parameters.F0Hz, atol: 1e-3f, rtol: 1e-5f);
        AssertClose.Equal(Reference.Floats("short_amplitudes"), parameters.Amplitudes);
        AssertClose.Equal(Reference.Matrix("short_harmonic_distribution"), parameters.HarmonicDistribution);
        AssertClose.Equal(Reference.Matrix("short_noise_magnitudes"), parameters.NoiseMagnitudes);
    }
}
