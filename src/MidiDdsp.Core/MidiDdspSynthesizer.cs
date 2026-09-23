using MidiDdsp.Core.Checkpoint;
using MidiDdsp.Core.Dsp;
using MidiDdsp.Core.Midi;
using MidiDdsp.Core.Models;
using MidiDdsp.Core.Nn;

namespace MidiDdsp.Core;

public sealed record SynthesisOptions
{
    /// <summary>Semitones to transpose every note by.</summary>
    public int PitchOffset { get; init; }

    /// <summary>Playback speed; 2 plays twice as fast.</summary>
    public double SpeedRate { get; init; } = 1.0;

    public F0SamplingMethod F0Sampling { get; init; } = F0SamplingMethod.TopP;

    /// <summary>Seed for f0 sampling and noise; null for a random seed.</summary>
    public int? Seed { get; init; }
}

/// <summary>One synthesized instrument part.</summary>
public sealed record PartResult(
    int PartNumber,
    MidiInstrument Part,
    Instrument Instrument,
    Matrix Expression,
    SynthesisParameters Parameters,
    HarmonicControls HarmonicControls,
    Matrix NoiseMagnitudes,
    float[] Audio);

public sealed record MidiSynthesisResult(
    float[] Mix,
    IReadOnlyList<PartResult> Parts,
    IReadOnlyList<MidiInstrument> SkippedParts);

/// <summary>
/// MIDI file to audio with the pretrained MIDI-DDSP models, following
/// <c>midi_ddsp_synthesize.synthesize_midi</c>: each part whose program is a
/// MIDI-DDSP instrument is turned into a note sequence, given note expressions
/// by the Expression Generator, rendered to synthesis parameters by the
/// Synthesis Generator and synthesized with DDSP and the instrument's reverb.
/// Parts are zero-padded to the longest one, as in the original's batch, and
/// summed. Other parts are skipped (the original can optionally render them
/// with FluidSynth instead).
/// </summary>
public sealed class MidiDdspSynthesizer
{
    private readonly ExpressionGenerator _expressionGenerator;
    private readonly SynthesisGenerator _synthesisGenerator;
    private readonly Reverb _reverb;
    private readonly HarmonicSynthesizer _harmonic = new();
    private readonly FilteredNoiseSynthesizer _noise = new();

    public MidiDdspSynthesizer(ExpressionGenerator expressionGenerator, SynthesisGenerator synthesisGenerator, Reverb reverb)
    {
        _expressionGenerator = expressionGenerator;
        _synthesisGenerator = synthesisGenerator;
        _reverb = reverb;
    }

    /// <summary>
    /// Loads the pretrained models from the extracted
    /// <c>midi_ddsp_model_weights_urmp_9_10</c> directory.
    /// </summary>
    public static MidiDdspSynthesizer Load(string weightsDirectory)
    {
        using var expression = TfCheckpoint.Open(Path.Combine(weightsDirectory, "expression_generator", "5000"));
        using var synthesis = TfCheckpoint.Open(Path.Combine(weightsDirectory, "synthesis_generator", "50000"));
        return new MidiDdspSynthesizer(
            ExpressionGenerator.Load(expression),
            SynthesisGenerator.Load(synthesis),
            Reverb.Load(synthesis));
    }

    public MidiSynthesisResult Synthesize(MidiFile midi, SynthesisOptions? options = null)
    {
        options ??= new SynthesisOptions();
        var random = options.Seed is int seed ? new Random(seed) : new Random();

        var parts = new List<(int Number, MidiInstrument Part, Instrument Instrument, Matrix Expression, FrameConditioning Conditioning)>();
        var skipped = new List<MidiInstrument>();
        for (int i = 0; i < midi.Instruments.Count; i++)
        {
            var part = midi.Instruments[i];
            var instrument = Instruments.ForMidiProgram(part.Program);
            if (instrument is null || part.Notes.Count == 0)
            {
                skipped.Add(part);
                continue;
            }
            var sequence = NoteSequence.FromNotes(part.Notes, options.PitchOffset, options.SpeedRate);
            var expression = _expressionGenerator.Generate(sequence.Pitches, sequence.LengthSeconds, instrument.Id);
            parts.Add((i, part, instrument, expression, FrameConditioning.Build(sequence, expression)));
        }

        if (parts.Count == 0)
            return new MidiSynthesisResult([], [], skipped);

        int frames = parts.Max(p => p.Conditioning.FrameCount);
        var sampler = new F0Sampler(options.F0Sampling, random);
        var results = new List<PartResult>();
        foreach (var p in parts)
        {
            var parameters = _synthesisGenerator.Generate(p.Conditioning.PadTo(frames), p.Instrument.Id, sampler);
            var harmonicControls = _harmonic.GetControls(parameters.Amplitudes, parameters.HarmonicDistribution, parameters.F0Hz);
            var noiseMagnitudes = _noise.GetControls(parameters.NoiseMagnitudes);

            var dry = _harmonic.Synthesize(harmonicControls);
            var noise = _noise.Synthesize(noiseMagnitudes, random);
            for (int i = 0; i < dry.Length; i++)
                dry[i] += noise[i];
            var audio = _reverb.Apply(dry, p.Instrument.Id);

            results.Add(new PartResult(p.Number, p.Part, p.Instrument, p.Expression, parameters, harmonicControls, noiseMagnitudes, audio));
        }

        var mix = new float[frames * DdspMath.FrameSize];
        for (int i = 0; i < mix.Length; i++)
            mix[i] = (float)results.Sum(r => (double)r.Audio[i]);
        return new MidiSynthesisResult(mix, results, skipped);
    }
}
