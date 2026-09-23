using MidiDdsp.Core.Checkpoint;
using MidiDdsp.Core.Dsp;
#if !NETFRAMEWORK
using MidiDdsp.Core.Fallback;
#endif
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

    /// <summary>
    /// Render parts MIDI-DDSP cannot play with FluidSynth instead of skipping
    /// them (the original's <c>use_fluidsynth</c>). Needs libfluidsynth 2.x.
    /// </summary>
    public bool UseFluidSynth { get; init; }

    /// <summary>Soundfont for FluidSynth; null tries <see cref="FluidSynthRenderer.DefaultSoundFonts"/>.</summary>
    public string? SoundFontPath { get; init; }

    /// <summary>Volume of FluidSynth parts relative to full scale; the original uses 0.25.</summary>
    public float FluidSynthVolume { get; init; } = 0.25f;
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

/// <summary>A part rendered with FluidSynth.</summary>
public sealed record FallbackPartResult(int PartNumber, MidiInstrument Part, float[] Audio);

public sealed record MidiSynthesisResult(
    float[] Mix,
    IReadOnlyList<PartResult> Parts,
    IReadOnlyList<FallbackPartResult> FallbackParts,
    IReadOnlyList<MidiInstrument> SkippedParts);

/// <summary>
/// MIDI file to audio with the pretrained MIDI-DDSP models, following
/// <c>midi_ddsp_synthesize.synthesize_midi</c>: each part whose program is a
/// MIDI-DDSP instrument is turned into a note sequence, given note expressions
/// by the Expression Generator, rendered to synthesis parameters by the
/// Synthesis Generator and synthesized with DDSP and the instrument's reverb.
/// Parts are zero-padded to the longest one, as in the original's batch, and
/// summed. Other parts are skipped, or rendered with FluidSynth when
/// <see cref="SynthesisOptions.UseFluidSynth"/> is set.
/// </summary>
/// <remarks>
/// The original's FluidSynth fallback does not work as released: it raises a
/// <c>KeyError</c> for every program except 26, and it mixes pyfluidsynth's
/// 16-bit integer samples (times 0.25) with audio in [−1, 1]. Here the
/// FluidSynth output is converted to [−1, 1] before the 0.25 volume is applied.
/// </remarks>
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
        var fallback = new List<FallbackPartResult>();
        var skipped = new List<MidiInstrument>();
#if !NETFRAMEWORK
        FluidSynthRenderer? fluidSynth = null;
#endif
        for (int i = 0; i < midi.Instruments.Count; i++)
        {
            var part = midi.Instruments[i];
            var instrument = Instruments.ForMidiProgram(part.Program);
#if !NETFRAMEWORK
            // FluidSynth interop (MidiDdsp.Core.Fallback) needs .NET 7+'s LibraryImport-based
            // marshalling, with no .NET Framework equivalent -- unavailable on net472, which
            // NoteEditor doesn't need it for anyway (it already renders every non-DDSP instrument
            // through its own BASS/soundfont engine). A net472 build always falls through to the
            // plain "skip" branch below instead, exactly as if UseFluidSynth were never set.
            if (instrument is null && options.UseFluidSynth && part.Notes.Count > 0)
            {
                fluidSynth ??= FluidSynthRenderer.Create(options.SoundFontPath, DdspMath.SampleRate);
                float scale = options.FluidSynthVolume / 32768f;
                var audio = fluidSynth.Render(part).Select(v => v * scale).ToArray();
                fallback.Add(new FallbackPartResult(i, part, audio));
                continue;
            }
#endif
            if (instrument is null || part.Notes.Count == 0)
            {
                skipped.Add(part);
                continue;
            }
            var sequence = NoteSequence.FromNotes(part.Notes, options.PitchOffset, options.SpeedRate);
            var expression = _expressionGenerator.Generate(sequence.Pitches, sequence.LengthSeconds, instrument.Id);
            parts.Add((i, part, instrument, expression, FrameConditioning.Build(sequence, expression)));
        }

        int frames = parts.Count == 0 ? 0 : parts.Max(p => p.Conditioning.FrameCount);
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

        // ensure_same_length: every stem is zero-padded to the longest, then summed.
        var stems = results.Select(r => r.Audio).Concat(fallback.Select(f => f.Audio)).ToList();
        var mix = new double[stems.Count == 0 ? 0 : stems.Max(a => a.Length)];
        foreach (var stem in stems)
            for (int i = 0; i < stem.Length; i++)
                mix[i] += stem[i];
        return new MidiSynthesisResult(mix.Select(v => (float)v).ToArray(), results, fallback, skipped);
    }
}
