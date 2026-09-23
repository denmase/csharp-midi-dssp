using System.Diagnostics;
using MidiDdsp.Core;
using MidiDdsp.Core.Audio;
using MidiDdsp.Core.Checkpoint;
using MidiDdsp.Core.Dsp;
using MidiDdsp.Core.Midi;
using MidiDdsp.Core.Models;

const string WeightsFolder = "midi_ddsp_model_weights_urmp_9_10";

try
{
    return args.FirstOrDefault() switch
    {
        "synthesize" => Synthesize(args[1..]),
        "list-weights" when args.Length == 2 => ListWeights(args[1]),
        _ => Usage(),
    };
}
catch (Exception e) when (e is IOException or DllNotFoundException or InvalidDataException
                           or NotSupportedException or FormatException or ArgumentException)
{
    Console.Error.WriteLine($"error: {e.Message}");
    return 1;
}

// Places the pretrained weights are looked for when --weights is not given.
static IEnumerable<string> WeightCandidates()
{
    if (Environment.GetEnvironmentVariable("MIDI_DDSP_WEIGHTS") is { Length: > 0 } fromEnv)
        yield return fromEnv;
    yield return Path.Combine(AppContext.BaseDirectory, "weights", WeightsFolder);
    yield return Path.Combine(Directory.GetCurrentDirectory(), "weights", WeightsFolder);
}

static bool HasWeights(string dir) =>
    File.Exists(Path.Combine(dir, "expression_generator", "5000.index")) &&
    File.Exists(Path.Combine(dir, "synthesis_generator", "50000.index"));

static int Usage()
{
    Console.Error.WriteLine("""
        Usage:
          midi-ddsp synthesize <input.mid> <output.wav> [options]
              --weights <dir>      pretrained weights (default: $MIDI_DDSP_WEIGHTS, then
                                   weights/midi_ddsp_model_weights_urmp_9_10 next to this
                                   program, then in the current directory)
              --seed <n>           seed for f0 sampling and noise
              --argmax             pick the most likely f0 instead of top-p sampling
              --pitch-offset <n>   transpose by n semitones
              --speed <rate>       playback speed (default 1)
              --stems <dir>        also write each part to <dir>/<part>_<instrument>.wav
              --fluidsynth         render other instruments with FluidSynth instead of skipping them
              --soundfont <sf2>    soundfont for --fluidsynth (default: a .sf2 file next to this program
                                   or in its soundfonts folder, else a GM soundfont in
                                   /usr/share/sounds/sf2)
              --float              write 32-bit float WAV instead of 16-bit PCM
          midi-ddsp list-weights <checkpoint-prefix>
        """);
    return 1;
}

static int Synthesize(string[] args)
{
    var positional = new List<string>();
    string? weights = null;
    string? stemsDir = null;
    bool asFloat = false;
    var options = new SynthesisOptions();
    for (int i = 0; i < args.Length; i++)
    {
        string Next() => ++i < args.Length ? args[i] : throw new ArgumentException($"{args[i - 1]} needs a value.");
        switch (args[i])
        {
            case "--weights": weights = Next(); break;
            case "--seed": options = options with { Seed = int.Parse(Next()) }; break;
            case "--argmax": options = options with { F0Sampling = F0SamplingMethod.Argmax }; break;
            case "--pitch-offset": options = options with { PitchOffset = int.Parse(Next()) }; break;
            case "--speed": options = options with { SpeedRate = double.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture) }; break;
            case "--stems": stemsDir = Next(); break;
            case "--fluidsynth": options = options with { UseFluidSynth = true }; break;
            case "--soundfont": options = options with { SoundFontPath = Next() }; break;
            case "--float": asFloat = true; break;
            default:
                if (args[i].StartsWith("--"))
                    return Usage();
                positional.Add(args[i]);
                break;
        }
    }
    if (positional.Count != 2)
        return Usage();

    void Write(string path, float[] audio)
    {
        if (asFloat)
            WavWriter.WriteFloat32(path, audio, DdspMath.SampleRate);
        else
            WavWriter.WritePcm16(path, audio, DdspMath.SampleRate);
    }

    weights ??= WeightCandidates().FirstOrDefault(HasWeights);
    if (weights is null)
    {
        Console.Error.WriteLine("error: pretrained weights not found. Looked in:");
        foreach (var candidate in WeightCandidates())
            Console.Error.WriteLine($"  {candidate}");
        Console.Error.WriteLine("Download them with tools/download_weights.sh, or pass --weights <dir>.");
        return 1;
    }

    var watch = Stopwatch.StartNew();
    var synthesizer = MidiDdspSynthesizer.Load(weights);
    var midi = MidiFile.Load(positional[0]);
    var result = synthesizer.Synthesize(midi, options);

    static string Describe(MidiInstrument part) => $"program {part.Program}{(part.IsDrum ? " (drums)" : "")}";

    foreach (var skipped in result.SkippedParts)
        Console.WriteLine($"Skipping part with {Describe(skipped)}: not a MIDI-DDSP instrument (use --fluidsynth to render it).");
    if (result.Parts.Count == 0 && result.FallbackParts.Count == 0)
    {
        Console.Error.WriteLine("No part of this MIDI file uses a MIDI-DDSP instrument.");
        return 2;
    }

    foreach (var part in result.Parts)
        Console.WriteLine($"Part {part.PartNumber}: {part.Instrument.Name}, {part.Part.Notes.Count} notes");
    foreach (var part in result.FallbackParts)
        Console.WriteLine($"Part {part.PartNumber}: {Describe(part.Part)} with FluidSynth, {part.Part.Notes.Count} notes");
    Write(positional[1], result.Mix);
    if (stemsDir is not null)
    {
        Directory.CreateDirectory(stemsDir);
        foreach (var part in result.Parts)
            Write(Path.Combine(stemsDir, $"{part.PartNumber}_{part.Instrument.Name}.wav"), part.Audio);
        foreach (var part in result.FallbackParts)
            Write(Path.Combine(stemsDir, $"{part.PartNumber}_program{part.Part.Program}_fluidsynth.wav"), part.Audio);
    }

    double seconds = result.Mix.Length / (double)DdspMath.SampleRate;
    Console.WriteLine($"Wrote {seconds:F1} s of audio to {positional[1]} in {watch.Elapsed.TotalSeconds:F1} s.");
    return 0;
}

static int ListWeights(string prefix)
{
    using var checkpoint = TfCheckpoint.Open(prefix);
    long totalParams = 0;
    foreach (var info in checkpoint.Tensors.Values.OrderBy(t => t.Name, StringComparer.Ordinal))
    {
        if (info.DataType == TfDataType.String)
            continue; // object graph metadata, not a weight
        var name = info.Name.EndsWith(TfCheckpoint.VariableSuffix)
            ? info.Name[..^TfCheckpoint.VariableSuffix.Length]
            : info.Name;
        Console.WriteLine($"{name}  {info.DataType}  [{string.Join(", ", info.Shape)}]");
        totalParams += info.ElementCount;
    }
    Console.WriteLine($"{totalParams:N0} parameters");
    return 0;
}
