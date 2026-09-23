using MidiDdsp.Core;
using MidiDdsp.Core.Dsp;
using MidiDdsp.Core.Fallback;
using MidiDdsp.Core.Midi;
using MidiDdsp.Core.Models;

namespace MidiDdsp.Tests.Fallback;

/// <summary>
/// Locates a soundfont for FluidSynth tests: MIDI_DDSP_SOUNDFONT, or the
/// TimGM6mb.sf2 the reference was rendered with (Debian/Ubuntu package
/// timgm6mb-soundfont).
/// </summary>
internal static class TestSoundFont
{
    public static string? Path { get; } = Find();

    private static string? Find()
    {
        var path = Environment.GetEnvironmentVariable("MIDI_DDSP_SOUNDFONT");
        if (string.IsNullOrEmpty(path))
            path = "/usr/share/sounds/sf2/TimGM6mb.sf2";
        return File.Exists(path) ? path : null;
    }
}

/// <summary>A fact skipped unless libfluidsynth and the test soundfont are available.</summary>
public sealed class RequiresFluidSynthFactAttribute : FactAttribute
{
    public RequiresFluidSynthFactAttribute()
    {
        if (!FluidSynthRenderer.IsLibraryAvailable)
            Skip = "libfluidsynth not found; install FluidSynth 2.x or set MIDI_DDSP_FLUIDSYNTH_LIBRARY.";
        else if (TestSoundFont.Path is null)
            Skip = "Soundfont not found; install timgm6mb-soundfont or set MIDI_DDSP_SOUNDFONT.";
    }
}

/// <summary>
/// Compares with tools/reference/export_fluidsynth_reference.py, which read
/// Reference/fluidsynth_test.mid with pretty_midi and rendered each part with
/// Instrument.fluidsynth() (pyfluidsynth 1.3.0, libfluidsynth 2.3.4, TimGM6mb).
/// </summary>
public class FluidSynthTests
{
    private static readonly Npz Reference = Npz.LoadReference("fluidsynth_reference.npz");
    private static readonly string MidiPath = System.IO.Path.Combine(AppContext.BaseDirectory, "Reference", "fluidsynth_test.mid");

    [Fact]
    public void ControlEventsAreReadLikePrettyMidi()
    {
        var midi = MidiFile.Load(MidiPath);

        Assert.Equal(Reference.Ints("instrument_count")[0], midi.Instruments.Count);
        for (int i = 0; i < midi.Instruments.Count; i++)
        {
            var part = midi.Instruments[i];
            Assert.Equal(Reference.Ints($"i{i}_program")[0], part.Program);
            Assert.Equal(Reference.Ints($"i{i}_is_drum")[0] == 1, part.IsDrum);
            Assert.Equal(Reference.Ints($"i{i}_velocity"), part.Notes.Select(n => n.Velocity));
            Assert.Equal(Reference.Doubles($"i{i}_bend_time"), part.PitchBends.Select(b => b.Time));
            Assert.Equal(Reference.Ints($"i{i}_bend_pitch"), part.PitchBends.Select(b => b.Pitch));
            Assert.Equal(Reference.Doubles($"i{i}_cc_time"), part.ControlChanges.Select(c => c.Time));
            Assert.Equal(Reference.Ints($"i{i}_cc_number"), part.ControlChanges.Select(c => c.Number));
            Assert.Equal(Reference.Ints($"i{i}_cc_value"), part.ControlChanges.Select(c => c.Value));
        }
    }

    [RequiresFluidSynthFact]
    public void RenderingMatchesPrettyMidi()
    {
        var midi = MidiFile.Load(MidiPath);
        var renderer = FluidSynthRenderer.Create(TestSoundFont.Path, DdspMath.SampleRate);

        for (int i = 0; i < midi.Instruments.Count; i++)
        {
            var expected = Reference.Floats($"i{i}_audio");
            var actual = renderer.Render(midi.Instruments[i]);
            Assert.Equal(expected.Length, actual.Length);
            // Same library and events; only FluidSynth's 16-bit dither noise, drawn
            // from a per-process random table, differs (by at most 2 LSB here,
            // against peaks of 700-2300).
            AssertClose.Equal(expected, actual, atol: 2f, rtol: 0);
        }
    }

    [RequiresFluidSynthFact]
    public void MissingSoundFontIsReported()
    {
        Assert.Throws<FileNotFoundException>(() =>
            FluidSynthRenderer.Create("/does/not/exist.sf2", DdspMath.SampleRate));
    }

    [RequiresWeightsFact]
    public void PipelineRendersOtherInstrumentsWithFluidSynth()
    {
        if (!FluidSynthRenderer.IsLibraryAvailable || TestSoundFont.Path is null)
            return;
        var synthesizer = MidiDdspSynthesizer.Load(PretrainedWeights.Directory!);
        var midi = MidiFile.Load(System.IO.Path.Combine(AppContext.BaseDirectory, "Reference", "pipeline_test.mid"));

        var result = synthesizer.Synthesize(midi, new SynthesisOptions
        {
            F0Sampling = F0SamplingMethod.Argmax,
            Seed = 1,
            UseFluidSynth = true,
            SoundFontPath = TestSoundFont.Path,
        });

        Assert.Equal([0, 1], result.Parts.Select(p => p.PartNumber));
        Assert.Equal([2, 3], result.FallbackParts.Select(p => p.PartNumber)); // guitar, drums
        Assert.Empty(result.SkippedParts);

        var renderer = FluidSynthRenderer.Create(TestSoundFont.Path, DdspMath.SampleRate);
        foreach (var fallback in result.FallbackParts)
        {
            var raw = renderer.Render(fallback.Part);
            Assert.Equal(raw.Select(v => v * 0.25f / 32768f), fallback.Audio);
            Assert.InRange(fallback.Audio.Max(Math.Abs), 1e-3f, 0.25f);
        }

        int length = result.Parts.Select(p => p.Audio.Length).Concat(result.FallbackParts.Select(p => p.Audio.Length)).Max();
        Assert.Equal(length, result.Mix.Length);
        for (int i = 0; i < length; i += 997)
        {
            double sum = result.Parts.Sum(p => i < p.Audio.Length ? p.Audio[i] : 0.0)
                + result.FallbackParts.Sum(p => i < p.Audio.Length ? p.Audio[i] : 0.0);
            Assert.Equal(sum, result.Mix[i], 1e-6);
        }
    }
}
