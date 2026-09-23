using MidiDdsp.Core.Midi;

namespace MidiDdsp.Core.Fallback;

/// <summary>
/// Renders an instrument part with FluidSynth the way <c>pretty_midi</c>'s
/// <c>Instrument.fluidsynth()</c> does: a new synthesizer per part (gain 0.2,
/// 256 MIDI channels), the part's program on channel 0 (drums: bank 128 on
/// channel 9), events played in time order with note-offs first, rendering the
/// samples between events, and one second of tail after the last event.
/// </summary>
public sealed class FluidSynthRenderer
{
    private const double Gain = 0.2;
    private const int MidiChannels = 256;

    private FluidSynthRenderer(string soundFontPath, int sampleRate)
    {
        SoundFontPath = soundFontPath;
        SampleRate = sampleRate;
    }

    public string SoundFontPath { get; }
    public int SampleRate { get; }

    /// <summary>Soundfonts tried when none is given; the first is the original's default.</summary>
    public static readonly IReadOnlyList<string> DefaultSoundFonts =
    [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/default-GM.sf2",
        "/usr/share/sounds/sf2/TimGM6mb.sf2",
        "/usr/share/soundfonts/FluidR3_GM.sf2",
        "/usr/share/soundfonts/default.sf2",
    ];

    /// <summary>
    /// Checks that libfluidsynth and the soundfont are available.
    /// </summary>
    /// <param name="soundFontPath">A .sf2 file, or null to use the first of <see cref="DefaultSoundFonts"/> that exists.</param>
    public static FluidSynthRenderer Create(string? soundFontPath, int sampleRate)
    {
        if (!FluidSynthNative.TryLoad(out var error))
            throw new DllNotFoundException(error);
        soundFontPath ??= DefaultSoundFonts.FirstOrDefault(File.Exists)
            ?? throw new FileNotFoundException(
                $"No soundfont given and none found at {string.Join(", ", DefaultSoundFonts)}.");
        if (!File.Exists(soundFontPath))
            throw new FileNotFoundException($"Soundfont not found: {soundFontPath}", soundFontPath);
        return new FluidSynthRenderer(soundFontPath, sampleRate);
    }

    public static bool IsLibraryAvailable => FluidSynthNative.TryLoad(out _);

    /// <summary>
    /// Renders the part's left channel in 16-bit sample units (−32768 to 32767,
    /// as pyfluidsynth returns them), or an empty array for a part without notes.
    /// </summary>
    public float[] Render(MidiInstrument part)
    {
        if (part.Notes.Count == 0)
            return [];

        var events = BuildEvents(part);
        double currentTime = events[0].Time;
        // Turn times into the gap to the next event; the last event gets 1 s.
        var deltas = new double[events.Count];
        for (int i = 0; i < events.Count - 1; i++)
            deltas[i] = events[i + 1].Time - events[i].Time;
        deltas[^1] = 1.0;
        double totalTime = currentTime + deltas.Sum();

        var settings = FluidSynthNative.NewSettings();
        var synth = IntPtr.Zero;
        try
        {
            FluidSynthNative.SettingsSetNum(settings, "synth.gain", Gain);
            FluidSynthNative.SettingsSetNum(settings, "synth.sample-rate", SampleRate);
            FluidSynthNative.SettingsSetInt(settings, "synth.midi-channels", MidiChannels);
            synth = FluidSynthNative.NewSynth(settings);
            if (synth == IntPtr.Zero)
                throw new InvalidOperationException("FluidSynth could not create a synthesizer.");

            int soundFont = FluidSynthNative.SoundFontLoad(synth, SoundFontPath, 0);
            if (soundFont < 0)
                throw new InvalidDataException($"FluidSynth could not load soundfont {SoundFontPath}.");

            int channel = part.IsDrum ? 9 : 0;
            if (part.IsDrum)
            {
                if (FluidSynthNative.ProgramSelect(synth, channel, soundFont, 128, part.Program) == -1)
                    FluidSynthNative.ProgramSelect(synth, channel, soundFont, 128, 0);
            }
            else
            {
                FluidSynthNative.ProgramSelect(synth, channel, soundFont, 0, part.Program);
            }

            var output = new float[(int)Math.Ceiling(SampleRate * totalTime)];
            var buffer = Array.Empty<short>();
            for (int i = 0; i < events.Count; i++)
            {
                var e = events[i];
                switch (e.Type)
                {
                    case EventType.NoteOn:
                        if (e.Data1 is >= 0 and <= 128 && e.Data2 is >= 0 and <= 128)
                            FluidSynthNative.NoteOn(synth, channel, e.Data1, e.Data2);
                        break;
                    case EventType.NoteOff:
                        FluidSynthNative.NoteOff(synth, channel, e.Data1);
                        break;
                    case EventType.PitchBend:
                        FluidSynthNative.PitchBend(synth, channel, e.Data1 + 8192);
                        break;
                    case EventType.ControlChange:
                        FluidSynthNative.ControlChange(synth, channel, e.Data1, e.Data2);
                        break;
                }

                int start = (int)(SampleRate * currentTime);
                int end = (int)(SampleRate * (currentTime + deltas[i]));
                int count = end - start;
                if (count > 0)
                {
                    if (buffer.Length < 2 * count)
                        buffer = new short[2 * count];
                    Write(synth, count, buffer);
                    if (end > output.Length)
                        Array.Resize(ref output, end);
                    for (int s = 0; s < count; s++)
                        output[start + s] += buffer[2 * s];
                }
                currentTime += deltas[i];
            }
            return output;
        }
        finally
        {
            if (synth != IntPtr.Zero)
                FluidSynthNative.DeleteSynth(synth);
            FluidSynthNative.DeleteSettings(settings);
        }
    }

    private static unsafe void Write(IntPtr synth, int count, short[] interleaved)
    {
        fixed (short* p = interleaved)
            FluidSynthNative.WriteS16(synth, count, p, 0, 2, p, 1, 2);
    }

    private enum EventType { NoteOff, NoteOn, PitchBend, ControlChange }

    private readonly record struct Event(double Time, EventType Type, int Data1, int Data2);

    /// <summary>Notes (on, off), then pitch bends, then control changes; stably sorted by time with note-offs first.</summary>
    private static List<Event> BuildEvents(MidiInstrument part)
    {
        var events = new List<Event>();
        foreach (var note in part.Notes)
        {
            events.Add(new Event(note.Start, EventType.NoteOn, note.Pitch, note.Velocity));
            events.Add(new Event(note.End, EventType.NoteOff, note.Pitch, 0));
        }
        foreach (var (time, pitch) in part.PitchBends)
            events.Add(new Event(time, EventType.PitchBend, pitch, 0));
        foreach (var (time, number, value) in part.ControlChanges)
            events.Add(new Event(time, EventType.ControlChange, number, value));
        return events.OrderBy(e => e.Time).ThenBy(e => e.Type != EventType.NoteOff).ToList();
    }
}
