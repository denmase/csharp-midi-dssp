namespace MidiDdsp.Core.Models;

/// <summary>A MIDI-DDSP instrument: its model id and General MIDI program.</summary>
public sealed record Instrument(int Id, string Name, string Abbreviation, int MidiProgram);

/// <summary>
/// The instruments of the pretrained model (URMP), from
/// <c>midi_ddsp/data_handling/instrument_name_utils.py</c>. MIDI programs are 0-based.
/// </summary>
public static class Instruments
{
    public static readonly IReadOnlyList<Instrument> All =
    [
        new(0, "violin", "vn", 40),
        new(1, "viola", "va", 41),
        new(2, "cello", "vc", 42),
        new(3, "double bass", "db", 43),
        new(4, "flute", "fl", 73),
        new(5, "oboe", "ob", 68),
        new(6, "clarinet", "cl", 71),
        new(7, "saxophone", "sax", 66),
        new(8, "bassoon", "bn", 70),
        new(9, "trumpet", "tpt", 56),
        new(10, "horn", "hn", 60),
        new(11, "trombone", "tbn", 57),
        new(12, "tuba", "tba", 58),
    ];

    /// <summary>
    /// The instrument MIDI-DDSP synthesizes for a program, or null if it has none.
    /// The original's "guitar" (program 26) is excluded, as in <c>synthesize_midi</c>.
    /// </summary>
    public static Instrument? ForMidiProgram(int program) =>
        All.FirstOrDefault(i => i.MidiProgram == program);

    public static Instrument? ForName(string nameOrAbbreviation) =>
        All.FirstOrDefault(i =>
            string.Equals(i.Name, nameOrAbbreviation, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(i.Abbreviation, nameOrAbbreviation, StringComparison.OrdinalIgnoreCase));
}
