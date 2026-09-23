namespace MidiDdsp.Core.Models;

/// <summary>A MIDI note with times in seconds.</summary>
public readonly record struct MidiNote(double Start, double End, int Pitch);

/// <summary>
/// A monophonic note sequence at the 250 Hz frame rate, with rests as pitch 0:
/// the Expression Generator's input (midi_ddsp's <c>note_list_to_sequence</c>).
/// </summary>
public sealed class NoteSequence
{
    /// <summary>Frames per second of the models' control signals.</summary>
    public const int FrameRate = 250;

    /// <summary>Seconds per frame, as the original writes it (<c>0.004</c>).</summary>
    public const float FrameSeconds = 0.004f;

    public NoteSequence(int[] pitches, int[] lengthFrames)
    {
        if (pitches.Length != lengthFrames.Length)
            throw new ArgumentException("pitches and lengthFrames must have the same length.");
        Pitches = pitches;
        LengthFrames = lengthFrames;
    }

    /// <summary>MIDI pitch per note, 0 for a rest.</summary>
    public int[] Pitches { get; }

    /// <summary>Length of each note in frames.</summary>
    public int[] LengthFrames { get; }

    public int Count => Pitches.Length;

    /// <summary>Note lengths in seconds, as the Expression Generator takes them.</summary>
    public float[] LengthSeconds => LengthFrames.Select(f => f * FrameSeconds).ToArray();

    /// <summary>
    /// Builds the sequence from notes in the order given, like
    /// <c>note_list_to_sequence</c>: a rest is inserted for each gap between
    /// notes, a leading rest covers the time before the first note (unless
    /// <paramref name="removeStartSilence"/>), and a 1 second rest is appended.
    /// Overlapping notes are not trimmed, as in the original.
    /// </summary>
    public static NoteSequence FromNotes(
        IReadOnlyList<MidiNote> notes, int pitchOffset = 0, double speedRate = 1.0, bool removeStartSilence = false)
    {
        if (notes.Count == 0)
            throw new ArgumentException("At least one note is required.", nameof(notes));

        var pitches = new List<int>();
        var lengths = new List<int>();
        int previousOff = 1_000_000;
        foreach (var note in notes)
        {
            int on = TimeToFrame(note.Start / speedRate);
            int off = TimeToFrame(note.End / speedRate);
            if (on - previousOff > 0)
            {
                pitches.Add(0);
                lengths.Add(on - previousOff);
            }
            pitches.Add(note.Pitch + pitchOffset);
            lengths.Add(off - on);
            previousOff = off;
        }

        if (!removeStartSilence)
        {
            pitches.Insert(0, 0);
            lengths.Insert(0, TimeToFrame(notes[0].Start / speedRate));
        }

        pitches.Add(0);
        lengths.Add(FrameRate);
        return new NoteSequence(pitches.ToArray(), lengths.ToArray());
    }

    /// <summary>Python's <c>int(round(t * 250))</c>, which rounds halves to even.</summary>
    public static int TimeToFrame(double seconds) =>
        (int)Math.Round(seconds * FrameRate, MidpointRounding.ToEven);
}
