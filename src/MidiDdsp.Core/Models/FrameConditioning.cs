using MidiDdsp.Core.Nn;

namespace MidiDdsp.Core.Models;

/// <summary>
/// Frame-level input of the Synthesis Generator, built from a note sequence
/// and its note expression controls. Reproduces midi_ddsp's
/// <c>expression_generator_output_to_conditioning_df</c>,
/// <c>conditioning_df_to_dict</c>, <c>conditioning_df_to_midi_features</c> and
/// the <c>'index_length'</c> position code of <c>ExpressionMidiDecoder</c>,
/// including their edge cases (see remarks).
/// </summary>
/// <remarks>
/// Faithfully kept quirks of the original:
/// <list type="bullet">
/// <item>Each note's values are written to frames <c>[onset, offset]</c> inclusive;
/// the next note then overwrites its onset frame.</item>
/// <item>The offset flag is set at <c>offset - 1</c>; for a zero-length note at
/// frame 0 that is NumPy index -1, the last frame (which the trailing rest
/// flags anyway).</item>
/// <item>Only the first 100 note regions get a relative position; later
/// frames get 0 (ddsp's <c>get_note_mask_from_onset</c> <c>max_regions</c>).</item>
/// </list>
/// </remarks>
public sealed class FrameConditioning
{
    /// <summary>Maximum note regions ddsp's note mask tracks.</summary>
    public const int MaxRegions = 100;

    /// <summary>Feature columns: 6 expression controls, pitch/127, onset, offset, relative position.</summary>
    public const int FeatureSize = ExpressionGenerator.OutputSize + 4;

    private FrameConditioning(Matrix features, float[] quantizedPitch)
    {
        Features = features;
        QuantizedPitch = quantizedPitch;
    }

    /// <summary>The <c>z_conditioning</c> input of the preconditioning stack, <c>[frames, 10]</c>.</summary>
    public Matrix Features { get; }

    /// <summary>MIDI pitch per frame (<c>q_pitch</c>), 0 in rests.</summary>
    public float[] QuantizedPitch { get; }

    public int FrameCount => QuantizedPitch.Length;

    /// <summary>
    /// Zero-pads to <paramref name="frames"/> frames, as the original does to
    /// batch parts of different lengths (<c>ensure_same_length</c>).
    /// </summary>
    public FrameConditioning PadTo(int frames)
    {
        if (frames < FrameCount)
            throw new ArgumentOutOfRangeException(nameof(frames), "Cannot pad to fewer frames.");
        var features = new Matrix(frames, FeatureSize);
        Features.Data.CopyTo(features.Data, 0);
        var pitch = new float[frames];
        QuantizedPitch.CopyTo(pitch, 0);
        return new FrameConditioning(features, pitch);
    }

    /// <param name="notes">The note sequence given to the Expression Generator.</param>
    /// <param name="expression">Expression Generator output, <c>[notes, 6]</c>; clipped to [0, 1] here.</param>
    public static FrameConditioning Build(NoteSequence notes, Matrix expression)
    {
        if (expression.Rows != notes.Count || expression.Cols != ExpressionGenerator.OutputSize)
            throw new ArgumentException("Expression must have one row of 6 values per note.");

        // Note boundaries, from the lengths in seconds as the original round-trips them.
        var lengthSeconds = notes.LengthSeconds;
        var onsetFrames = new int[notes.Count];
        var offsetFrames = new int[notes.Count];
        int onset = 0;
        for (int i = 0; i < notes.Count; i++)
        {
            int length = (int)Math.Round(lengthSeconds[i] / 0.004, MidpointRounding.ToEven);
            onsetFrames[i] = onset;
            offsetFrames[i] = onset + length;
            onset += length;
        }

        int frames = offsetFrames[^1];
        if (frames <= 0)
            throw new ArgumentException("The note sequence has no frames.");

        int n = ExpressionGenerator.OutputSize;
        var features = new Matrix(frames, FeatureSize);
        var qPitch = new float[frames];
        var onsets = new bool[frames];
        var offsets = new bool[frames];
        for (int i = 0; i < notes.Count; i++)
        {
            int on = onsetFrames[i], off = offsetFrames[i];
            var controls = expression.Row(i);
            for (int t = on; t <= off && t < frames; t++)
            {
                var row = features.Row(t);
                for (int c = 0; c < n; c++)
                    row[c] = Math.Max(0f, Math.Min(1f, controls[c]));
                qPitch[t] = notes.Pitches[i];
            }
            if (on < frames)
                onsets[on] = true;
            offsets[off - 1 >= 0 ? off - 1 : frames + off - 1] = true;
        }

        var relativePosition = RelativePosition(qPitch, onsets);
        for (int t = 0; t < frames; t++)
        {
            var row = features.Row(t);
            row[n] = qPitch[t] / 127f;
            row[n + 1] = onsets[t] ? 1f : 0f;
            row[n + 2] = offsets[t] ? 1f : 0f;
            row[n + 3] = relativePosition[t];
        }
        return new FrameConditioning(features, qPitch);
    }

    /// <summary>
    /// Position of each sounding frame within its note, <c>index / length</c>
    /// with both counted in sounding frames of the note's region (1-based).
    /// </summary>
    private static float[] RelativePosition(float[] qPitch, bool[] onsets)
    {
        int frames = qPitch.Length;
        var region = new int[frames];
        int current = -1;
        for (int t = 0; t < frames; t++)
        {
            if (t == 0 || onsets[t])
                current++;
            region[t] = current;
        }

        bool Sounding(int t) => qPitch[t] > 0 && region[t] < MaxRegions;

        var regionLength = new int[current + 1];
        for (int t = 0; t < frames; t++)
            if (Sounding(t))
                regionLength[region[t]]++;

        var result = new float[frames];
        var index = new int[current + 1];
        for (int t = 0; t < frames; t++)
            if (Sounding(t))
                result[t] = (float)++index[region[t]] / regionLength[region[t]];
        return result;
    }
}
