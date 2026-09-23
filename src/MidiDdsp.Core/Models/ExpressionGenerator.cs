using MidiDdsp.Core.Checkpoint;
using MidiDdsp.Core.Nn;

namespace MidiDdsp.Core.Models;

/// <summary>
/// MIDI-DDSP Expression Generator (<c>midi_ddsp/modules/expression_generator.py</c>):
/// predicts six note expression controls per note from a note sequence.
/// <list type="number">
/// <item>Embed pitch, note length and instrument (64 dims each) and run a bidirectional GRU(128).</item>
/// <item>For each note, feed the encoding and the previous note's output through a
/// two-layer GRU(128), layer norm and an <c>FcStackOut</c> head.</item>
/// <item>Rest notes (pitch 0) get all-zero expression.</item>
/// </list>
/// Inference is autoregressive but deterministic: the original does not sample here.
/// </summary>
public sealed class ExpressionGenerator
{
    /// <summary>Output channels, in the order of midi_ddsp's <c>CONDITIONING_KEYS</c>.</summary>
    public static readonly IReadOnlyList<string> OutputNames =
        ["volume", "vol_fluc", "vibrato", "brightness", "attack", "vol_peak_pos"];

    public const int OutputSize = 6;

    private readonly Embedding _pitchEmbedding;
    private readonly Dense _durationEmbedding;
    private readonly Embedding _instrumentEmbedding;
    private readonly BidirectionalGru _conditionRnn;
    private readonly Gru _rnn1;
    private readonly Gru _rnn2;
    private readonly DdspLayerNorm _norm;
    private readonly FcStackOut _outputHead;

    private ExpressionGenerator(
        Embedding pitchEmbedding, Dense durationEmbedding, Embedding instrumentEmbedding,
        BidirectionalGru conditionRnn, Gru rnn1, Gru rnn2, DdspLayerNorm norm, FcStackOut outputHead)
    {
        _pitchEmbedding = pitchEmbedding;
        _durationEmbedding = durationEmbedding;
        _instrumentEmbedding = instrumentEmbedding;
        _conditionRnn = conditionRnn;
        _rnn1 = rnn1;
        _rnn2 = rnn2;
        _norm = norm;
        _outputHead = outputHead;

        if (_rnn1.InputSize != _conditionRnn.OutputSize + OutputSize || _outputHead.OutputSize != OutputSize)
            throw new InvalidDataException("Expression generator weights have unexpected shapes.");
    }

    /// <summary>Loads from the pretrained <c>expression_generator/5000</c> checkpoint.</summary>
    public static ExpressionGenerator Load(TfCheckpoint checkpoint)
    {
        var root = new WeightScope(checkpoint);
        return new ExpressionGenerator(
            Embedding.Load(root["pitch_emb"]),
            Dense.Load(root["duration_emb"]),
            Embedding.Load(root["instrument_emb"]),
            BidirectionalGru.Load(root["birnn"]),
            Gru.Load(root["rnn1/cell"]),
            Gru.Load(root["rnn2/cell"]),
            DdspLayerNorm.Load(root["norm"]),
            FcStackOut.Load(root["dense_out/dense_out"], layers: 2));
    }

    /// <summary>
    /// Encodes the note sequence into the per-note conditioning, <c>[notes, 256]</c>.
    /// </summary>
    /// <param name="notePitch">MIDI pitch per note, 0 for a rest.</param>
    /// <param name="noteLength">Note length in seconds (frames at 250 Hz × 0.004).</param>
    /// <param name="instrumentId">MIDI-DDSP instrument id.</param>
    public Matrix EncodeConditioning(ReadOnlySpan<int> notePitch, ReadOnlySpan<float> noteLength, int instrumentId)
    {
        if (notePitch.Length != noteLength.Length)
            throw new ArgumentException("notePitch and noteLength must have the same length.");

        int dim = _pitchEmbedding.Dimension;
        var features = new Matrix(notePitch.Length, 3 * dim);
        var instrument = _instrumentEmbedding[instrumentId];
        for (int i = 0; i < notePitch.Length; i++)
        {
            var row = features.Row(i);
            _pitchEmbedding[notePitch[i]].CopyTo(row);
            _durationEmbedding.Forward([noteLength[i]], row.Slice(dim, dim));
            instrument.CopyTo(row[(2 * dim)..]);
        }
        return _conditionRnn.Forward(features);
    }

    /// <summary>Generates the note expression controls, <c>[notes, 6]</c>.</summary>
    public Matrix Generate(ReadOnlySpan<int> notePitch, ReadOnlySpan<float> noteLength, int instrumentId)
    {
        var conditioning = EncodeConditioning(notePitch, noteLength, instrumentId);
        var outputs = new Matrix(notePitch.Length, OutputSize);

        var h1 = new float[_rnn1.Units];
        var h2 = new float[_rnn2.Units];
        var input = new float[_rnn1.InputSize];
        Span<float> previous = stackalloc float[OutputSize]; // the zero "go" frame
        for (int i = 0; i < notePitch.Length; i++)
        {
            conditioning.Row(i).CopyTo(input);
            previous.CopyTo(input.AsSpan(conditioning.Cols));

            _rnn1.Step(input, h1);
            _rnn2.Step(h1, h2);

            var hidden = (float[])h2.Clone();
            _norm.ForwardInPlace(hidden); // a single frame, as in the original's per-step loop
            var output = outputs.Row(i);
            _outputHead.Forward(hidden, output);
            if (notePitch[i] == 0)
                output.Clear();

            output.CopyTo(previous);
        }
        return outputs;
    }
}
