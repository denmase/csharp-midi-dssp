using MidiDdsp.Core.Checkpoint;
using MidiDdsp.Core.Nn;

namespace MidiDdsp.Core.Models;

/// <summary>
/// Raw per-frame outputs of the Synthesis Generator, before DDSP's scaling
/// (<c>exp_sigmoid</c>) is applied by the synthesizers.
/// </summary>
public sealed record SynthesisParameters(
    float[] F0Midi,
    float[] F0Hz,
    float[] Amplitudes,
    Matrix HarmonicDistribution,
    Matrix NoiseMagnitudes)
{
    public int FrameCount => F0Hz.Length;
}

/// <summary>
/// The inference path of MIDI-DDSP's Synthesis Generator: the
/// <c>ExpressionMidiDecoder</c> with a <c>MidiToSynthAutoregDecoder</c>
/// (<c>midi_ddsp/modules/midi_decoder.py</c>, <c>synth_params_decoder.py</c>).
/// <list type="number">
/// <item>Frame conditioning (10 features) → <c>FcStackOut(512 × 5 → 256)</c>, plus a 64-dim instrument embedding.</item>
/// <item>f0: bidirectional GRU(256), then an autoregressive two-layer GRU(256) that picks
/// one of 201 pitch-deviation bins (−1 to +1 semitone in 0.01 steps) per frame.</item>
/// <item>Amplitude, harmonic distribution (60) and noise magnitudes (65): a dilated conv
/// stack over the conditioning plus an embedding of the generated f0.</item>
/// </list>
/// The audio encoder used in training (DDSP Inference) is not needed and not loaded.
/// </summary>
public sealed class SynthesisGenerator
{
    public const int PitchDeviationBins = 201;
    public const int NumHarmonics = 60;
    public const int NumNoiseBands = 65;

    private readonly FcStackOut _preconditioning;
    private readonly Embedding _instrumentEmbedding;
    private readonly BidirectionalGru _f0ConditionRnn;
    private readonly Gru _f0Rnn1;
    private readonly Gru _f0Rnn2;
    private readonly DdspLayerNorm _f0Norm;
    private readonly Dense _f0Out;
    private readonly Dense _pitchEmbedding;
    private readonly DilatedConvStack _harmonicNet;
    private readonly DdspLayerNorm _harmonicNorm;
    private readonly Dense _harmonicOut;

    private SynthesisGenerator(WeightScope midiDecoder)
    {
        var decoder = midiDecoder["decoder"];
        var f0 = decoder["midi_to_f0"];
        var harmonic = decoder["midi_f0_to_harmonic"];

        _preconditioning = FcStackOut.Load(midiDecoder["z_preconditioning_stack"], layers: 5);
        _instrumentEmbedding = Embedding.Load(midiDecoder["instrument_emb"]);
        _f0ConditionRnn = BidirectionalGru.Load(f0["birnn"]);
        _f0Rnn1 = Gru.Load(f0["rnn1/cell"]);
        _f0Rnn2 = Gru.Load(f0["rnn2/cell"]);
        _f0Norm = DdspLayerNorm.Load(f0["norm"]);
        _f0Out = Dense.Load(f0["dense_out"]);
        _pitchEmbedding = Dense.Load(decoder["q_pitch_emb"]);
        _harmonicNet = DilatedConvStack.Load(harmonic["net"], stacks: 4, layersPerStack: 5);
        _harmonicNorm = DdspLayerNorm.Load(harmonic["norm"]);
        _harmonicOut = Dense.Load(harmonic["dense_out"]);

        if (_preconditioning.InputSize != FrameConditioning.FeatureSize
            || _f0Out.OutputSize != PitchDeviationBins
            || _f0Rnn1.InputSize != _f0ConditionRnn.OutputSize + PitchDeviationBins
            || _harmonicOut.OutputSize != 1 + NumHarmonics + NumNoiseBands)
            throw new InvalidDataException("Synthesis generator weights have unexpected shapes.");
    }

    /// <summary>Loads from the pretrained <c>synthesis_generator/50000</c> checkpoint.</summary>
    public static SynthesisGenerator Load(TfCheckpoint checkpoint) =>
        new(new WeightScope(checkpoint, "midi_decoder"));

    /// <summary>
    /// The latent sequence <c>z_midi_decoder</c>: preconditioned features plus
    /// the instrument embedding, <c>[frames, 320]</c>.
    /// </summary>
    public Matrix EncodeConditioning(FrameConditioning conditioning, int instrumentId)
    {
        var z = _preconditioning.Forward(conditioning.Features);
        var instrument = new Matrix(z.Rows, _instrumentEmbedding.Dimension);
        var embedding = _instrumentEmbedding[instrumentId];
        for (int t = 0; t < z.Rows; t++)
            embedding.CopyTo(instrument.Row(t));
        return Matrix.ConcatColumns(z, instrument);
    }

    public SynthesisParameters Generate(FrameConditioning conditioning, int instrumentId, F0Sampler sampler)
    {
        var z = EncodeConditioning(conditioning, instrumentId);
        var f0Midi = GenerateF0Midi(z, conditioning.QuantizedPitch, sampler);

        // Harmonic head input: z plus an embedding of the generated pitch.
        var pitch = new Matrix(f0Midi.Length, 1, f0Midi.Select(m => m / 127f).ToArray());
        var harmonicIn = Matrix.ConcatColumns(z, _pitchEmbedding.Forward(pitch));
        var x = _harmonicNet.Forward(harmonicIn);
        _harmonicNorm.ForwardInPlace(x);
        var y = _harmonicOut.Forward(x);

        return new SynthesisParameters(
            f0Midi,
            f0Midi.Select(m => m == 0f ? 0f : MidiToHz(m)).ToArray(),
            y.SliceColumns(0, 1).Data,
            y.SliceColumns(1, NumHarmonics),
            y.SliceColumns(1 + NumHarmonics, NumNoiseBands));
    }

    /// <summary><c>440 · 2^((midi − 69) / 12)</c>.</summary>
    public static float MidiToHz(float midi) => 440f * MathF.Pow(2f, (midi - 69f) / 12f);

    /// <summary>Autoregressive f0 in MIDI units: the chosen deviation bin plus the note's pitch.</summary>
    private float[] GenerateF0Midi(Matrix z, float[] quantizedPitch, F0Sampler sampler)
    {
        var condition = _f0ConditionRnn.Forward(z);
        var h1 = new float[_f0Rnn1.Units];
        var h2 = new float[_f0Rnn2.Units];
        var input = new float[_f0Rnn1.InputSize];
        var logits = new float[PitchDeviationBins];
        int previousBin = -1; // the all-zero "go" frame
        var f0Midi = new float[z.Rows];

        for (int t = 0; t < z.Rows; t++)
        {
            condition.Row(t).CopyTo(input);
            var previous = input.AsSpan(condition.Cols);
            previous.Clear();
            if (previousBin >= 0)
                previous[previousBin] = 1f;

            _f0Rnn1.Step(input, h1);
            _f0Rnn2.Step(h1, h2);
            var hidden = (float[])h2.Clone();
            _f0Norm.ForwardInPlace(hidden);
            _f0Out.Forward(hidden, logits);

            previousBin = sampler.Sample(logits);
            // get_float_f0_dv: bin / 100 − 1, computed in float64 then cast.
            float deviation = (float)(previousBin / 100.0 - 1.0);
            f0Midi[t] = deviation + quantizedPitch[t];
        }
        return f0Midi;
    }
}
