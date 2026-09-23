using MidiDdsp.Core.Checkpoint;
using MidiDdsp.Core.Models;
using MidiDdsp.Core.Nn;

namespace MidiDdsp.Tests.Nn;

/// <summary>
/// Runs C# layers with the pretrained weights on the inputs recorded by
/// tools/reference/export_layer_reference.py and compares with the outputs
/// the original TensorFlow model produced.
/// </summary>
public class PretrainedLayerTests
{
    private static readonly Npz Reference = Npz.LoadReference("layer_reference.npz");

    [RequiresWeightsFact]
    public void ExpressionGeneratorConditioningMatches()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.ExpressionGenerator);
        var model = ExpressionGenerator.Load(checkpoint);

        var encoded = model.EncodeConditioning(
            Reference.Ints("eg_note_pitch"), Reference.Floats("eg_note_length"), Reference.Ints("eg_instrument_id")[0]);

        AssertClose.Equal(Reference.Matrix("eg_encoded_cond"), encoded);
    }

    [RequiresWeightsFact]
    public void ExpressionGeneratorOutputMatches()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.ExpressionGenerator);
        var model = ExpressionGenerator.Load(checkpoint);
        var pitch = Reference.Ints("eg_note_pitch");

        var output = model.Generate(pitch, Reference.Floats("eg_note_length"), Reference.Ints("eg_instrument_id")[0]);

        AssertClose.Equal(Reference.Matrix("eg_output"), output);
        for (int i = 0; i < pitch.Length; i++)
            if (pitch[i] == 0)
                Assert.All(output.Row(i).ToArray(), v => Assert.Equal(0f, v));
    }

    [RequiresWeightsFact]
    public void PreconditioningStackMatches()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var stack = FcStackOut.Load(new WeightScope(checkpoint, "midi_decoder/z_preconditioning_stack"), layers: 5);

        AssertClose.Equal(Reference.Matrix("sg_precond_out"), stack.Forward(Reference.Matrix("sg_precond_in")));
    }

    [RequiresWeightsFact]
    public void InstrumentEmbeddingMatches()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var embedding = Embedding.Load(new WeightScope(checkpoint, "midi_decoder/instrument_emb"));
        var expected = Reference.Matrix("sg_instrument_emb");

        int[] ids = [0, 7, 19];
        for (int i = 0; i < ids.Length; i++)
            Assert.Equal(expected.Row(i).ToArray(), embedding[ids[i]].ToArray());
    }

    [RequiresWeightsFact]
    public void BidirectionalGruMatches()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var birnn = BidirectionalGru.Load(new WeightScope(checkpoint, "midi_decoder/decoder/midi_to_f0/birnn"));

        AssertClose.Equal(Reference.Matrix("sg_f0_birnn_out"), birnn.Forward(Reference.Matrix("sg_f0_birnn_in")));
    }

    [RequiresWeightsFact]
    public void SingleFrameNormAndDenseMatch()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var scope = new WeightScope(checkpoint, "midi_decoder/decoder/midi_to_f0");
        var norm = DdspLayerNorm.Load(scope["norm"]);
        var dense = Dense.Load(scope["dense_out"]);

        var frame = Reference.Floats("sg_f0_head_in");
        norm.ForwardInPlace(frame);
        var logits = new float[dense.OutputSize];
        dense.Forward(frame, logits);

        AssertClose.Equal(Reference.Floats("sg_f0_head_out"), logits);
    }

    [RequiresWeightsFact]
    public void PitchEmbeddingDenseMatches()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var dense = Dense.Load(new WeightScope(checkpoint, "midi_decoder/decoder/q_pitch_emb"));
        var pitch = Reference.Matrix("sg_q_pitch_in");
        for (int i = 0; i < pitch.Data.Length; i++)
            pitch.Data[i] /= 127f;

        AssertClose.Equal(Reference.Matrix("sg_q_pitch_emb_out"), dense.Forward(pitch));
    }

    [RequiresWeightsFact]
    public void DilatedConvStackWithHeadMatches()
    {
        using var checkpoint = TfCheckpoint.Open(PretrainedWeights.SynthesisGenerator);
        var scope = new WeightScope(checkpoint, "midi_decoder/decoder/midi_f0_to_harmonic");
        var net = DilatedConvStack.Load(scope["net"], stacks: 4, layersPerStack: 5);
        var norm = DdspLayerNorm.Load(scope["norm"]);
        var dense = Dense.Load(scope["dense_out"]);

        var x = net.Forward(Reference.Matrix("sg_harmonic_in"));
        norm.ForwardInPlace(x);
        var y = dense.Forward(x);

        AssertClose.Equal(Reference.Matrix("sg_harmonic_amplitudes"), y.SliceColumns(0, 1));
        AssertClose.Equal(Reference.Matrix("sg_harmonic_distribution"), y.SliceColumns(1, 60));
        AssertClose.Equal(Reference.Matrix("sg_harmonic_noise_magnitudes"), y.SliceColumns(61, 65));
    }
}
