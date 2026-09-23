using MidiDdsp.Core.Checkpoint;
using MidiDdsp.Core.Nn;

namespace MidiDdsp.Core.Dsp;

/// <summary>
/// MIDI-DDSP's <c>ReverbModules</c>: one learned 48,000-sample (3 s) impulse
/// response per instrument, applied with ddsp's <c>effects.Reverb</c>.
/// At inference the response is faded out after its first second by
/// <c>exp(−4t)</c>, its first tap is zeroed, and the dry signal is added back.
/// </summary>
public sealed class Reverb
{
    private const int DecayStart = 16000;
    private const float DecayExponent = 4f;

    private readonly Embedding _impulseResponses;
    private readonly Dictionary<int, FftConvolver> _convolvers = [];

    private Reverb(Embedding impulseResponses)
    {
        _impulseResponses = impulseResponses;
    }

    public int Length => _impulseResponses.Dimension;

    /// <summary>Loads <c>reverb_module/magnitudes_embedding</c> from the synthesis generator checkpoint.</summary>
    public static Reverb Load(TfCheckpoint checkpoint) =>
        new(Embedding.Load(new WeightScope(checkpoint, "reverb_module/magnitudes_embedding")));

    /// <summary>The impulse response used at inference for an instrument.</summary>
    public float[] ImpulseResponse(int instrumentId)
    {
        var ir = _impulseResponses[instrumentId].ToArray();
        int decayLength = ir.Length - DecayStart;
        float step = 1f / (decayLength - 1); // tf.linspace(0, 1, decayLength)
        for (int i = DecayStart; i < ir.Length; i++)
            ir[i] *= MathF.Exp(-DecayExponent * ((i - DecayStart) * step));
        ir[0] = 0f; // the dry path is added separately
        return ir;
    }

    /// <summary>Returns <c>audio + audio * ir</c>, cropped to the input length.</summary>
    public float[] Apply(ReadOnlySpan<float> audio, int instrumentId)
    {
        if (!_convolvers.TryGetValue(instrumentId, out var convolver))
            _convolvers[instrumentId] = convolver = new FftConvolver(ImpulseResponse(instrumentId));
        var wet = convolver.Convolve(audio);
        for (int i = 0; i < wet.Length; i++)
            wet[i] += audio[i];
        return wet;
    }
}
