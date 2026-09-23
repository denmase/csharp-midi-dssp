using System.Numerics.Tensors;

namespace MidiDdsp.Core.Nn;

/// <summary>
/// Keras <c>GRU</c> with its TF2 defaults: <c>reset_after=True</c>, tanh
/// activation, sigmoid recurrent activation, gates ordered <c>[z, r, h]</c>.
/// <code>
/// z  = σ(x·Wz + bz + h·Uz + b'z)
/// r  = σ(x·Wr + br + h·Ur + b'r)
/// h~ = tanh(x·Wh + bh + r ⊙ (h·Uh + b'h))
/// h  = z ⊙ h_prev + (1 − z) ⊙ h~
/// </code>
/// </summary>
public sealed class Gru
{
    private readonly Dense _input;      // kernel [in, 3u] + input bias
    private readonly Dense _recurrent;  // recurrent_kernel [u, 3u] + recurrent bias

    public Gru(Dense input, Dense recurrent)
    {
        if (input.OutputSize != recurrent.OutputSize || recurrent.OutputSize != 3 * recurrent.InputSize)
            throw new ArgumentException("GRU kernel shapes are inconsistent.");
        _input = input;
        _recurrent = recurrent;
    }

    public int InputSize => _input.InputSize;
    public int Units => _recurrent.InputSize;

    /// <summary>Loads a Keras GRU cell (<c>kernel</c>, <c>recurrent_kernel</c>, <c>bias</c> of shape [2, 3u]).</summary>
    public static Gru Load(WeightScope cell)
    {
        var (kernel, kernelShape) = cell.Load("kernel", 2);
        var (recurrent, recurrentShape) = cell.Load("recurrent_kernel", 2);
        var (bias, biasShape) = cell.Load("bias", 2);
        if (biasShape[0] != 2)
            throw new InvalidDataException("Only reset_after=True GRUs (bias shape [2, 3u]) are supported.");
        int gates = biasShape[1];
        return new Gru(
            new Dense(kernel, bias[..gates], kernelShape[0]),
            new Dense(recurrent, bias[gates..], recurrentShape[0]));
    }

    /// <summary>Advances the state <paramref name="h"/> by one input frame, in place.</summary>
    public void Step(ReadOnlySpan<float> x, Span<float> h)
    {
        var xw = new float[3 * Units];
        _input.Forward(x, xw);
        StepFromProjectedInput(xw, h);
    }

    /// <summary>Runs over a sequence from a zero initial state and returns every output.</summary>
    public Matrix Forward(Matrix x, bool reverse = false)
    {
        var projected = _input.Forward(x);
        var outputs = new Matrix(x.Rows, Units);
        var h = new float[Units];
        for (int i = 0; i < x.Rows; i++)
        {
            int t = reverse ? x.Rows - 1 - i : i;
            StepFromProjectedInput(projected.Row(t), h);
            h.CopyTo(outputs.Row(t));
        }
        return outputs;
    }

    private void StepFromProjectedInput(ReadOnlySpan<float> xw, Span<float> h)
    {
        int u = Units;
        var hu = new float[3 * u];
        _recurrent.Forward(h, hu);

        var z = new float[u];
        var r = new float[u];
        TensorPrimitives.Add(xw[..u], hu.AsSpan(0, u), z);
        TensorPrimitives.Sigmoid(z, z);
        TensorPrimitives.Add(xw.Slice(u, u), hu.AsSpan(u, u), r);
        TensorPrimitives.Sigmoid(r, r);

        var candidate = new float[u];
        TensorPrimitives.Multiply(r, hu.AsSpan(2 * u, u), candidate);
        TensorPrimitives.Add(candidate, xw.Slice(2 * u, u), candidate);
        TensorPrimitives.Tanh(candidate, candidate);

        for (int i = 0; i < u; i++)
            h[i] = z[i] * h[i] + (1f - z[i]) * candidate[i];
    }
}

/// <summary>Keras <c>Bidirectional(GRU(return_sequences=True))</c> with concatenated outputs.</summary>
public sealed class BidirectionalGru
{
    private readonly Gru _forward;
    private readonly Gru _backward;

    public BidirectionalGru(Gru forward, Gru backward)
    {
        _forward = forward;
        _backward = backward;
    }

    public int OutputSize => _forward.Units + _backward.Units;

    public static BidirectionalGru Load(WeightScope scope) =>
        new(Gru.Load(scope["forward_layer/cell"]), Gru.Load(scope["backward_layer/cell"]));

    public Matrix Forward(Matrix x) =>
        Matrix.ConcatColumns(_forward.Forward(x), _backward.Forward(x, reverse: true));
}
