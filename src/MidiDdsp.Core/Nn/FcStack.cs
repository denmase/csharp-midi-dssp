namespace MidiDdsp.Core.Nn;

/// <summary>
/// ddsp <c>nn.FcStackOut</c>: a stack of <c>Dense -> LayerNorm -> leaky ReLU</c>
/// blocks (ddsp <c>nn.Fc</c>) followed by an output <c>Dense</c>.
/// </summary>
public sealed class FcStackOut
{
    private readonly (Dense Dense, KerasLayerNorm Norm)[] _stack;
    private readonly Dense _out;

    public FcStackOut((Dense, KerasLayerNorm)[] stack, Dense output)
    {
        _stack = stack;
        _out = output;
    }

    public int InputSize => _stack.Length > 0 ? _stack[0].Dense.InputSize : _out.InputSize;
    public int OutputSize => _out.OutputSize;

    /// <summary>
    /// Loads from the checkpoint layout of an <c>FcStackOut</c>:
    /// <c>stack/layer_with_weights-{i}/layer_with_weights-{0: Dense, 1: LayerNorm}</c>
    /// and <c>dense_out</c>.
    /// </summary>
    public static FcStackOut Load(WeightScope scope, int layers)
    {
        var stack = new (Dense, KerasLayerNorm)[layers];
        for (int i = 0; i < layers; i++)
        {
            var fc = scope["stack"][$"layer_with_weights-{i}"];
            stack[i] = (Dense.Load(fc["layer_with_weights-0"]), KerasLayerNorm.Load(fc["layer_with_weights-1"]));
        }
        return new FcStackOut(stack, Dense.Load(scope["dense_out"]));
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> y)
    {
        ReadOnlySpan<float> current = x;
        foreach (var (dense, norm) in _stack)
        {
            var hidden = new float[dense.OutputSize];
            dense.Forward(current, hidden);
            norm.ForwardInPlace(hidden);
            Activations.LeakyReluInPlace(hidden);
            current = hidden;
        }
        _out.Forward(current, y);
    }

    public Matrix Forward(Matrix x)
    {
        var y = new Matrix(x.Rows, OutputSize);
        for (int r = 0; r < x.Rows; r++)
            Forward(x.Row(r), y.Row(r));
        return y;
    }
}
