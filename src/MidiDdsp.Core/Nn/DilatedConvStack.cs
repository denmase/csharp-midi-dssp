using System.Numerics.Tensors;

namespace MidiDdsp.Core.Nn;

/// <summary>
/// A 1-D convolution over time with <c>'same'</c> padding and stride 1, loaded
/// from the <c>Conv2D</c> with a <c>(k, 1)</c> kernel that ddsp uses for it
/// (kernel stored as <c>[k, 1, in, out]</c>).
/// </summary>
public sealed class Conv1D
{
    private readonly float[] _kernel;
    private readonly float[] _bias;

    public Conv1D(float[] kernel, float[] bias, int kernelSize, int inputChannels, int dilation)
    {
        if (kernel.Length != kernelSize * inputChannels * bias.Length)
            throw new ArgumentException("Kernel size does not match [kernelSize, 1, in, out].");
        _kernel = kernel;
        _bias = bias;
        KernelSize = kernelSize;
        InputChannels = inputChannels;
        Dilation = dilation;
    }

    public int KernelSize { get; }
    public int InputChannels { get; }
    public int OutputChannels => _bias.Length;
    public int Dilation { get; }

    public static Conv1D Load(WeightScope scope, int dilation)
    {
        var (kernel, shape) = scope.Load("kernel", 4);
        if (shape[1] != 1)
            throw new InvalidDataException("Expected a (k, 1) convolution kernel.");
        var (bias, _) = scope.Load("bias", 1);
        return new Conv1D(kernel, bias, shape[0], shape[2], dilation);
    }

    public Matrix Forward(Matrix x)
    {
        if (x.Cols != InputChannels)
            throw new ArgumentException($"Expected {InputChannels} channels, got {x.Cols}.");

        // TensorFlow 'same' padding: total (k - 1) * dilation, the smaller half on the left.
        int padLeft = (KernelSize - 1) * Dilation / 2;
        int outCh = OutputChannels;
        var y = new Matrix(x.Rows, outCh);
        for (int t = 0; t < x.Rows; t++)
        {
            var dest = y.Row(t);
            _bias.CopyTo(dest);
            for (int j = 0; j < KernelSize; j++)
            {
                int source = t + j * Dilation - padLeft;
                if (source < 0 || source >= x.Rows)
                    continue;
                var input = x.Row(source);
                int tap = j * InputChannels * outCh;
                for (int i = 0; i < InputChannels; i++)
                    TensorPrimitives.MultiplyAdd(_kernel.AsSpan(tap + i * outCh, outCh), input[i], dest, dest);
            }
        }
        return y;
    }
}

/// <summary>
/// ddsp <c>nn.DilatedConvStack</c> without conditioning or resampling:
/// <c>x = conv_in(x)</c>, then for each layer <c>x += norm(conv(relu(x)))</c>,
/// with dilation <c>2^j</c> for layer <c>j</c> within each stack.
/// </summary>
public sealed class DilatedConvStack
{
    private readonly Conv1D _convIn;
    private readonly (Conv1D Conv, DdspLayerNorm Norm)[] _layers;

    public DilatedConvStack(Conv1D convIn, (Conv1D, DdspLayerNorm)[] layers)
    {
        _convIn = convIn;
        _layers = layers;
    }

    public int OutputChannels => _convIn.OutputChannels;

    public static DilatedConvStack Load(WeightScope scope, int stacks, int layersPerStack, int dilationBase = 2)
    {
        var layers = new (Conv1D, DdspLayerNorm)[stacks * layersPerStack];
        for (int i = 0; i < layers.Length; i++)
        {
            int dilation = (int)Math.Pow(dilationBase, i % layersPerStack);
            layers[i] = (
                Conv1D.Load(scope[$"layers/{i}/layer_with_weights-0"], dilation),
                DdspLayerNorm.Load(scope[$"norms/{i}"]));
        }
        return new DilatedConvStack(Conv1D.Load(scope["conv_in"], dilation: 1), layers);
    }

    public Matrix Forward(Matrix x)
    {
        x = _convIn.Forward(x);
        foreach (var (conv, norm) in _layers)
        {
            var activated = new Matrix(x.Rows, x.Cols, (float[])x.Data.Clone());
            Activations.ReluInPlace(activated.Data);
            var y = conv.Forward(activated);
            norm.ForwardInPlace(y);
            TensorPrimitives.Add(x.Data, y.Data, x.Data);
        }
        return x;
    }
}
