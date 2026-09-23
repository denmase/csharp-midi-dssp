using System.Numerics.Tensors;

namespace MidiDdsp.Core.Nn;

/// <summary>Keras <c>Dense</c>: <c>y = x · kernel + bias</c>, kernel stored as <c>[in, out]</c>.</summary>
public sealed class Dense
{
    private readonly float[] _kernel;
    private readonly float[] _bias;

    public Dense(float[] kernel, float[] bias, int inputSize)
    {
        if (kernel.Length != inputSize * bias.Length)
            throw new ArgumentException("Kernel size does not match [inputSize, bias.Length].");
        _kernel = kernel;
        _bias = bias;
        InputSize = inputSize;
    }

    public int InputSize { get; }
    public int OutputSize => _bias.Length;

    public static Dense Load(WeightScope scope)
    {
        var (kernel, shape) = scope.Load("kernel", 2);
        var (bias, _) = scope.Load("bias", 1);
        return new Dense(kernel, bias, shape[0]);
    }

    public void Forward(ReadOnlySpan<float> x, Span<float> y)
    {
        CheckSizes(x.Length, y.Length);
        _bias.CopyTo(y);
        for (int i = 0; i < x.Length; i++)
            TensorPrimitives.MultiplyAdd(_kernel.AsSpan(i * y.Length, y.Length), x[i], y, y);
    }

    public Matrix Forward(Matrix x)
    {
        var y = new Matrix(x.Rows, OutputSize);
        for (int r = 0; r < x.Rows; r++)
            Forward(x.Row(r), y.Row(r));
        return y;
    }

    private void CheckSizes(int inLength, int outLength)
    {
        if (inLength != InputSize || outLength != OutputSize)
            throw new ArgumentException(
                $"Dense expects {InputSize} -> {OutputSize}, got {inLength} -> {outLength}.");
    }
}

/// <summary>Keras <c>Embedding</c>: a lookup table of <c>[vocabulary, dimension]</c>.</summary>
public sealed class Embedding
{
    private readonly float[] _table;

    public Embedding(float[] table, int dimension)
    {
        _table = table;
        Dimension = dimension;
        VocabularySize = table.Length / dimension;
    }

    public int Dimension { get; }
    public int VocabularySize { get; }

    public static Embedding Load(WeightScope scope)
    {
        var (table, shape) = scope.Load("embeddings", 2);
        return new Embedding(table, shape[1]);
    }

    public ReadOnlySpan<float> this[int index]
    {
        get
        {
            if ((uint)index >= (uint)VocabularySize)
                throw new ArgumentOutOfRangeException(nameof(index), $"Index {index} outside vocabulary of {VocabularySize}.");
            return _table.AsSpan(index * Dimension, Dimension);
        }
    }
}

/// <summary>Keras <c>LayerNormalization</c> over the last axis (epsilon 1e-3, the Keras default).</summary>
public sealed class KerasLayerNorm
{
    private const float Epsilon = 1e-3f;
    private readonly float[] _gamma;
    private readonly float[] _beta;

    public KerasLayerNorm(float[] gamma, float[] beta)
    {
        _gamma = gamma;
        _beta = beta;
    }

    public static KerasLayerNorm Load(WeightScope scope) =>
        new(scope.Load("gamma", 1).Values, scope.Load("beta", 1).Values);

    public void ForwardInPlace(Span<float> x)
    {
        var (mean, variance) = Stats.MeanVariance(x);
        float invStd = 1f / MathF.Sqrt(variance + Epsilon);
        for (int i = 0; i < x.Length; i++)
            x[i] = (x[i] - mean) * invStd * _gamma[i] + _beta[i];
    }
}

/// <summary>
/// ddsp <c>nn.Normalize('layer')</c>. Unlike Keras layer norm, the mean and
/// variance are taken over <em>all</em> time steps and channels of the input,
/// then a per-channel scale and shift is applied (epsilon 1e-5). Applied to a
/// single frame, as in the autoregressive decoders, this reduces to
/// normalizing that frame's channels.
/// </summary>
public sealed class DdspLayerNorm
{
    private const float Epsilon = 1e-5f;
    private readonly float[] _scale;
    private readonly float[] _shift;

    public DdspLayerNorm(float[] scale, float[] shift)
    {
        _scale = scale;
        _shift = shift;
    }

    public static DdspLayerNorm Load(WeightScope scope) =>
        new(scope.LoadVector("scale"), scope.LoadVector("shift"));

    public void ForwardInPlace(Matrix x)
    {
        if (x.Cols != _scale.Length)
            throw new ArgumentException($"Expected {_scale.Length} channels, got {x.Cols}.");
        var (mean, variance) = Stats.MeanVariance(x.Data);
        float invStd = 1f / MathF.Sqrt(variance + Epsilon);
        for (int r = 0; r < x.Rows; r++)
        {
            var row = x.Row(r);
            for (int c = 0; c < row.Length; c++)
                row[c] = (row[c] - mean) * invStd * _scale[c] + _shift[c];
        }
    }

    public void ForwardInPlace(Span<float> frame)
    {
        if (frame.Length != _scale.Length)
            throw new ArgumentException($"Expected {_scale.Length} channels, got {frame.Length}.");
        var (mean, variance) = Stats.MeanVariance(frame);
        float invStd = 1f / MathF.Sqrt(variance + Epsilon);
        for (int c = 0; c < frame.Length; c++)
            frame[c] = (frame[c] - mean) * invStd * _scale[c] + _shift[c];
    }
}

public static class Activations
{
    /// <summary><c>tf.nn.leaky_relu</c> with its default slope of 0.2.</summary>
    public static void LeakyReluInPlace(Span<float> x, float alpha = 0.2f)
    {
        for (int i = 0; i < x.Length; i++)
            if (x[i] < 0) x[i] *= alpha;
    }

    public static void ReluInPlace(Span<float> x)
    {
        for (int i = 0; i < x.Length; i++)
            if (x[i] < 0) x[i] = 0;
    }
}

internal static class Stats
{
    /// <summary>Population mean and variance, accumulated in double precision.</summary>
    public static (float Mean, float Variance) MeanVariance(ReadOnlySpan<float> x)
    {
        double sum = 0;
        foreach (var v in x) sum += v;
        double mean = sum / x.Length;
        double sq = 0;
        foreach (var v in x)
        {
            double d = v - mean;
            sq += d * d;
        }
        return ((float)mean, (float)(sq / x.Length));
    }
}
