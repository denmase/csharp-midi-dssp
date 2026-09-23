using MidiDdsp.Core.Nn;

namespace MidiDdsp.Core.Dsp;

/// <summary>
/// ddsp <c>synths.FilteredNoise</c> with its defaults (initial bias −5,
/// <c>window_size=257</c>): uniform white noise filtered, frame by frame, by a
/// zero-phase FIR filter built from each frame's band magnitudes.
/// </summary>
/// <remarks>
/// With 65 bands the filter has 128 taps. <c>window_size=257</c> is larger than
/// that, so ddsp uses a full-length Hann window, and <c>fft_convolve</c> centers
/// the filter with a delay of 62 samples.
/// </remarks>
public sealed class FilteredNoiseSynthesizer
{
    private const float InitialBias = -5f;

    public FilteredNoiseSynthesizer(int frameSize = DdspMath.FrameSize)
    {
        FrameSize = frameSize;
    }

    public int FrameSize { get; }

    /// <summary><c>FilteredNoise.get_controls</c>: <c>exp_sigmoid(magnitudes − 5)</c>.</summary>
    public Matrix GetControls(Matrix magnitudes)
    {
        var scaled = new Matrix(magnitudes.Rows, magnitudes.Cols);
        for (int i = 0; i < scaled.Data.Length; i++)
            scaled.Data[i] = DdspMath.ExpSigmoid(magnitudes.Data[i] + InitialBias);
        return scaled;
    }

    /// <summary>Filters fresh uniform noise in [−1, 1).</summary>
    public float[] Synthesize(Matrix magnitudes, Random random)
    {
        var noise = new float[magnitudes.Rows * FrameSize];
        for (int i = 0; i < noise.Length; i++)
            noise[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        return Synthesize(magnitudes, noise);
    }

    /// <summary>
    /// <c>core.frequency_filter(noise, magnitudes, window_size=257)</c> on a
    /// given noise signal of <c>frames × frameSize</c> samples.
    /// </summary>
    public float[] Synthesize(Matrix magnitudes, ReadOnlySpan<float> noise)
    {
        int frames = magnitudes.Rows;
        if (noise.Length != frames * FrameSize)
            throw new ArgumentException($"Expected {frames * FrameSize} noise samples, got {noise.Length}.");

        int irSize = 2 * (magnitudes.Cols - 1);
        int delay = (irSize - 1) / 2 - 1; // crop_and_compensate_delay with padding='same'
        var full = new double[noise.Length + irSize];
        var impulseResponse = new float[irSize];
        var window = DdspMath.HannWindow(irSize);
        var cosTable = CosineTable(irSize);

        for (int f = 0; f < frames; f++)
        {
            ImpulseResponse(magnitudes.Row(f), window, cosTable, impulseResponse);
            int start = f * FrameSize;
            for (int s = 0; s < FrameSize; s++)
            {
                double x = noise[start + s];
                int offset = start + s;
                for (int m = 0; m < irSize; m++)
                    full[offset + m] += x * impulseResponse[m];
            }
        }

        var output = new float[noise.Length];
        for (int n = 0; n < output.Length; n++)
            output[n] = (float)full[n + delay];
        return output;
    }

    /// <summary>
    /// <c>frequency_impulse_response</c>: the real inverse DFT of the magnitudes,
    /// Hann-windowed and shifted to be centered at <c>irSize / 2</c>. With the
    /// window's own shift this is <c>ir[m] = hann[m] · irfft(mag)[(m + N/2) mod N]</c>.
    /// </summary>
    private static void ImpulseResponse(ReadOnlySpan<float> magnitudes, float[] window, double[] cosTable, float[] ir)
    {
        int n = ir.Length;
        int bins = magnitudes.Length;
        for (int m = 0; m < n; m++)
        {
            int t = (m + n / 2) % n;
            double sum = magnitudes[0] + magnitudes[bins - 1] * (t % 2 == 0 ? 1 : -1);
            for (int k = 1; k < bins - 1; k++)
                sum += 2.0 * magnitudes[k] * cosTable[k * t % n];
            ir[m] = (float)(window[m] * sum / n);
        }
    }

    private static double[] CosineTable(int n)
    {
        var table = new double[n];
        for (int i = 0; i < n; i++)
            table[i] = Math.Cos(2.0 * Math.PI * i / n);
        return table;
    }
}
