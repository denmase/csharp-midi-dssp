namespace MidiDdsp.Core.Dsp;

/// <summary>In-place iterative radix-2 complex FFT in double precision.</summary>
public static class Fft
{
    /// <summary>Transforms <paramref name="re"/>/<paramref name="im"/> in place. Length must be a power of two.</summary>
    /// <param name="inverse">Computes the inverse transform, including the 1/N scaling.</param>
    public static void Transform(double[] re, double[] im, bool inverse = false)
    {
        int n = re.Length;
        if (n != im.Length || n == 0 || (n & (n - 1)) != 0)
            throw new ArgumentException("FFT length must be a power of two and re/im must match.");

        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        for (int length = 2; length <= n; length <<= 1)
        {
            double angle = (inverse ? 2 : -2) * Math.PI / length;
            int half = length >> 1;
            for (int k = 0; k < half; k++)
            {
                double wr = Math.Cos(angle * k), wi = Math.Sin(angle * k);
                for (int start = 0; start < n; start += length)
                {
                    int a = start + k, b = a + half;
                    double tr = re[b] * wr - im[b] * wi;
                    double ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }

        if (inverse)
        {
            for (int i = 0; i < n; i++)
            {
                re[i] /= n;
                im[i] /= n;
            }
        }
    }

    public static int NextPowerOfTwo(int n)
    {
        int p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }
}

/// <summary>
/// Linear convolution of a long signal with a fixed impulse response, by FFT
/// overlap-add in blocks so memory stays bounded for long audio.
/// </summary>
public sealed class FftConvolver
{
    private readonly int _irLength;
    private readonly int _blockSize;
    private readonly double[] _irRe;
    private readonly double[] _irIm;

    public FftConvolver(ReadOnlySpan<float> impulseResponse, int blockSize = 16384)
    {
        _irLength = impulseResponse.Length;
        _blockSize = blockSize;
        int fftSize = Fft.NextPowerOfTwo(blockSize + _irLength - 1);
        _irRe = new double[fftSize];
        _irIm = new double[fftSize];
        for (int i = 0; i < _irLength; i++)
            _irRe[i] = impulseResponse[i];
        Fft.Transform(_irRe, _irIm);
    }

    /// <summary>Returns the first <c>signal.Length</c> samples of <c>signal * impulseResponse</c>.</summary>
    public float[] Convolve(ReadOnlySpan<float> signal)
    {
        int fftSize = _irRe.Length;
        var output = new double[signal.Length];
        var re = new double[fftSize];
        var im = new double[fftSize];
        for (int start = 0; start < signal.Length; start += _blockSize)
        {
            int count = Math.Min(_blockSize, signal.Length - start);
            Array.Clear(re);
            Array.Clear(im);
            for (int i = 0; i < count; i++)
                re[i] = signal[start + i];

            Fft.Transform(re, im);
            for (int k = 0; k < fftSize; k++)
            {
                double r = re[k] * _irRe[k] - im[k] * _irIm[k];
                im[k] = re[k] * _irIm[k] + im[k] * _irRe[k];
                re[k] = r;
            }
            Fft.Transform(re, im, inverse: true);

            int end = Math.Min(signal.Length, start + count + _irLength - 1);
            for (int i = start; i < end; i++)
                output[i] += re[i - start];
        }
        return output.Select(v => (float)v).ToArray();
    }
}
