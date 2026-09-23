using MidiDdsp.Core.Dsp;

namespace MidiDdsp.Tests.Dsp;

public class FftTests
{
    [Fact]
    public void MatchesNaiveDft()
    {
        var random = new Random(5);
        int n = 64;
        var re = Enumerable.Range(0, n).Select(_ => random.NextDouble() - 0.5).ToArray();
        var im = Enumerable.Range(0, n).Select(_ => random.NextDouble() - 0.5).ToArray();
        var (expectedRe, expectedIm) = (new double[n], new double[n]);
        for (int k = 0; k < n; k++)
            for (int t = 0; t < n; t++)
            {
                double a = -2 * Math.PI * k * t / n;
                expectedRe[k] += re[t] * Math.Cos(a) - im[t] * Math.Sin(a);
                expectedIm[k] += re[t] * Math.Sin(a) + im[t] * Math.Cos(a);
            }

        var (originalRe, originalIm) = ((double[])re.Clone(), (double[])im.Clone());
        Fft.Transform(re, im);
        for (int k = 0; k < n; k++)
        {
            Assert.Equal(expectedRe[k], re[k], 1e-9);
            Assert.Equal(expectedIm[k], im[k], 1e-9);
        }

        Fft.Transform(re, im, inverse: true);
        for (int t = 0; t < n; t++)
        {
            Assert.Equal(originalRe[t], re[t], 1e-12);
            Assert.Equal(originalIm[t], im[t], 1e-12);
        }
    }

    [Fact]
    public void ConvolverMatchesDirectConvolutionAcrossBlocks()
    {
        var random = new Random(9);
        var signal = Enumerable.Range(0, 1000).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
        var ir = Enumerable.Range(0, 70).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();

        var actual = new FftConvolver(ir, blockSize: 128).Convolve(signal);

        for (int n = 0; n < signal.Length; n++)
        {
            double expected = 0;
            for (int m = 0; m < ir.Length && m <= n; m++)
                expected += signal[n - m] * ir[m];
            Assert.Equal(expected, actual[n], 1e-5);
        }
    }
}
