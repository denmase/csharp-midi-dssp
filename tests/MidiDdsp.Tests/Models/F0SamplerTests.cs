using MidiDdsp.Core.Models;

namespace MidiDdsp.Tests.Models;

public class F0SamplerTests
{
    [Fact]
    public void ArgmaxReturnsFirstMaximum()
    {
        Assert.Equal(1, F0Sampler.Argmax([0f, 3f, 1f, 3f]));
    }

    [Fact]
    public void TopPOnlySamplesFromTheNucleus()
    {
        // Softmax ≈ [0.64, 0.24, 0.09, 0.03, ...]: the first three reach 0.95
        // only after the third, so indices 0-2 are kept and the rest never drawn.
        float[] logits = [3f, 2f, 1f, 0f, -1f, -2f];
        var sampler = new F0Sampler(F0SamplingMethod.TopP, new Random(7));
        var counts = new int[logits.Length];
        for (int i = 0; i < 20_000; i++)
            counts[sampler.Sample(logits)]++;

        Assert.All(counts.Skip(3), c => Assert.Equal(0, c));
        Assert.All(counts.Take(3), c => Assert.True(c > 0));
        // Renormalized over the nucleus, index 0 has probability ≈ 0.665.
        Assert.InRange(counts[0] / 20_000.0, 0.64, 0.69);
    }

    [Fact]
    public void TopPKeepsTheTopClassWhenItDominates()
    {
        float[] logits = [0f, 20f, 0f];
        var sampler = new F0Sampler(F0SamplingMethod.TopP, new Random(1));
        for (int i = 0; i < 100; i++)
            Assert.Equal(1, sampler.Sample(logits));
    }

    [Fact]
    public void SeededSamplingIsReproducible()
    {
        float[] logits = [1f, 1.2f, 0.8f, 1.1f];
        var a = new F0Sampler(F0SamplingMethod.TopP, new Random(42));
        var b = new F0Sampler(F0SamplingMethod.TopP, new Random(42));
        for (int i = 0; i < 50; i++)
            Assert.Equal(a.Sample(logits), b.Sample(logits));
    }
}
