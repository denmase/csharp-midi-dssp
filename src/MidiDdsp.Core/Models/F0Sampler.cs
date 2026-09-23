namespace MidiDdsp.Core.Models;

/// <summary>How the f0 decoder picks each frame's pitch deviation bin.</summary>
public enum F0SamplingMethod
{
    /// <summary>Nucleus sampling with p = 0.95, as the pretrained model runs by default.</summary>
    TopP,

    /// <summary>Always the most likely bin. Deterministic.</summary>
    Argmax,
}

/// <summary>Picks a class index from logits, like midi_ddsp's <c>sample_from</c>.</summary>
public sealed class F0Sampler
{
    public const float TopPThreshold = 0.95f;

    private readonly Random? _random;

    public F0Sampler(F0SamplingMethod method, Random? random = null)
    {
        Method = method;
        _random = random ?? (method == F0SamplingMethod.TopP ? new Random() : null);
    }

    public F0SamplingMethod Method { get; }

    public int Sample(ReadOnlySpan<float> logits) => Method switch
    {
        F0SamplingMethod.Argmax => Argmax(logits),
        _ => SampleTopP(logits),
    };

    /// <summary>Index of the first maximum, like <c>tf.argmax</c>.</summary>
    public static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best])
                best = i;
        return best;
    }

    /// <summary>
    /// <c>top_p_sample</c> then <c>tf.random.categorical</c>: keep the most likely
    /// classes until the probability mass before them reaches p (so at least one
    /// is kept), plus any class tied with the smallest kept logit, then sample
    /// from the softmax over the kept classes.
    /// </summary>
    private int SampleTopP(ReadOnlySpan<float> logits)
    {
        var sorted = logits.ToArray();
        Array.Sort(sorted, (a, b) => b.CompareTo(a));

        float max = sorted[0];
        double total = 0;
        foreach (var v in sorted)
            total += Math.Exp(v - max);

        float minKept = sorted[0];
        double cumulative = 0; // exclusive cumulative probability
        foreach (var v in sorted)
        {
            if (cumulative >= TopPThreshold)
                break;
            minKept = v;
            cumulative += Math.Exp(v - max) / total;
        }

        double keptTotal = 0;
        foreach (var v in logits)
            if (v >= minKept)
                keptTotal += Math.Exp(v - max);

        double target = _random!.NextDouble() * keptTotal;
        int last = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] < minKept)
                continue;
            last = i;
            target -= Math.Exp(logits[i] - max);
            if (target < 0)
                return i;
        }
        return last;
    }
}
