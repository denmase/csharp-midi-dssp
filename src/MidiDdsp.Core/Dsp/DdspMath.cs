namespace MidiDdsp.Core.Dsp;

public static class DdspMath
{
    public const int SampleRate = 16000;

    /// <summary>Audio samples per control frame (16 kHz / 250 Hz).</summary>
    public const int FrameSize = 64;

    private static readonly float LogTen = MathF.Log(10f);

    /// <summary>ddsp <c>core.exp_sigmoid</c>: <c>2 · sigmoid(x)^ln(10) + 1e-7</c>.</summary>
    public static float ExpSigmoid(float x)
    {
        float sigmoid = 1f / (1f + MathF.Exp(-x));
        return 2f * MathF.Pow(sigmoid, LogTen) + 1e-7f;
    }

    /// <summary>Periodic Hann window, like <c>tf.signal.hann_window</c>.</summary>
    public static float[] HannWindow(int length)
    {
        var window = new float[length];
        for (int i = 0; i < length; i++)
            window[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / length));
        return window;
    }
}
