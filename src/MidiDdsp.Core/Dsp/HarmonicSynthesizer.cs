using MidiDdsp.Core.Nn;

namespace MidiDdsp.Core.Dsp;

/// <summary>Scaled harmonic synthesizer controls, per frame.</summary>
public sealed record HarmonicControls(float[] Amplitudes, Matrix HarmonicDistribution, float[] F0Hz);

/// <summary>
/// ddsp <c>synths.Harmonic</c> as MIDI-DDSP configures it: <c>exp_sigmoid</c>
/// scaling, harmonics at or above Nyquist removed, amplitudes upsampled with
/// overlapping Hann windows, frequencies upsampled linearly, and phase
/// accumulated with ddsp's chunked <c>angular_cumsum</c>.
/// </summary>
public sealed class HarmonicSynthesizer
{
    private const int CumsumChunk = 1000;

    public HarmonicSynthesizer(int sampleRate = DdspMath.SampleRate, int frameSize = DdspMath.FrameSize)
    {
        SampleRate = sampleRate;
        FrameSize = frameSize;
    }

    public int SampleRate { get; }
    public int FrameSize { get; }

    /// <summary><c>Harmonic.get_controls</c>: scales the network outputs and normalizes the distribution.</summary>
    public HarmonicControls GetControls(ReadOnlySpan<float> amplitudes, Matrix harmonicDistribution, float[] f0Hz)
    {
        int frames = f0Hz.Length;
        if (amplitudes.Length != frames || harmonicDistribution.Rows != frames)
            throw new ArgumentException("All controls must have one entry per frame.");

        var scaledAmplitudes = new float[frames];
        var distribution = new Matrix(frames, harmonicDistribution.Cols);
        float nyquist = SampleRate / 2f;
        for (int t = 0; t < frames; t++)
        {
            scaledAmplitudes[t] = DdspMath.ExpSigmoid(amplitudes[t]);
            var source = harmonicDistribution.Row(t);
            var row = distribution.Row(t);
            float sum = 0;
            for (int k = 0; k < row.Length; k++)
            {
                row[k] = f0Hz[t] * (k + 1) >= nyquist ? 0f : DdspMath.ExpSigmoid(source[k]);
                sum += row[k];
            }
            float divisor = sum == 0f ? 1e-7f : sum;
            for (int k = 0; k < row.Length; k++)
                row[k] /= divisor;
        }
        return new HarmonicControls(scaledAmplitudes, distribution, f0Hz);
    }

    /// <summary><c>Harmonic.get_signal</c>: renders <c>frames × frameSize</c> samples.</summary>
    public float[] Synthesize(HarmonicControls controls)
    {
        int frames = controls.F0Hz.Length;
        int harmonics = controls.HarmonicDistribution.Cols;
        int samples = frames * FrameSize;
        var audio = new float[samples];
        var window = DdspMath.HannWindow(2 * FrameSize);
        float nyquist = SampleRate / 2f;
        float twoPi = (float)(2.0 * Math.PI);
        // Linear resize of frames to samples (tf.compat.v1.image.resize, align_corners=False).
        float scale = (float)frames / samples;

        var phase = new float[harmonics];
        var chunkOffset = new float[harmonics];
        for (int n = 0; n < samples; n++)
        {
            float source = n * scale;
            int t0 = (int)MathF.Floor(source);
            int t1 = Math.Min(t0 + 1, frames - 1);
            float frac = source - t0;

            // Overlapping-window amplitude upsampling (the last frame is repeated as the endpoint).
            int j = n / FrameSize, m = n % FrameSize;
            int jNext = Math.Min(j + 1, frames - 1);
            float wCurrent = window[m + FrameSize], wNext = window[m];

            if (n % CumsumChunk == 0 && n > 0)
            {
                // angular_cumsum: restart each chunk's cumsum from the wrapped end of the last.
                for (int k = 0; k < harmonics; k++)
                    chunkOffset[k] = (chunkOffset[k] + phase[k] % twoPi) % twoPi;
                Array.Clear(phase, 0, phase.Length);
            }

            float sample = 0;
            for (int k = 0; k < harmonics; k++)
            {
                float f0 = controls.F0Hz[t0] * (k + 1);
                float frequency = f0 + (controls.F0Hz[t1] * (k + 1) - f0) * frac;
                phase[k] += frequency * twoPi / SampleRate;

                float amplitude =
                    controls.Amplitudes[j] * controls.HarmonicDistribution[j, k] * wCurrent +
                    controls.Amplitudes[jNext] * controls.HarmonicDistribution[jNext, k] * wNext;
                if (frequency >= nyquist)
                    continue;
                sample += amplitude * MathF.Sin((phase[k] + chunkOffset[k]) % twoPi);
            }
            audio[n] = sample;
        }
        return audio;
    }
}
