using System.Text;

namespace MidiDdsp.Core.Audio;

/// <summary>Writes mono WAV files.</summary>
public static class WavWriter
{
    /// <summary>
    /// Writes 16-bit PCM, scaling by 32767 and clipping to [−1, 1] (what the
    /// original's <c>soundfile.write</c> produces for in-range audio).
    /// </summary>
    public static void WritePcm16(string path, ReadOnlySpan<float> samples, int sampleRate)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        WriteHeader(writer, samples.Length, sampleRate, bitsPerSample: 16, formatTag: 1);
        foreach (var s in samples)
            writer.Write((short)Math.Round(Math.Clamp(s, -1f, 1f) * 32767f));
    }

    /// <summary>Writes 32-bit IEEE float samples, without clipping.</summary>
    public static void WriteFloat32(string path, ReadOnlySpan<float> samples, int sampleRate)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        WriteHeader(writer, samples.Length, sampleRate, bitsPerSample: 32, formatTag: 3);
        foreach (var s in samples)
            writer.Write(s);
    }

    private static void WriteHeader(BinaryWriter writer, int sampleCount, int sampleRate, int bitsPerSample, short formatTag)
    {
        int blockAlign = bitsPerSample / 8;
        int dataSize = checked(sampleCount * blockAlign);
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(36 + dataSize);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));
        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16);
        writer.Write(formatTag);
        writer.Write((short)1); // mono
        writer.Write(sampleRate);
        writer.Write(sampleRate * blockAlign);
        writer.Write((short)blockAlign);
        writer.Write((short)bitsPerSample);
        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataSize);
    }
}
