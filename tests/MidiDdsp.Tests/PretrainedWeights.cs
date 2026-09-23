namespace MidiDdsp.Tests;

/// <summary>
/// Locates the pretrained MIDI-DDSP weights: the MIDI_DDSP_WEIGHTS environment
/// variable if set, otherwise weights/midi_ddsp_model_weights_urmp_9_10 at the
/// repository root (where tools/download_weights.sh puts them).
/// </summary>
internal static class PretrainedWeights
{
    public static string? Directory { get; } = Find();

    public static string ExpressionGenerator => Path.Combine(Directory!, "expression_generator", "5000");
    public static string SynthesisGenerator => Path.Combine(Directory!, "synthesis_generator", "50000");

    private static string? Find()
    {
        var fromEnv = Environment.GetEnvironmentVariable("MIDI_DDSP_WEIGHTS");
        if (!string.IsNullOrEmpty(fromEnv))
            return System.IO.Directory.Exists(fromEnv) ? fromEnv : null;

        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "MidiDdsp.sln")))
            {
                var candidate = Path.Combine(dir.FullName, "weights", "midi_ddsp_model_weights_urmp_9_10");
                return System.IO.Directory.Exists(candidate) ? candidate : null;
            }
        }
        return null;
    }
}

/// <summary>A fact that is skipped when the pretrained weights are not available.</summary>
public sealed class RequiresWeightsFactAttribute : FactAttribute
{
    public RequiresWeightsFactAttribute()
    {
        if (PretrainedWeights.Directory is null)
            Skip = "Pretrained weights not found; run tools/download_weights.sh or set MIDI_DDSP_WEIGHTS.";
    }
}
