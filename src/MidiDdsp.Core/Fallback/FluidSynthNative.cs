using System.Reflection;
using System.Runtime.InteropServices;

namespace MidiDdsp.Core.Fallback;

/// <summary>
/// P/Invoke declarations for the parts of the libfluidsynth 2.x C API that
/// <c>pretty_midi</c> (through <c>pyfluidsynth</c>) uses. The library is
/// found at run time, so the rest of MIDI-DDSP works without it installed.
/// </summary>
internal static partial class FluidSynthNative
{
    private const string Library = "fluidsynth";

    /// <summary>Environment variable naming the libfluidsynth file to load, if set.</summary>
    public const string LibraryVariable = "MIDI_DDSP_FLUIDSYNTH_LIBRARY";

    private static readonly string[] Candidates = OperatingSystem.IsWindows()
        ? ["libfluidsynth-3.dll", "libfluidsynth-2.dll", "fluidsynth.dll"]
        : OperatingSystem.IsMacOS()
            ? ["libfluidsynth.3.dylib", "libfluidsynth.dylib",
               "/opt/homebrew/lib/libfluidsynth.dylib", "/usr/local/lib/libfluidsynth.dylib"]
            : ["libfluidsynth.so.3", "libfluidsynth.so.2", "libfluidsynth.so"];

    static FluidSynthNative()
    {
        NativeLibrary.SetDllImportResolver(typeof(FluidSynthNative).Assembly, Resolve);
    }

    /// <summary>Loads libfluidsynth, returning false if it cannot be found.</summary>
    public static bool TryLoad(out string? error)
    {
        if (TryLoadHandle(out _))
        {
            error = null;
            return true;
        }
        error = Environment.GetEnvironmentVariable(LibraryVariable) is { Length: > 0 } path
            ? $"Could not load libfluidsynth from {path} ({LibraryVariable})."
            : $"Could not find libfluidsynth (tried {string.Join(", ", Candidates)}). " +
              $"Install FluidSynth 2.x or set {LibraryVariable} to the library's path.";
        return false;
    }

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath) =>
        name == Library && TryLoadHandle(out var handle) ? handle : IntPtr.Zero;

    private static bool TryLoadHandle(out IntPtr handle)
    {
        var configured = Environment.GetEnvironmentVariable(LibraryVariable);
        if (!string.IsNullOrEmpty(configured))
            return NativeLibrary.TryLoad(configured, out handle);
        foreach (var candidate in Candidates)
            if (NativeLibrary.TryLoad(candidate, typeof(FluidSynthNative).Assembly, null, out handle))
                return true;
        handle = IntPtr.Zero;
        return false;
    }

    [LibraryImport(Library, EntryPoint = "new_fluid_settings")]
    public static partial IntPtr NewSettings();

    [LibraryImport(Library, EntryPoint = "delete_fluid_settings")]
    public static partial void DeleteSettings(IntPtr settings);

    [LibraryImport(Library, EntryPoint = "fluid_settings_setnum", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SettingsSetNum(IntPtr settings, string name, double value);

    [LibraryImport(Library, EntryPoint = "fluid_settings_setint", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SettingsSetInt(IntPtr settings, string name, int value);

    [LibraryImport(Library, EntryPoint = "new_fluid_synth")]
    public static partial IntPtr NewSynth(IntPtr settings);

    [LibraryImport(Library, EntryPoint = "delete_fluid_synth")]
    public static partial void DeleteSynth(IntPtr synth);

    [LibraryImport(Library, EntryPoint = "fluid_synth_sfload", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SoundFontLoad(IntPtr synth, string filename, int resetPresets);

    [LibraryImport(Library, EntryPoint = "fluid_synth_program_select")]
    public static partial int ProgramSelect(IntPtr synth, int channel, int soundFontId, int bank, int preset);

    [LibraryImport(Library, EntryPoint = "fluid_synth_noteon")]
    public static partial int NoteOn(IntPtr synth, int channel, int key, int velocity);

    [LibraryImport(Library, EntryPoint = "fluid_synth_noteoff")]
    public static partial int NoteOff(IntPtr synth, int channel, int key);

    [LibraryImport(Library, EntryPoint = "fluid_synth_pitch_bend")]
    public static partial int PitchBend(IntPtr synth, int channel, int value);

    [LibraryImport(Library, EntryPoint = "fluid_synth_cc")]
    public static partial int ControlChange(IntPtr synth, int channel, int number, int value);

    [LibraryImport(Library, EntryPoint = "fluid_synth_write_s16")]
    public static unsafe partial int WriteS16(
        IntPtr synth, int length, short* left, int leftOffset, int leftIncrement,
        short* right, int rightOffset, int rightIncrement);
}
