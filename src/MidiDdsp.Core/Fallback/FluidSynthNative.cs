using System.Runtime.InteropServices;
using System.Text;

namespace MidiDdsp.Core.Fallback;

/// <summary>
/// The parts of the libfluidsynth 2.x C API that <c>pretty_midi</c> (through
/// <c>pyfluidsynth</c>) uses, bound at run time through <see cref="NativeLoader"/>
/// so they work on .NET 8 and .NET Framework alike, and so the rest of
/// MIDI-DDSP works without FluidSynth installed.
/// </summary>
internal static unsafe class FluidSynthNative
{
    /// <summary>Environment variable naming the libfluidsynth file to load, if set.</summary>
    public const string LibraryVariable = "MIDI_DDSP_FLUIDSYNTH_LIBRARY";

    private static readonly string[] Candidates = NativeLoader.IsWindows
        ? ["libfluidsynth-3.dll", "libfluidsynth-2.dll", "fluidsynth.dll"]
        : NativeLoader.IsMacOS
            ? ["libfluidsynth.3.dylib", "libfluidsynth.dylib",
               "/opt/homebrew/lib/libfluidsynth.dylib", "/usr/local/lib/libfluidsynth.dylib"]
            : ["libfluidsynth.so.3", "libfluidsynth.so.2", "libfluidsynth.so"];

    private static readonly Lazy<(Api? Api, string? Error)> Loaded = new(Load);

    /// <summary>Loads libfluidsynth, returning false (with a reason) if it cannot be loaded.</summary>
    public static bool TryLoad(out string? error)
    {
        error = Loaded.Value.Error;
        return Loaded.Value.Api is not null;
    }

    private static Api Functions =>
        Loaded.Value.Api ?? throw new DllNotFoundException(Loaded.Value.Error);

    /// <summary>
    /// Tries <see cref="LibraryVariable"/>, then each candidate next to the
    /// program, then each candidate on the system's library search path. When a
    /// library file exists but fails to load, the loader's message is kept: on
    /// Windows that usually means its dependency DLLs are missing, or that its
    /// bitness does not match the process.
    /// </summary>
    private static (Api?, string?) Load()
    {
        var configured = Environment.GetEnvironmentVariable(LibraryVariable);
        if (!string.IsNullOrEmpty(configured))
        {
            return NativeLoader.TryLoad(configured, out var handle, out var reason)
                ? (new Api(handle), null)
                : (null, $"Could not load libfluidsynth from {configured} ({LibraryVariable}): {reason}");
        }

        string? firstFailure = null;
        foreach (var candidate in Candidates)
        {
            var local = Path.Combine(AppContext.BaseDirectory, candidate);
            if (!File.Exists(local))
                continue;
            if (NativeLoader.TryLoad(local, out var handle, out var reason))
                return (new Api(handle), null);
            firstFailure ??= $"Found {local} but could not load it: {reason}";
        }

        foreach (var candidate in Candidates)
        {
            if (NativeLoader.TryLoad(candidate, out var handle, out _))
                return (new Api(handle), null);
        }

        if (firstFailure is not null)
        {
            string bits = Environment.Is64BitProcess ? "64-bit (x64)" : "32-bit (x86)";
            return (null, firstFailure + (NativeLoader.IsWindows
                ? $" Copy every DLL from the bin folder of a {bits} FluidSynth release next to it, not only libfluidsynth-3.dll."
                : ""));
        }
        return (null,
            $"Could not find libfluidsynth (tried {string.Join(", ", Candidates)} next to this program and on the " +
            $"library search path). Install FluidSynth 2.x or set {LibraryVariable} to the library's path.");
    }

    public static IntPtr NewSettings() => Functions.NewSettings();

    public static void DeleteSettings(IntPtr settings) => Functions.DeleteSettings(settings);

    public static int SettingsSetNum(IntPtr settings, string name, double value)
    {
        fixed (byte* n = Utf8(name))
            return Functions.SettingsSetNum(settings, n, value);
    }

    public static int SettingsSetInt(IntPtr settings, string name, int value)
    {
        fixed (byte* n = Utf8(name))
            return Functions.SettingsSetInt(settings, n, value);
    }

    public static IntPtr NewSynth(IntPtr settings) => Functions.NewSynth(settings);

    public static void DeleteSynth(IntPtr synth) => Functions.DeleteSynth(synth);

    public static int SoundFontLoad(IntPtr synth, string filename, int resetPresets)
    {
        fixed (byte* f = Utf8(filename))
            return Functions.SoundFontLoad(synth, f, resetPresets);
    }

    public static int ProgramSelect(IntPtr synth, int channel, int soundFontId, int bank, int preset) =>
        Functions.ProgramSelect(synth, channel, soundFontId, bank, preset);

    public static int NoteOn(IntPtr synth, int channel, int key, int velocity) =>
        Functions.NoteOn(synth, channel, key, velocity);

    public static int NoteOff(IntPtr synth, int channel, int key) => Functions.NoteOff(synth, channel, key);

    public static int PitchBend(IntPtr synth, int channel, int value) => Functions.PitchBend(synth, channel, value);

    public static int ControlChange(IntPtr synth, int channel, int number, int value) =>
        Functions.ControlChange(synth, channel, number, value);

    public static int WriteS16(
        IntPtr synth, int length, short* left, int leftOffset, int leftIncrement,
        short* right, int rightOffset, int rightIncrement) =>
        Functions.WriteS16(synth, length, left, leftOffset, leftIncrement, right, rightOffset, rightIncrement);

    /// <summary>A null-terminated UTF-8 copy of a string, as FluidSynth expects for names and paths.</summary>
    private static byte[] Utf8(string value)
    {
        var bytes = new byte[Encoding.UTF8.GetByteCount(value) + 1];
        Encoding.UTF8.GetBytes(value, 0, value.Length, bytes, 0);
        return bytes;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate IntPtr NewSettingsFn();
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void DeleteFn(IntPtr handle);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int SetNumFn(IntPtr settings, byte* name, double value);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int SetIntFn(IntPtr settings, byte* name, int value);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate IntPtr NewSynthFn(IntPtr settings);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int SfLoadFn(IntPtr synth, byte* filename, int resetPresets);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int ProgramSelectFn(IntPtr synth, int channel, int soundFontId, int bank, int preset);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Int3Fn(IntPtr synth, int a, int b, int c);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Int2Fn(IntPtr synth, int a, int b);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int WriteS16Fn(IntPtr synth, int length, short* left, int leftOffset, int leftIncrement,
                                    short* right, int rightOffset, int rightIncrement);

    /// <summary>The bound functions of one loaded libfluidsynth.</summary>
    private sealed class Api(IntPtr library)
    {
        public readonly NewSettingsFn NewSettings = Bind<NewSettingsFn>(library, "new_fluid_settings");
        public readonly DeleteFn DeleteSettings = Bind<DeleteFn>(library, "delete_fluid_settings");
        public readonly SetNumFn SettingsSetNum = Bind<SetNumFn>(library, "fluid_settings_setnum");
        public readonly SetIntFn SettingsSetInt = Bind<SetIntFn>(library, "fluid_settings_setint");
        public readonly NewSynthFn NewSynth = Bind<NewSynthFn>(library, "new_fluid_synth");
        public readonly DeleteFn DeleteSynth = Bind<DeleteFn>(library, "delete_fluid_synth");
        public readonly SfLoadFn SoundFontLoad = Bind<SfLoadFn>(library, "fluid_synth_sfload");
        public readonly ProgramSelectFn ProgramSelect = Bind<ProgramSelectFn>(library, "fluid_synth_program_select");
        public readonly Int3Fn NoteOn = Bind<Int3Fn>(library, "fluid_synth_noteon");
        public readonly Int2Fn NoteOff = Bind<Int2Fn>(library, "fluid_synth_noteoff");
        public readonly Int2Fn PitchBend = Bind<Int2Fn>(library, "fluid_synth_pitch_bend");
        public readonly Int3Fn ControlChange = Bind<Int3Fn>(library, "fluid_synth_cc");
        public readonly WriteS16Fn WriteS16 = Bind<WriteS16Fn>(library, "fluid_synth_write_s16");

        private static T Bind<T>(IntPtr library, string name) where T : Delegate =>
            Marshal.GetDelegateForFunctionPointer<T>(NativeLoader.GetExport(library, name));
    }
}
