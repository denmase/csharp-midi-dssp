using System.ComponentModel;
using System.Runtime.InteropServices;

namespace MidiDdsp.Core.Fallback;

/// <summary>
/// Loads a native library and looks up its exports, on .NET 8 through
/// <c>NativeLibrary</c> and on .NET Framework through <c>LoadLibraryEx</c> /
/// <c>GetProcAddress</c> (Windows) or <c>dlopen</c> / <c>dlsym</c> (Mono on
/// Linux and macOS). Failures return the loader's own reason.
/// </summary>
internal static class NativeLoader
{
    public static bool IsWindows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
    public static bool IsMacOS => RuntimeInformation.IsOSPlatform(OSPlatform.OSX);

    /// <summary>Loads a library by path, or by file name through the system's search path.</summary>
    public static bool TryLoad(string pathOrName, out IntPtr handle, out string? reason)
    {
#if NETFRAMEWORK
        if (IsWindows)
        {
            // For a full path, also search that library's own folder for its dependencies.
            uint flags = Path.IsPathRooted(pathOrName) ? LoadWithAlteredSearchPath : 0;
            handle = LoadLibraryExW(pathOrName, IntPtr.Zero, flags);
            reason = handle == IntPtr.Zero ? new Win32Exception(Marshal.GetLastWin32Error()).Message : null;
            return handle != IntPtr.Zero;
        }
        handle = IsMacOS ? DlopenMac(pathOrName, RtldNow) : DlopenLinux(pathOrName, RtldNow);
        reason = handle == IntPtr.Zero ? DlError() : null;
        return handle != IntPtr.Zero;
#else
        try
        {
            handle = Path.IsPathRooted(pathOrName)
                ? NativeLibrary.Load(pathOrName)
                : NativeLibrary.Load(pathOrName, typeof(NativeLoader).Assembly, null);
            reason = null;
            return true;
        }
        catch (Exception e) when (e is DllNotFoundException or BadImageFormatException)
        {
            handle = IntPtr.Zero;
            reason = e.Message;
            return false;
        }
#endif
    }

    /// <summary>The address of an exported function; throws if the library does not export it.</summary>
    public static IntPtr GetExport(IntPtr handle, string name)
    {
#if NETFRAMEWORK
        var address = IsWindows ? GetProcAddress(handle, name)
            : IsMacOS ? DlsymMac(handle, name) : DlsymLinux(handle, name);
        return address != IntPtr.Zero
            ? address
            : throw new EntryPointNotFoundException($"The FluidSynth library does not export {name}.");
#else
        return NativeLibrary.GetExport(handle, name);
#endif
    }

#if NETFRAMEWORK
    private const uint LoadWithAlteredSearchPath = 0x00000008;
    private const int RtldNow = 2;

    [DllImport("kernel32", EntryPoint = "LoadLibraryExW", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibraryExW(string fileName, IntPtr file, uint flags);

    [DllImport("kernel32", CharSet = CharSet.Ansi, BestFitMapping = false, SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr module, string procName);

    [DllImport("libdl.so.2", EntryPoint = "dlopen")]
    private static extern IntPtr DlopenLinux(string fileName, int flags);

    [DllImport("libdl.so.2", EntryPoint = "dlsym")]
    private static extern IntPtr DlsymLinux(IntPtr handle, string symbol);

    [DllImport("libdl.so.2", EntryPoint = "dlerror")]
    private static extern IntPtr DlerrorLinux();

    [DllImport("libSystem.dylib", EntryPoint = "dlopen")]
    private static extern IntPtr DlopenMac(string fileName, int flags);

    [DllImport("libSystem.dylib", EntryPoint = "dlsym")]
    private static extern IntPtr DlsymMac(IntPtr handle, string symbol);

    [DllImport("libSystem.dylib", EntryPoint = "dlerror")]
    private static extern IntPtr DlerrorMac();

    private static string DlError() =>
        Marshal.PtrToStringAnsi(IsMacOS ? DlerrorMac() : DlerrorLinux()) ?? "unknown dlopen error";
#endif
}
