#if NETFRAMEWORK
namespace System.Runtime.CompilerServices
{
    // The C# compiler only needs this type's presence -- not its runtime behavior -- to allow
    // `init` accessors and record types. .NET 5+ ships it in the BCL; .NET Framework doesn't, so
    // this backports it, following the well-known community pattern for downlevel `init`/`record`
    // support (e.g. https://github.com/dotnet/roslyn/issues/45510).
    internal static class IsExternalInit
    {
    }
}
#endif
