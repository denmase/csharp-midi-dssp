#if NETFRAMEWORK
// Backports System.Index/System.Range (added to corlib in .NET Core 3.0) so the C# 8 `^`/`..`
// syntax compiles on .NET Framework. The compiler binds `x[^n]`/`x[a..b]` structurally against
// these exact member names/shapes -- it needs no other runtime support for types (like Span<T>,
// already available here via the System.Memory package) that expose their own `Slice(int, int)`
// method. Plain array range-slicing (`arr[a..b]` producing a new array) additionally needs
// `RuntimeHelpers.GetSubArray`, which has no .NET Framework equivalent and isn't backported here;
// every array-range-slice in this codebase was rewritten to avoid it instead (see Gru.cs,
// WeightScope.cs). Mirrors the reference implementation in dotnet/runtime.
namespace System
{
    internal readonly struct Index : IEquatable<Index>
    {
        private readonly int _value;

        public Index(int value, bool fromEnd = false)
        {
            if (value < 0)
                throw new ArgumentOutOfRangeException(nameof(value));
            _value = fromEnd ? ~value : value;
        }

        private Index(int value)
        {
            _value = value;
        }

        public static Index Start => new(0);
        public static Index End => new(~0);

        public static Index FromStart(int value)
        {
            if (value < 0)
                throw new ArgumentOutOfRangeException(nameof(value));
            return new Index(value);
        }

        public static Index FromEnd(int value)
        {
            if (value < 0)
                throw new ArgumentOutOfRangeException(nameof(value));
            return new Index(~value);
        }

        public int Value => _value < 0 ? ~_value : _value;
        public bool IsFromEnd => _value < 0;

        public int GetOffset(int length)
        {
            int offset = _value;
            if (IsFromEnd)
                offset += length + 1;
            return offset;
        }

        public static implicit operator Index(int value) => FromStart(value);

        public bool Equals(Index other) => _value == other._value;
        public override bool Equals(object? obj) => obj is Index other && Equals(other);
        public override int GetHashCode() => _value;
        public override string ToString() => IsFromEnd ? "^" + (uint)Value : ((uint)Value).ToString();
    }

    internal readonly struct Range : IEquatable<Range>
    {
        public Index Start { get; }
        public Index End { get; }

        public Range(Index start, Index end)
        {
            Start = start;
            End = end;
        }

        public static Range StartAt(Index start) => new(start, Index.End);
        public static Range EndAt(Index end) => new(Index.Start, end);
        public static Range All => new(Index.Start, Index.End);

        public (int Offset, int Length) GetOffsetAndLength(int length)
        {
            int start = Start.GetOffset(length);
            int end = End.GetOffset(length);
            if ((uint)end > (uint)length || (uint)start > (uint)end)
                throw new ArgumentOutOfRangeException(nameof(length));
            return (start, end - start);
        }

        public bool Equals(Range other) => Start.Equals(other.Start) && End.Equals(other.End);
        public override bool Equals(object? obj) => obj is Range other && Equals(other);
        public override int GetHashCode() => Start.GetHashCode() * 31 + End.GetHashCode();
        public override string ToString() => Start + ".." + End;
    }
}
#endif
