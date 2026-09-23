using System.Buffers.Binary;

namespace MidiDdsp.Core.Checkpoint;

/// <summary>
/// Minimal protobuf wire-format reader, enough to decode the few TensorFlow
/// messages stored in a checkpoint index (BundleHeaderProto, BundleEntryProto,
/// TensorShapeProto).
/// </summary>
internal ref struct ProtoReader
{
    private readonly ReadOnlySpan<byte> _data;
    private int _pos;

    public ProtoReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _pos = 0;
    }

    public bool AtEnd => _pos >= _data.Length;

    /// <summary>Reads the next field tag, returning its number and wire type.</summary>
    public (int Field, int WireType) ReadTag()
    {
        ulong tag = ReadVarint();
        return ((int)(tag >> 3), (int)(tag & 7));
    }

    public ulong ReadVarint()
    {
        ulong result = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            if (_pos >= _data.Length)
                throw new InvalidDataException("Truncated protobuf varint.");
            byte b = _data[_pos++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
                return result;
        }
        throw new InvalidDataException("Malformed protobuf varint.");
    }

    public uint ReadFixed32()
    {
        if (_pos + 4 > _data.Length)
            throw new InvalidDataException("Truncated protobuf fixed32.");
        uint v = BinaryPrimitives.ReadUInt32LittleEndian(_data.Slice(_pos, 4));
        _pos += 4;
        return v;
    }

    public ReadOnlySpan<byte> ReadLengthDelimited() => ReadBytes(checked((int)ReadVarint()));

    public ReadOnlySpan<byte> ReadBytes(int length)
    {
        if (length < 0 || _pos + length > _data.Length)
            throw new InvalidDataException("Truncated data while reading bytes.");
        var slice = _data.Slice(_pos, length);
        _pos += length;
        return slice;
    }

    public void Skip(int wireType)
    {
        switch (wireType)
        {
            case 0: ReadVarint(); break;
            case 1: _pos += 8; break;
            case 2: ReadLengthDelimited(); break;
            case 5: _pos += 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wireType}.");
        }
        if (_pos > _data.Length)
            throw new InvalidDataException("Truncated protobuf field.");
    }
}
