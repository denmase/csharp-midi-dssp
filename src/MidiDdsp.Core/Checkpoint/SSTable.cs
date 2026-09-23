using System.Buffers.Binary;

namespace MidiDdsp.Core.Checkpoint;

/// <summary>
/// Reader for the LevelDB-style sorted string table that TensorFlow uses for
/// a checkpoint's <c>.index</c> file. Only uncompressed blocks are supported,
/// which is what TensorFlow's BundleWriter produces.
/// </summary>
internal static class SSTable
{
    private const int FooterSize = 48;
    private const int BlockTrailerSize = 5; // 1 byte compression type + 4 byte crc
    private const ulong TableMagic = 0xdb4775248b80fb57UL;

    /// <summary>Returns every key/value pair in the table, in key order.</summary>
    public static List<KeyValuePair<string, byte[]>> ReadAll(byte[] file)
    {
        if (file.Length < FooterSize)
            throw new InvalidDataException("Checkpoint index is too small to be an SSTable.");

        var footer = file.AsSpan(file.Length - FooterSize);
        ulong magic = BinaryPrimitives.ReadUInt64LittleEndian(footer[^8..]);
        if (magic != TableMagic)
            throw new InvalidDataException("Checkpoint index has a bad SSTable magic number.");

        var reader = new ProtoReader(footer);
        _ = ReadHandle(ref reader); // metaindex handle, unused by TensorFlow
        var indexHandle = ReadHandle(ref reader);

        var entries = new List<KeyValuePair<string, byte[]>>();
        foreach (var (_, handleBytes) in ReadBlock(file, indexHandle))
        {
            var handleReader = new ProtoReader(handleBytes);
            var dataHandle = ReadHandle(ref handleReader);
            foreach (var (key, value) in ReadBlock(file, dataHandle))
                entries.Add(new(key, value));
        }
        return entries;
    }

    private static (long Offset, long Size) ReadHandle(ref ProtoReader reader) =>
        ((long)reader.ReadVarint(), (long)reader.ReadVarint());

    private static List<(string Key, byte[] Value)> ReadBlock(byte[] file, (long Offset, long Size) handle)
    {
        if (handle.Offset < 0 || handle.Offset + handle.Size + BlockTrailerSize > file.Length)
            throw new InvalidDataException("SSTable block handle points outside the index file.");

        int offset = checked((int)handle.Offset);
        int size = checked((int)handle.Size);
        byte compression = file[offset + size];
        if (compression != 0)
            throw new NotSupportedException(
                $"SSTable block uses compression type {compression}; only uncompressed blocks are supported.");

        var block = file.AsSpan(offset, size);
        int numRestarts = (int)BinaryPrimitives.ReadUInt32LittleEndian(block[^4..]);
        int entriesEnd = block.Length - 4 - 4 * numRestarts;
        if (entriesEnd < 0)
            throw new InvalidDataException("SSTable block has a bad restart count.");

        var result = new List<(string, byte[])>();
        var reader = new ProtoReader(block[..entriesEnd]);
        byte[] lastKey = [];
        while (!reader.AtEnd)
        {
            int shared = (int)reader.ReadVarint();
            int nonShared = (int)reader.ReadVarint();
            int valueLength = (int)reader.ReadVarint();
            var keyDelta = reader.ReadBytes(nonShared);
            var value = reader.ReadBytes(valueLength);

            if (shared > lastKey.Length)
                throw new InvalidDataException("SSTable entry shares more bytes than the previous key has.");
            var key = new byte[shared + nonShared];
            lastKey.AsSpan(0, shared).CopyTo(key);
            keyDelta.CopyTo(key.AsSpan(shared));
            lastKey = key;

            result.Add((System.Text.Encoding.UTF8.GetString(key), value.ToArray()));
        }
        return result;
    }
}
