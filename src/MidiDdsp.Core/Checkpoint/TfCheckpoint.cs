using System.Numerics;

namespace MidiDdsp.Core.Checkpoint;

/// <summary>TensorFlow <c>DataType</c> enum values that can appear in a checkpoint.</summary>
public enum TfDataType
{
    Invalid = 0,
    Float = 1,
    Double = 2,
    Int32 = 3,
    UInt8 = 4,
    Int16 = 5,
    Int8 = 6,
    String = 7,
    Int64 = 9,
    Bool = 10,
    Half = 19,
}

/// <summary>Location and metadata of one tensor in a checkpoint.</summary>
public sealed record CheckpointTensorInfo(
    string Name,
    TfDataType DataType,
    long[] Shape,
    int ShardId,
    long Offset,
    long Size,
    uint MaskedCrc32c)
{
    public long ElementCount => Shape.Aggregate(1L, (a, b) => a * b);
}

/// <summary>
/// Reads a TensorFlow 2 checkpoint (the <c>prefix.index</c> +
/// <c>prefix.data-XXXXX-of-YYYYY</c> files written by <c>tf.train.Checkpoint</c>
/// or Keras <c>save_weights</c>) without TensorFlow.
/// </summary>
public sealed class TfCheckpoint : IDisposable
{
    /// <summary>Suffix TF2 object-based checkpoints append to every variable key.</summary>
    public const string VariableSuffix = "/.ATTRIBUTES/VARIABLE_VALUE";

    private readonly string _prefix;
    private readonly int _numShards;
    private readonly FileStream?[] _shards;
    private readonly Dictionary<string, CheckpointTensorInfo> _tensors;

    private TfCheckpoint(string prefix, int numShards, Dictionary<string, CheckpointTensorInfo> tensors)
    {
        _prefix = prefix;
        _numShards = numShards;
        _shards = new FileStream?[numShards];
        _tensors = tensors;
    }

    /// <summary>Every tensor in the checkpoint, keyed by its full checkpoint key.</summary>
    public IReadOnlyDictionary<string, CheckpointTensorInfo> Tensors => _tensors;

    /// <summary>
    /// Opens a checkpoint from its prefix, e.g. <c>.../synthesis_generator/50000</c>
    /// for <c>50000.index</c> and <c>50000.data-00000-of-00001</c>.
    /// </summary>
    public static TfCheckpoint Open(string prefix)
    {
        var indexPath = prefix + ".index";
        if (!File.Exists(indexPath))
            throw new FileNotFoundException($"Checkpoint index not found: {indexPath}", indexPath);

        int numShards = 1;
        var tensors = new Dictionary<string, CheckpointTensorInfo>(StringComparer.Ordinal);
        foreach (var (key, value) in SSTable.ReadAll(File.ReadAllBytes(indexPath)))
        {
            if (key.Length == 0)
                numShards = ParseHeader(value);
            else
                tensors[key] = ParseEntry(key, value);
        }
        return new TfCheckpoint(prefix, numShards, tensors);
    }

    /// <summary>Looks up a variable by its object path, without the <see cref="VariableSuffix"/>.</summary>
    public CheckpointTensorInfo GetVariable(string path) =>
        _tensors.TryGetValue(path + VariableSuffix, out var info)
            ? info
            : throw new KeyNotFoundException($"Variable '{path}' not found in checkpoint {_prefix}.");

    /// <summary>Reads a float32 tensor's values in row-major order, verifying its checksum.</summary>
    public float[] ReadFloats(CheckpointTensorInfo info)
    {
        if (info.DataType != TfDataType.Float)
            throw new InvalidOperationException($"Tensor '{info.Name}' is {info.DataType}, not Float.");
        if (info.Size != info.ElementCount * sizeof(float))
            throw new InvalidDataException(
                $"Tensor '{info.Name}' has {info.Size} bytes but shape needs {info.ElementCount * sizeof(float)}.");

        var bytes = ReadBytes(info);
        var values = new float[info.ElementCount];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    public float[] ReadFloats(string key) => ReadFloats(
        _tensors.TryGetValue(key, out var info)
            ? info
            : throw new KeyNotFoundException($"Tensor '{key}' not found in checkpoint {_prefix}."));

    /// <summary>Reads a tensor's raw bytes, verifying its checksum.</summary>
    public byte[] ReadBytes(CheckpointTensorInfo info)
    {
        var stream = GetShard(info.ShardId);
        var bytes = new byte[info.Size];
        stream.Seek(info.Offset, SeekOrigin.Begin);
        stream.ReadExactly(bytes);

        uint actual = MaskCrc(Crc32c(bytes));
        if (actual != info.MaskedCrc32c)
            throw new InvalidDataException($"Checksum mismatch for tensor '{info.Name}'.");
        return bytes;
    }

    public void Dispose()
    {
        foreach (var shard in _shards)
            shard?.Dispose();
    }

    private FileStream GetShard(int shardId)
    {
        if (shardId < 0 || shardId >= _numShards)
            throw new InvalidDataException($"Shard id {shardId} is out of range (checkpoint has {_numShards}).");
        return _shards[shardId] ??= File.OpenRead($"{_prefix}.data-{shardId:D5}-of-{_numShards:D5}");
    }

    // BundleHeaderProto: 1 num_shards, 2 endianness, 3 version.
    private static int ParseHeader(ReadOnlySpan<byte> data)
    {
        int numShards = 1;
        var reader = new ProtoReader(data);
        while (!reader.AtEnd)
        {
            var (field, wireType) = reader.ReadTag();
            switch (field)
            {
                case 1: numShards = (int)reader.ReadVarint(); break;
                case 2:
                    if (reader.ReadVarint() != 0)
                        throw new NotSupportedException("Big-endian checkpoints are not supported.");
                    break;
                default: reader.Skip(wireType); break;
            }
        }
        return numShards;
    }

    // BundleEntryProto: 1 dtype, 2 shape, 3 shard_id, 4 offset, 5 size, 6 crc32c, 7 slices.
    private static CheckpointTensorInfo ParseEntry(string name, ReadOnlySpan<byte> data)
    {
        var dtype = TfDataType.Invalid;
        long[] shape = [];
        int shardId = 0;
        long offset = 0, size = 0;
        uint crc = 0;

        var reader = new ProtoReader(data);
        while (!reader.AtEnd)
        {
            var (field, wireType) = reader.ReadTag();
            switch (field)
            {
                case 1: dtype = (TfDataType)reader.ReadVarint(); break;
                case 2: shape = ParseShape(reader.ReadLengthDelimited()); break;
                case 3: shardId = (int)reader.ReadVarint(); break;
                case 4: offset = (long)reader.ReadVarint(); break;
                case 5: size = (long)reader.ReadVarint(); break;
                case 6: crc = reader.ReadFixed32(); break;
                case 7: throw new NotSupportedException($"Tensor '{name}' is partitioned into slices, which is not supported.");
                default: reader.Skip(wireType); break;
            }
        }
        return new CheckpointTensorInfo(name, dtype, shape, shardId, offset, size, crc);
    }

    // TensorShapeProto: 2 repeated Dim { 1 size, 2 name }, 3 unknown_rank.
    private static long[] ParseShape(ReadOnlySpan<byte> data)
    {
        var dims = new List<long>();
        var reader = new ProtoReader(data);
        while (!reader.AtEnd)
        {
            var (field, wireType) = reader.ReadTag();
            if (field != 2)
            {
                reader.Skip(wireType);
                continue;
            }
            long size = 0;
            var dimReader = new ProtoReader(reader.ReadLengthDelimited());
            while (!dimReader.AtEnd)
            {
                var (dimField, dimWireType) = dimReader.ReadTag();
                if (dimField == 1)
                    size = (long)dimReader.ReadVarint();
                else
                    dimReader.Skip(dimWireType);
            }
            dims.Add(size);
        }
        return dims.ToArray();
    }

    /// <summary>CRC-32C (Castagnoli), as used by TensorFlow and LevelDB.</summary>
    internal static uint Crc32c(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;
        int i = 0;
        for (; i + 8 <= data.Length; i += 8)
            crc = BitOperations.Crc32C(crc, BitConverter.ToUInt64(data.Slice(i, 8)));
        for (; i < data.Length; i++)
            crc = BitOperations.Crc32C(crc, data[i]);
        return ~crc;
    }

    /// <summary>LevelDB/TensorFlow checksum masking (crc32c::Mask).</summary>
    internal static uint MaskCrc(uint crc) => ((crc >> 15) | (crc << 17)) + 0xa282ead8u;
}
