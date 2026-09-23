using MidiDdsp.Core.Checkpoint;

namespace MidiDdsp.Core.Nn;

/// <summary>
/// A view of a checkpoint rooted at an object path, used by layers to load
/// their variables by relative name (e.g. <c>kernel</c>, <c>bias</c>).
/// </summary>
public sealed class WeightScope
{
    private readonly TfCheckpoint _checkpoint;
    private readonly string _path;

    public WeightScope(TfCheckpoint checkpoint, string path = "")
    {
        _checkpoint = checkpoint;
        _path = path.Trim('/');
    }

    public WeightScope this[string child] =>
        new(_checkpoint, _path.Length == 0 ? child : $"{_path}/{child}");

    /// <summary>Loads a float32 variable, checking its rank.</summary>
    public (float[] Values, int[] Shape) Load(string name, int rank)
    {
        var info = _checkpoint.GetVariable(_path.Length == 0 ? name : $"{_path}/{name}");
        if (info.Shape.Length != rank)
            throw new InvalidDataException(
                $"Variable '{info.Name}' has shape [{string.Join(", ", info.Shape)}], expected rank {rank}.");
        return (_checkpoint.ReadFloats(info), info.Shape.Select(d => checked((int)d)).ToArray());
    }

    /// <summary>Loads a per-channel vector stored with any number of leading 1-dimensions.</summary>
    public float[] LoadVector(string name)
    {
        var info = _checkpoint.GetVariable(_path.Length == 0 ? name : $"{_path}/{name}");
        if (info.Shape.Length == 0 || info.Shape.Take(info.Shape.Length - 1).Any(d => d != 1))
            throw new InvalidDataException(
                $"Variable '{info.Name}' has shape [{string.Join(", ", info.Shape)}], expected a vector.");
        return _checkpoint.ReadFloats(info);
    }
}
