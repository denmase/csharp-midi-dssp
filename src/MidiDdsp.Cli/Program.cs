using MidiDdsp.Core.Checkpoint;

if (args.Length != 2 || args[0] != "list-weights")
{
    Console.Error.WriteLine("Usage: midi-ddsp list-weights <checkpoint-prefix>");
    Console.Error.WriteLine("  e.g. midi-ddsp list-weights weights/midi_ddsp_model_weights_urmp_9_10/synthesis_generator/50000");
    return 1;
}

using var checkpoint = TfCheckpoint.Open(args[1]);
long totalParams = 0;
foreach (var info in checkpoint.Tensors.Values.OrderBy(t => t.Name, StringComparer.Ordinal))
{
    if (info.DataType == TfDataType.String)
        continue; // object graph metadata, not a weight
    var name = info.Name.EndsWith(TfCheckpoint.VariableSuffix)
        ? info.Name[..^TfCheckpoint.VariableSuffix.Length]
        : info.Name;
    Console.WriteLine($"{name}  {info.DataType}  [{string.Join(", ", info.Shape)}]");
    totalParams += info.ElementCount;
}
Console.WriteLine($"{totalParams:N0} parameters");
return 0;
