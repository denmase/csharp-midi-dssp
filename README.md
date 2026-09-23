# MIDI-DDSP for C#

A port of [MIDI-DDSP](https://github.com/magenta/midi-ddsp) (Wu et al., ICLR 2022)
to .NET, aimed at turning MIDI into audio with the **pretrained** MIDI-DDSP
models, written in plain C# with no TensorFlow dependency.

## Status

Early work in progress.

- [x] Read the pretrained TensorFlow checkpoints directly from C#
  (`MidiDdsp.Core.Checkpoint.TfCheckpoint`), verified tensor-by-tensor against
  TensorFlow's own reader.
- [ ] Neural network layers (Dense, Embedding, GRU, LayerNorm, dilated Conv1D)
- [ ] Expression Generator
- [ ] Synthesis Generator (MIDI decoder path used for synthesis)
- [ ] DDSP synthesis: harmonic, filtered noise, reverb
- [ ] MIDI input, WAV output, end-to-end CLI

Only the inference path is ported. The DDSP Inference module (the audio
encoder used during training) is not needed to synthesize MIDI.

## Pretrained weights

The weights are published on the `models` branch of `magenta/midi-ddsp` as
TensorFlow checkpoints. They are loaded as-is; no conversion step is needed.

```bash
tools/download_weights.sh          # extracts to ./weights (git-ignored)
dotnet run --project src/MidiDdsp.Cli -- list-weights \
  weights/midi_ddsp_model_weights_urmp_9_10/synthesis_generator/50000
```

## Build and test

Requires the .NET 8 SDK.

```bash
dotnet test
```

Tests that need the pretrained weights look in `./weights` or in the directory
named by `MIDI_DDSP_WEIGHTS`, and are skipped if neither exists.

## Verifying against the original

Correctness is checked against reference data produced by the original
Python/TensorFlow implementation and committed under
`tests/MidiDdsp.Tests/Reference/`. See [tools/reference](tools/reference/README.md)
for how that data is generated.

## License

MIDI-DDSP is Apache-2.0. The pretrained weights were trained on the
[URMP dataset](https://labsites.rochester.edu/air/projects/URMP.html); check its
terms before redistributing them or audio made with them.
