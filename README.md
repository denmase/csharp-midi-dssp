# MIDI-DDSP for C#

A port of [MIDI-DDSP](https://github.com/magenta/midi-ddsp) (Wu et al., ICLR 2022)
to .NET, aimed at turning MIDI into audio with the **pretrained** MIDI-DDSP
models, written in plain C# with no TensorFlow dependency.

## Status

Working end to end: MIDI file in, WAV out, using the pretrained models.

- [x] Read the pretrained TensorFlow checkpoints directly from C#
  (`MidiDdsp.Core.Checkpoint.TfCheckpoint`), verified tensor-by-tensor against
  TensorFlow's own reader.
- [x] Neural network layers (`MidiDdsp.Core.Nn`): Dense, Embedding, GRU and
  bidirectional GRU, Keras and ddsp layer norms, `FcStackOut`, dilated conv
  stack; each checked against the pretrained TensorFlow layers.
- [x] Expression Generator (`MidiDdsp.Core.Models.ExpressionGenerator`),
  matching the original end to end.
- [x] Synthesis Generator (`MidiDdsp.Core.Models.SynthesisGenerator`): note
  sequence, frame conditioning, and the MIDI decoder with top-p or argmax f0
  sampling; matches the original's synthesis parameters with argmax sampling.
- [x] DDSP synthesis (`MidiDdsp.Core.Dsp`): harmonic synthesizer, filtered
  noise and the per-instrument learned reverb, matching ddsp 3.2.0.
- [x] MIDI input (read the way `pretty_midi` reads it), WAV output and an
  end-to-end CLI, checked against the original `synthesize_midi`.

Only the inference path is ported. The DDSP Inference module (the audio
encoder used during training) is not needed to synthesize MIDI.

## Usage

```bash
tools/download_weights.sh
dotnet run -c Release --project src/MidiDdsp.Cli -- synthesize song.mid song.wav
```

Options: `--seed <n>` for repeatable output, `--argmax` for deterministic
pitch, `--pitch-offset <n>`, `--speed <rate>`, `--stems <dir>` to also write
each part, `--float` for 32-bit float WAV, `--weights <dir>`.

Parts whose General MIDI program is one of the 13 URMP instruments (violin,
viola, cello, double bass, flute, oboe, clarinet, saxophone, bassoon, trumpet,
horn, trombone, tuba) are synthesized; other parts are skipped. Each part
should be monophonic, as in the original.

From code:

```csharp
var synthesizer = MidiDdspSynthesizer.Load("weights/midi_ddsp_model_weights_urmp_9_10");
var result = synthesizer.Synthesize(MidiFile.Load("song.mid"), new SynthesisOptions { Seed = 1 });
WavWriter.WritePcm16("song.wav", result.Mix, DdspMath.SampleRate);
```

### Differences from the original

- Random numbers differ from TensorFlow's, so the sampled pitch detail and the
  noise are different on each run (as they are between runs of the original).
  With `--argmax` the synthesis parameters match the original's.
- Parts the models cannot play are skipped; the original can optionally render
  them with FluidSynth.
- 16-bit output is clipped to [−1, 1].

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
