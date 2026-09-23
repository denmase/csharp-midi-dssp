# MIDI-DDSP for C#

[![CI](https://github.com/denmase/csharp-midi-dssp/actions/workflows/ci.yml/badge.svg)](https://github.com/denmase/csharp-midi-dssp/actions/workflows/ci.yml)

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
- [x] FluidSynth fallback for instruments the models cannot play.

Only the inference path is ported. The DDSP Inference module (the audio
encoder used during training) is not needed to synthesize MIDI.

## Usage

```bash
tools/download_weights.sh
dotnet run -c Release --project src/MidiDdsp.Cli -- synthesize song.mid song.wav
```

The CLI looks for the weights in `$MIDI_DDSP_WEIGHTS`, then in
`weights/midi_ddsp_model_weights_urmp_9_10` next to the program (where the CI
builds bundle them), then under the current directory; `--weights <dir>`
overrides this.

Options: `--seed <n>` for repeatable output, `--argmax` for deterministic
pitch, `--pitch-offset <n>`, `--speed <rate>`, `--stems <dir>` to also write
each part, `--float` for 32-bit float WAV, `--weights <dir>`,
`--fluidsynth [--soundfont <file.sf2>]` (see below).

Parts whose General MIDI program is one of the 13 URMP instruments (violin,
viola, cello, double bass, flute, oboe, clarinet, saxophone, bassoon, trumpet,
horn, trombone, tuba) are synthesized; other parts are skipped, or rendered
with FluidSynth with `--fluidsynth`. Each part should be monophonic, as in the
original.

### FluidSynth fallback

`--fluidsynth` renders the other parts (including drums) with
[FluidSynth](https://www.fluidsynth.org/) and a General MIDI soundfont, as the
original's `use_fluidsynth` option does. It needs the FluidSynth 2.x shared
library, loaded at run time:

- Linux: `sudo apt install libfluidsynth3 fluid-soundfont-gm`
- macOS: `brew install fluid-synth`
- Windows: a FluidSynth release, with `libfluidsynth-3.dll` on `PATH`

Set `MIDI_DDSP_FLUIDSYNTH_LIBRARY` to the library's path if it is not found.
Without `--soundfont`, `FluidR3_GM.sf2` (the original's default) or another GM
soundfont in `/usr/share/sounds/sf2` is used.

The original's fallback does not work as released: it fails with a `KeyError`
for every program except 26, and it adds pyfluidsynth's 16-bit integer samples
(× 0.25) to model audio in [−1, 1]. This port does what it evidently meant:
FluidSynth output converted to [−1, 1], at 0.25 volume. Rendering itself
matches `pretty_midi`'s `Instrument.fluidsynth()` up to FluidSynth's dither.

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
- The FluidSynth fallback is fixed as described above.
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
dotnet publish src/MidiDdsp.Cli -c Release -p:UseAppHost=false -o out   # portable build
dotnet out/midi-ddsp.dll synthesize song.mid song.wav
```

Tests that need the pretrained weights look in `./weights` or in the directory
named by `MIDI_DDSP_WEIGHTS`, and are skipped if neither exists. FluidSynth
tests need libfluidsynth and `TimGM6mb.sf2` (`timgm6mb-soundfont`, or set
`MIDI_DDSP_SOUNDFONT`), and are skipped otherwise.

A self-contained Windows executable, which needs no .NET installed:

```bash
dotnet publish src/MidiDdsp.Cli -c Release -r win-x64 --self-contained \
  -p:PublishSingleFile=true -p:EnableCompressionInSingleFile=true -p:DebugType=None -o out-win
out-win\midi-ddsp.exe synthesize song.mid song.wav --weights <weights-dir>
```

CI (GitHub Actions) runs one job on Linux that builds, runs every test with
the weights and FluidSynth installed, and uploads two artifacts:

- `midi-ddsp-portable`: framework-dependent and platform-independent, run with
  `dotnet midi-ddsp.dll` wherever the .NET 8 runtime is installed.
- `midi-ddsp-win-x64`: the self-contained `midi-ddsp.exe` (about 35 MB),
  cross-compiled for Windows x64.

Both include the pretrained weights in `weights/` (with a `NOTICE.txt` on
where they come from), so they run out of the box:
`midi-ddsp.exe synthesize song.mid song.wav`. The portable build is
smoke-tested with its bundled weights before upload.

## Verifying against the original

Correctness is checked against reference data produced by the original
Python/TensorFlow implementation and committed under
`tests/MidiDdsp.Tests/Reference/`. See [tools/reference](tools/reference/README.md)
for how that data is generated.

## License

MIDI-DDSP is Apache-2.0. The pretrained weights were trained on the
[URMP dataset](https://labsites.rochester.edu/air/projects/URMP.html); check its
terms before redistributing them or audio made with them.
