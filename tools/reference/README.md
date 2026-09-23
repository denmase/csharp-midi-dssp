# Reference data from the original MIDI-DDSP

The C# port is checked against numbers produced by the original Python/TensorFlow
code. The scripts here generate that data; their output is committed under
`tests/MidiDdsp.Tests/Reference/`, so you only need this environment when
adding or regenerating reference data, not to build or test the C# code.

## Environment

MIDI-DDSP targets TensorFlow 2.x of early 2022 and `ddsp==3.2.0`, so it needs
Python 3.9. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv -p 3.9 .venv-ref
VIRTUAL_ENV=.venv-ref uv pip install -r tools/reference/requirements.txt
```

That is enough for `export_checkpoint_reference.py`. Scripts that run the
models themselves also need `ddsp` and the `midi-ddsp` source. `ddsp==3.2.0`
pins `crepe`, which no longer builds and is only used for training losses, so
install `ddsp` without dependencies and add what it imports:

```bash
VIRTUAL_ENV=.venv-ref uv pip install "tensorflow-probability==0.16.0" \
  "tensorflow-addons==0.16.1" "tensorflow-datasets==4.6.0" gin-config==0.5.0 \
  librosa==0.9.2 note_seq==0.0.3 pretty_midi pandas tqdm dill mir_eval \
  matplotlib hmmlearn cloudml-hypertune "google-cloud-storage<2.10" \
  "protobuf==3.19.6"
VIRTUAL_ENV=.venv-ref uv pip install --no-deps ddsp==3.2.0
echo '"""Stub: crepe is only used by ddsp training losses."""' \
  > .venv-ref/lib/python3.9/site-packages/crepe.py
git clone https://github.com/magenta/midi-ddsp   # put it on PYTHONPATH
```

## Scripts

| Script | Output |
|---|---|
| `export_checkpoint_reference.py <weights_dir> <out.json>` | Name, dtype, shape, sums and first values of every tensor in both checkpoints (`checkpoint_reference.json`). |
| `export_layer_reference.py <weights_dir> <out.npz>` | Inputs and outputs of pretrained layers and of the full Expression Generator on seeded inputs (`layer_reference.npz`). Needs the `midi-ddsp` source on `PYTHONPATH`. |
| `export_synthesis_reference.py <weights_dir> <out.npz>` | The original pipeline from a note list to synthesis parameters, with argmax f0 (`synthesis_reference.npz`). Run with `tools/reference` and the `midi-ddsp` source on `PYTHONPATH`. |
| `export_ddsp_reference.py <weights_dir> <synthesis_reference.npz> <out.npz>` | ddsp's harmonic synth, filtered noise (on a saved noise signal) and the pretrained reverb applied to the first 300 frames of the synthesis reference (`ddsp_reference.npz`). |
| `export_pipeline_reference.py <weights_dir> <out.mid> <out.npz>` | Writes a multi-part test MIDI file, records how `pretty_midi` reads it, and runs the original `synthesize_midi` on it with argmax f0 (`pipeline_test.mid`, `pipeline_reference.npz`). |
| `export_fluidsynth_reference.py <soundfont.sf2> <out.mid> <out.npz>` | Writes a MIDI file with non-URMP parts, pitch bends and control changes, and records how `pretty_midi` reads it and renders it with FluidSynth (`fluidsynth_test.mid`, `fluidsynth_reference.npz`; TimGM6mb, pyfluidsynth 1.3.0, libfluidsynth 2.3.4). |
