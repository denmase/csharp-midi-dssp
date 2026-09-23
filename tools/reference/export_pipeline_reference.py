"""Run the original synthesize_midi end to end on a generated MIDI file.

Writes a small multi-part MIDI file (with a tempo change, overlapping notes, a
part MIDI-DDSP cannot play and a drum part), then records how pretty_midi
reads it and the control parameters and audio that synthesize_midi produces.
The f0 decoder uses argmax so the controls are deterministic; the audio still
contains TensorFlow's random noise.

Usage:
  PYTHONPATH=<midi-ddsp checkout>:tools/reference \
    python export_pipeline_reference.py <weights_dir> <out.mid> <out.npz>
"""

import sys

import mido
import numpy as np
import pretty_midi
import tensorflow as tf

from midi_ddsp.midi_ddsp_synthesize import synthesize_midi
from export_layer_reference import load_models

SEED = 2024


def write_midi(path):
  """A type-1 file written with mido so tempo and event order are explicit."""
  tpb = 480
  midi = mido.MidiFile(type=1, ticks_per_beat=tpb)

  conductor = mido.MidiTrack()
  conductor.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
  conductor.append(mido.MetaMessage('set_tempo', tempo=400000, time=4 * tpb))
  midi.tracks.append(conductor)

  def track(channel, program, notes):
    """notes: (start_tick, end_tick, pitch); emitted in time order."""
    t = mido.MidiTrack()
    t.append(mido.MetaMessage('track_name', name=f'ch{channel}', time=0))
    t.append(mido.Message('program_change', channel=channel, program=program,
                          time=0))
    events = []
    for start, end, pitch in notes:
      events.append((start, 1, mido.Message('note_on', channel=channel,
                                            note=pitch, velocity=90)))
      # Note-off as note_on with velocity 0 on odd pitches, to cover both.
      off = (mido.Message('note_on', channel=channel, note=pitch, velocity=0)
             if pitch % 2 else
             mido.Message('note_off', channel=channel, note=pitch))
      events.append((end, 0, off))
    events.sort(key=lambda e: (e[0], e[1]))
    now = 0
    for tick, _, message in events:
      t.append(message.copy(time=tick - now))
      now = tick
    midi.tracks.append(t)

  # Violin (program 40): legato line crossing the tempo change, one overlap.
  track(0, 40, [(0, 240, 67), (240, 480, 69), (480, 960, 71),
                (1200, 1440, 72), (1440, 1700, 74), (1680, 1920, 72),
                (1920, 2400, 71), (2400, 2880, 69)])
  # Flute (program 73): starts late, with rests.
  track(1, 73, [(480, 720, 79), (960, 1200, 81), (1200, 1680, 83),
                (2160, 2640, 81), (2640, 3120, 79)])
  # Acoustic guitar (program 24): not a MIDI-DDSP instrument, skipped.
  track(2, 24, [(0, 960, 55), (960, 1920, 57)])
  # Drums on channel 10 with program 0: also skipped.
  track(9, 0, [(0, 120, 36), (480, 600, 38)])
  midi.save(path)


def main():
  if len(sys.argv) != 4:
    sys.exit(__doc__)
  weights_dir, midi_path, output_path = sys.argv[1:]
  tf.random.set_seed(SEED)
  write_midi(midi_path)

  arrays = {}
  pm = pretty_midi.PrettyMIDI(midi_path)
  for i, instrument in enumerate(pm.instruments):
    arrays[f'pm{i}_program'] = np.array([instrument.program], dtype=np.int64)
    arrays[f'pm{i}_start'] = np.array([n.start for n in instrument.notes])
    arrays[f'pm{i}_end'] = np.array([n.end for n in instrument.notes])
    arrays[f'pm{i}_pitch'] = np.array([n.pitch for n in instrument.notes],
                                      dtype=np.int64)
  arrays['pm_instrument_count'] = np.array([len(pm.instruments)],
                                           dtype=np.int64)

  sg, eg = load_models(weights_dir)
  sg.midi_decoder.decoder.midi_to_f0.sampling_method = 'argmax'
  output = synthesize_midi(sg, eg, midi_path, display_progressbar=False)

  arrays['parts'] = np.array(output['part_synth_by_model'], dtype=np.int64)
  for part in output['part_synth_by_model']:
    controls = output['midi_control_params'][part]
    for key in ('f0_hz', 'amplitudes', 'harmonic_distribution',
                'noise_magnitudes'):
      arrays[f'part{part}_{key}'] = controls[key].astype(np.float32)
  arrays['mix_audio'] = output['mix_audio'].astype(np.float32)

  np.savez_compressed(output_path, **arrays)
  for name, value in arrays.items():
    print(f'{name}: {value.dtype} {list(value.shape)}')


if __name__ == '__main__':
  main()
