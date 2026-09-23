"""Record the Synthesis Generator's behaviour on a fixed note list.

Follows midi_ddsp_synthesize.synthesize_midi up to the synthesis parameters:
note list -> note sequence -> Expression Generator -> conditioning_df ->
frame-level conditioning -> Synthesis Generator MIDI decoder. The f0 decoder
normally samples (top-p); here it uses argmax so the result is deterministic.

Usage:
  PYTHONPATH=<midi-ddsp checkout> \
    python export_synthesis_reference.py <weights_dir> <output.npz>
"""

import sys

import numpy as np
import pretty_midi
import tensorflow as tf

from midi_ddsp.utils.inference_utils import conditioning_df_to_dict, \
  conditioning_df_to_midi_features
from midi_ddsp.utils.midi_synthesis_utils import note_list_to_sequence, \
  expression_generator_output_to_conditioning_df
from export_layer_reference import load_models

SEED = 4321
INSTRUMENT_ID = 2
FRAME_RATE = 250


def make_notes(rng, count, min_len, max_len, rest_every=4, start=None):
  """Notes with random pitch and length, with gaps before some of them."""
  notes = []
  t = float(rng.uniform(0.05, 0.2)) if start is None else start
  for i in range(count):
    length = float(rng.uniform(min_len, max_len))
    pitch = int(rng.integers(55, 80))
    notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=t,
                                  end=t + length))
    t += length
    if i % rest_every == rest_every - 1:
      t += float(rng.uniform(0.05, 0.15))
  return notes


def conditioning_inputs(eg, notes):
  note_sequence = note_list_to_sequence(notes, fs=FRAME_RATE)
  note_sequence['instrument_id'] = tf.constant([INSTRUMENT_ID])
  eg_output = eg(note_sequence, out=None, training=False)['output']
  conditioning_df = expression_generator_output_to_conditioning_df(
    eg_output, note_sequence)
  conditioning_dict = conditioning_df_to_dict(conditioning_df)
  midi_features = tuple(
    tf.convert_to_tensor(f) for f in
    conditioning_df_to_midi_features(conditioning_df))
  return note_sequence, eg_output, conditioning_dict, midi_features


def capture_z_conditioning(midi_decoder, conditioning_dict, midi_features,
                           run_decoder):
  """Runs gen_params_from_cond, capturing the preconditioning stack's input."""
  captured = {}
  stack = midi_decoder.z_preconditioning_stack
  original_decoder = midi_decoder.decoder

  def recording_stack(z):
    captured['z_conditioning'] = z
    return stack(z)

  midi_decoder.z_preconditioning_stack = recording_stack
  if not run_decoder:
    midi_decoder.decoder = lambda *args, **kwargs: {}
  try:
    z_midi_decoder, params = midi_decoder.gen_params_from_cond(
      conditioning_dict, midi_features,
      instrument_id=tf.constant([INSTRUMENT_ID]))
  finally:
    midi_decoder.z_preconditioning_stack = stack
    midi_decoder.decoder = original_decoder
  return captured['z_conditioning'], z_midi_decoder, params


def main():
  if len(sys.argv) != 3:
    sys.exit(__doc__)
  weights_dir, output_path = sys.argv[1], sys.argv[2]
  tf.random.set_seed(SEED)
  rng = np.random.default_rng(SEED)
  sg, eg = load_models(weights_dir)
  sg.midi_decoder.decoder.midi_to_f0.sampling_method = 'argmax'

  arrays = {'instrument_id': np.array([INSTRUMENT_ID], dtype=np.int64)}

  # Short phrase, run through the whole MIDI decoder.
  notes = make_notes(rng, 10, 0.08, 0.3)
  note_sequence, eg_output, conditioning_dict, midi_features = \
    conditioning_inputs(eg, notes)
  z_cond, z_midi_decoder, params = capture_z_conditioning(
    sg.midi_decoder, conditioning_dict, midi_features, run_decoder=True)
  arrays.update({
    'short_note_start': np.array([n.start for n in notes]),
    'short_note_end': np.array([n.end for n in notes]),
    'short_note_pitch': np.array([n.pitch for n in notes], dtype=np.int64),
    'short_seq_pitch': note_sequence['note_pitch'].numpy()[0],
    'short_seq_length': note_sequence['note_length'].numpy()[0, :, 0],
    'short_expression': eg_output.numpy()[0],
    'short_z_conditioning': z_cond.numpy()[0],
    'short_z_midi_decoder': z_midi_decoder.numpy()[0],
    'short_f0_midi': params['f0_output']['f0_midi'].numpy()[0, :, 0],
    'short_f0_hz': params['f0_hz'].numpy()[0, :, 0],
    'short_amplitudes': params['amplitudes'].numpy()[0, :, 0],
    'short_harmonic_distribution': params['harmonic_distribution'].numpy()[0],
    'short_noise_magnitudes': params['noise_magnitudes'].numpy()[0],
  })

  # Long phrase with more than 100 note regions, conditioning only. It starts
  # at time 0, so the sequence begins with a zero-length rest.
  notes = make_notes(rng, 70, 0.04, 0.08, rest_every=2, start=0.0)
  note_sequence, eg_output, conditioning_dict, midi_features = \
    conditioning_inputs(eg, notes)
  z_cond, _, _ = capture_z_conditioning(
    sg.midi_decoder, conditioning_dict, midi_features, run_decoder=False)
  arrays.update({
    'long_note_start': np.array([n.start for n in notes]),
    'long_note_end': np.array([n.end for n in notes]),
    'long_note_pitch': np.array([n.pitch for n in notes], dtype=np.int64),
    'long_expression': eg_output.numpy()[0],
    'long_z_conditioning': z_cond.numpy()[0],
  })

  np.savez_compressed(output_path, **arrays)
  for name, value in arrays.items():
    print(f'{name}: {value.dtype} {list(value.shape)}')


if __name__ == '__main__':
  main()
