"""Record reference outputs of the pretrained MIDI-DDSP networks.

Runs parts of the original models on fixed, seeded inputs and saves inputs and
outputs to an .npz file that the C# tests compare against.

Usage:
  PYTHONPATH=<midi-ddsp checkout> \
    python export_layer_reference.py <weights_dir> <output.npz>

Only deterministic computations are recorded here. The Expression Generator's
autoregressive inference is deterministic (it does not sample), so it is
recorded end to end.
"""

import os
import sys

import numpy as np
import tensorflow as tf

from midi_ddsp.hparams_synthesis_generator import hparams as hp
from midi_ddsp.modules.expression_generator import ExpressionGenerator, \
  get_fake_data_expression_generator
from midi_ddsp.modules.get_synthesis_generator import get_synthesis_generator, \
  get_fake_data_synthesis_generator
from midi_ddsp.utils.training_utils import get_hp

SEED = 1234
NUM_NOTES = 12
NUM_FRAMES = 40


def load_models(weights_dir):
  """Same as midi_ddsp_synthesize.load_pretrained_model, with explicit paths."""
  sg_path = os.path.join(weights_dir, 'synthesis_generator', '50000')
  eg_path = os.path.join(weights_dir, 'expression_generator', '5000')

  for k, v in get_hp(os.path.join(os.path.dirname(sg_path),
                                  'train.log')).items():
    setattr(hp, k, v)
  synthesis_generator = get_synthesis_generator(hp)
  synthesis_generator._build(get_fake_data_synthesis_generator(hp))
  synthesis_generator.load_weights(sg_path).expect_partial()

  n_out = 6
  expression_generator = ExpressionGenerator(n_out=n_out, nhid=128)
  fake_data = get_fake_data_expression_generator(n_out)
  _ = expression_generator(fake_data['cond'], out=fake_data['target'],
                           training=True)
  expression_generator.load_weights(eg_path).expect_partial()
  return synthesis_generator, expression_generator


def expression_generator_reference(eg, rng):
  # Pitches include rests (0), whose expression output is masked to zero.
  note_pitch = rng.integers(40, 90, size=[1, NUM_NOTES]).astype(np.int64)
  note_pitch[0, [3, 8]] = 0
  note_length = rng.uniform(0.02, 1.5, size=[1, NUM_NOTES, 1]).astype(
    np.float32)
  instrument_id = np.array([5], dtype=np.int64)
  cond = {
    'note_pitch': tf.constant(note_pitch),
    'note_length': tf.constant(note_length),
    'instrument_id': tf.constant(instrument_id),
  }
  encoded = eg.encode_cond(cond, training=False)
  outputs = eg(cond, training=False)
  return {
    'eg_note_pitch': note_pitch[0],
    'eg_note_length': note_length[0, :, 0],
    'eg_instrument_id': instrument_id,
    'eg_encoded_cond': encoded.numpy()[0],
    'eg_output': outputs['output'].numpy()[0],
  }


def synthesis_generator_reference(sg, rng):
  midi_decoder = sg.midi_decoder
  decoder = midi_decoder.decoder

  def normal(*shape):
    return rng.standard_normal(size=[1, *shape]).astype(np.float32)

  precond_in = normal(NUM_FRAMES, 10)
  f0_birnn_in = normal(NUM_FRAMES, 320)
  harmonic_in = normal(NUM_FRAMES, 384)
  f0_head_in = normal(1, 256)
  q_pitch_in = rng.uniform(40, 90, size=[1, NUM_FRAMES, 1]).astype(np.float32)

  harmonic_out = decoder.midi_f0_to_harmonic(tf.constant(harmonic_in))
  return {
    'sg_precond_in': precond_in[0],
    'sg_precond_out':
      midi_decoder.z_preconditioning_stack(precond_in).numpy()[0],
    'sg_instrument_emb': midi_decoder.instrument_emb(
      tf.constant([0, 7, 19], dtype=tf.int64)).numpy(),
    'sg_f0_birnn_in': f0_birnn_in[0],
    'sg_f0_birnn_out': decoder.midi_to_f0.birnn(f0_birnn_in).numpy()[0],
    # decode_out on a single frame, as in the autoregressive loop.
    'sg_f0_head_in': f0_head_in[0],
    'sg_f0_head_out':
      decoder.midi_to_f0.decode_out(tf.constant(f0_head_in)).numpy()[0],
    'sg_q_pitch_in': q_pitch_in[0],
    'sg_q_pitch_emb_out': decoder.q_pitch_emb(q_pitch_in / 127).numpy()[0],
    'sg_harmonic_in': harmonic_in[0],
    'sg_harmonic_amplitudes': harmonic_out['amplitudes'].numpy()[0],
    'sg_harmonic_distribution':
      harmonic_out['harmonic_distribution'].numpy()[0],
    'sg_harmonic_noise_magnitudes':
      harmonic_out['noise_magnitudes'].numpy()[0],
  }


def main():
  if len(sys.argv) != 3:
    sys.exit(__doc__)
  weights_dir, output_path = sys.argv[1], sys.argv[2]
  tf.random.set_seed(SEED)
  rng = np.random.default_rng(SEED)
  sg, eg = load_models(weights_dir)

  arrays = {}
  arrays.update(expression_generator_reference(eg, rng))
  arrays.update(synthesis_generator_reference(sg, rng))
  np.savez_compressed(output_path, **arrays)
  for name, value in arrays.items():
    print(f'{name}: {value.dtype} {list(value.shape)}')


if __name__ == '__main__':
  main()
