"""Record the original DDSP synthesis on the Synthesis Generator's outputs.

Takes the synthesis parameters from synthesis_reference.npz (the first
NUM_FRAMES frames) and runs them through the same ddsp processors the
original builds in get_process_group, plus the pretrained reverb.
FilteredNoise draws its own random noise, so the noise stage is recorded with
core.frequency_filter on a saved noise signal instead.

Usage:
  PYTHONPATH=<midi-ddsp checkout>:tools/reference \
    python export_ddsp_reference.py <weights_dir> <synthesis_reference.npz> \
      <output.npz>
"""

import sys

import ddsp
import numpy as np
import tensorflow as tf

from midi_ddsp.utils.inference_utils import get_process_group
from export_layer_reference import load_models

SEED = 99
NUM_FRAMES = 300
FRAME_SIZE = 64


def main():
  if len(sys.argv) != 4:
    sys.exit(__doc__)
  weights_dir, synthesis_path, output_path = sys.argv[1:]
  params = np.load(synthesis_path)
  instrument_id = int(params['instrument_id'][0])

  def frames(name):
    x = params[name][:NUM_FRAMES]
    return tf.constant(x.reshape(1, NUM_FRAMES, -1))

  processor_group = get_process_group(NUM_FRAMES, FRAME_SIZE,
                                      use_angular_cumsum=True)
  harmonic = processor_group.processors[0]
  noise_synth = processor_group.processors[1]

  harmonic_controls = harmonic.get_controls(
    frames('short_amplitudes'), frames('short_harmonic_distribution'),
    frames('short_f0_hz'))
  harmonic_audio = harmonic.get_signal(**harmonic_controls)

  noise_controls = noise_synth.get_controls(frames('short_noise_magnitudes'))
  rng = np.random.default_rng(SEED)
  white_noise = rng.uniform(-1, 1, size=[1, NUM_FRAMES * FRAME_SIZE]).astype(
    np.float32)
  noise_audio = ddsp.core.frequency_filter(
    white_noise, noise_controls['magnitudes'],
    window_size=noise_synth.window_size)

  dry = harmonic_audio + noise_audio
  sg, _ = load_models(weights_dir)
  wet = sg.reverb_module(dry, reverb_number=tf.constant([instrument_id]),
                         training=False)

  arrays = {
    'num_frames': np.array([NUM_FRAMES], dtype=np.int64),
    'amplitudes': harmonic_controls['amplitudes'].numpy()[0, :, 0],
    'harmonic_distribution':
      harmonic_controls['harmonic_distribution'].numpy()[0],
    'noise_magnitudes': noise_controls['magnitudes'].numpy()[0],
    'harmonic_audio': harmonic_audio.numpy()[0],
    'white_noise': white_noise[0],
    'noise_audio': noise_audio.numpy()[0],
    'reverb_audio': wet.numpy()[0],
  }
  np.savez_compressed(output_path, **arrays)
  for name, value in arrays.items():
    print(f'{name}: {value.dtype} {list(value.shape)}')


if __name__ == '__main__':
  main()
