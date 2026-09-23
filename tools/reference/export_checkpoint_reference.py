"""Dump a JSON summary of every tensor in the pretrained MIDI-DDSP checkpoints.

The C# CheckpointReader tests compare against this file, so the reader is
checked against TensorFlow's own checkpoint reader rather than against itself.

Usage:
  python export_checkpoint_reference.py <weights_dir> <output.json>

where <weights_dir> is the extracted `midi_ddsp_model_weights_urmp_9_10`
folder (see tools/download_weights.sh).
"""

import json
import os
import sys

import numpy as np
import tensorflow as tf

CHECKPOINTS = {
    'expression_generator': 'expression_generator/5000',
    'synthesis_generator': 'synthesis_generator/50000',
}
NUM_HEAD_VALUES = 8


def summarize(prefix):
  reader = tf.train.load_checkpoint(prefix)
  dtypes = reader.get_variable_to_dtype_map()
  tensors = []
  for name, shape in sorted(reader.get_variable_to_shape_map().items()):
    dtype = dtypes[name]
    entry = {'name': name, 'dtype': dtype.name, 'shape': list(shape)}
    if dtype.is_floating or dtype.is_integer:
      value = np.asarray(reader.get_tensor(name), dtype=np.float64)
      flat = value.reshape(-1)
      entry['sum'] = float(flat.sum())
      entry['sum_abs'] = float(np.abs(flat).sum())
      entry['head'] = [float(v) for v in flat[:NUM_HEAD_VALUES]]
    tensors.append(entry)
  return tensors


def main():
  if len(sys.argv) != 3:
    sys.exit(__doc__)
  weights_dir, output_path = sys.argv[1], sys.argv[2]
  reference = {
      'tensorflow_version': tf.__version__,
      'checkpoints': {
          key: summarize(os.path.join(weights_dir, rel))
          for key, rel in CHECKPOINTS.items()
      },
  }
  with open(output_path, 'w') as f:
    json.dump(reference, f, indent=1)
  for key, tensors in reference['checkpoints'].items():
    print(f'{key}: {len(tensors)} tensors')


if __name__ == '__main__':
  main()
