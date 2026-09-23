#!/usr/bin/env bash
# Download and extract the pretrained MIDI-DDSP weights (URMP, v9/10) from the
# `models` branch of github.com/magenta/midi-ddsp.
#
# Usage: tools/download_weights.sh [target_dir]   (default: ./weights)
set -euo pipefail

target="${1:-weights}"
name="midi_ddsp_model_weights_urmp_9_10"
url="https://github.com/magenta/midi-ddsp/raw/models/${name}.zip"

if [ -d "${target}/${name}" ]; then
  echo "Weights already present in ${target}/${name}"
  exit 0
fi

mkdir -p "${target}"
echo "Downloading ${url}"
curl -fL -o "${target}/${name}.zip" "${url}"
unzip -q "${target}/${name}.zip" -d "${target}"
rm "${target}/${name}.zip"
echo "Weights extracted to ${target}/${name}"
