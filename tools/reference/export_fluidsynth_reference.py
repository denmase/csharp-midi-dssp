"""Record pretty_midi's FluidSynth rendering, used by MIDI-DDSP's fallback.

Writes a MIDI file with parts MIDI-DDSP cannot play (guitar, piano with
pitch bends and control changes, drums), then records how pretty_midi reads
those events and what Instrument.fluidsynth() renders at 16 kHz with the
given soundfont.

Needs libfluidsynth 2.x and pyfluidsynth (1.3.0 was used).

Usage:
  python export_fluidsynth_reference.py <soundfont.sf2> <out.mid> <out.npz>
"""

import sys

import mido
import numpy as np
import pretty_midi

SAMPLE_RATE = 16000


def write_midi(path):
  tpb = 480
  midi = mido.MidiFile(type=1, ticks_per_beat=tpb)
  conductor = mido.MidiTrack()
  conductor.append(mido.MetaMessage('set_tempo', tempo=600000, time=0))
  midi.tracks.append(conductor)

  def track(messages):
    """messages: (tick, mido.Message) in any order."""
    t = mido.MidiTrack()
    now = 0
    for tick, message in sorted(messages, key=lambda m: m[0]):
      t.append(message.copy(time=tick - now))
      now = tick
    midi.tracks.append(t)

  # Nylon guitar with a volume CC before its first note (a pretty_midi
  # "straggler"), a chord, and a repeated note whose note-off and next
  # note-on fall on the same tick.
  guitar = [(0, mido.Message('control_change', channel=0, control=7,
                             value=100)),
            (0, mido.Message('program_change', channel=0, program=24))]
  for start, pitch, vel in [(0, 52, 80), (0, 56, 70), (0, 59, 90),
                            (480, 64, 100), (920, 64, 75), (1440, 62, 60)]:
    guitar += [(start, mido.Message('note_on', channel=0, note=pitch,
                                    velocity=vel)),
               (start + 440, mido.Message('note_off', channel=0,
                                          note=pitch))]
  track(guitar)

  # Piano with pitch bends and a sustain pedal.
  piano = [(0, mido.Message('program_change', channel=1, program=0))]
  for i, pitch in enumerate([60, 62, 64, 65]):
    piano += [(i * 360, mido.Message('note_on', channel=1, note=pitch,
                                     velocity=85)),
              (i * 360 + 300, mido.Message('note_on', channel=1, note=pitch,
                                           velocity=0))]
  piano += [(100, mido.Message('pitchwheel', channel=1, pitch=2048)),
            (400, mido.Message('pitchwheel', channel=1, pitch=-4096)),
            (700, mido.Message('pitchwheel', channel=1, pitch=0)),
            (720, mido.Message('control_change', channel=1, control=64,
                               value=127)),
            (1300, mido.Message('control_change', channel=1, control=64,
                                value=0))]
  track(piano)

  # Drums on channel 10.
  drums = []
  for i in range(6):
    note = 36 if i % 2 == 0 else 38
    drums += [(i * 240, mido.Message('note_on', channel=9, note=note,
                                     velocity=100)),
              (i * 240 + 60, mido.Message('note_off', channel=9, note=note))]
  track(drums)
  midi.save(path)


def main():
  if len(sys.argv) != 4:
    sys.exit(__doc__)
  sf2_path, midi_path, output_path = sys.argv[1:]
  write_midi(midi_path)

  pm = pretty_midi.PrettyMIDI(midi_path)
  arrays = {'instrument_count': np.array([len(pm.instruments)], dtype=np.int64)}
  for i, inst in enumerate(pm.instruments):
    arrays[f'i{i}_program'] = np.array([inst.program], dtype=np.int64)
    arrays[f'i{i}_is_drum'] = np.array([int(inst.is_drum)], dtype=np.int64)
    arrays[f'i{i}_velocity'] = np.array([n.velocity for n in inst.notes],
                                        dtype=np.int64)
    arrays[f'i{i}_bend_time'] = np.array([b.time for b in inst.pitch_bends],
                                         dtype=np.float64)
    arrays[f'i{i}_bend_pitch'] = np.array([b.pitch for b in inst.pitch_bends],
                                          dtype=np.int64)
    arrays[f'i{i}_cc_time'] = np.array(
      [c.time for c in inst.control_changes], dtype=np.float64)
    arrays[f'i{i}_cc_number'] = np.array(
      [c.number for c in inst.control_changes], dtype=np.int64)
    arrays[f'i{i}_cc_value'] = np.array(
      [c.value for c in inst.control_changes], dtype=np.int64)
    arrays[f'i{i}_audio'] = inst.fluidsynth(
      SAMPLE_RATE, synthesizer=sf2_path).astype(np.float32)

  np.savez_compressed(output_path, **arrays)
  for name, value in arrays.items():
    print(f'{name}: {value.dtype} {list(value.shape)}')


if __name__ == '__main__':
  main()
