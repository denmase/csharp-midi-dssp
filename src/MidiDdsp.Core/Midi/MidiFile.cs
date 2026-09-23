using System.Buffers.Binary;
using MidiDdsp.Core.Models;

namespace MidiDdsp.Core.Midi;

/// <summary>An instrument part, split by (program, channel, track) as <c>pretty_midi</c> does.</summary>
public sealed class MidiInstrument(int program, int channel, int track)
{
    /// <summary>0-based General MIDI program.</summary>
    public int Program { get; } = program;
    public int Channel { get; } = channel;
    public int Track { get; } = track;
    public bool IsDrum => Channel == 9;

    /// <summary>Notes in the order <c>pretty_midi</c> lists them: by when they end.</summary>
    public List<MidiNote> Notes { get; } = [];

    /// <summary>Pitch bends, −8192 to 8191, with times in seconds.</summary>
    public List<(double Time, int Pitch)> PitchBends { get; internal set; } = [];

    public List<(double Time, int Number, int Value)> ControlChanges { get; internal set; } = [];
}

/// <summary>
/// A Standard MIDI File read into instrument parts with the same semantics as
/// <c>pretty_midi.PrettyMIDI</c> (as used by MIDI-DDSP): tempo changes are
/// taken from the first track only, a note-off closes every open note of that
/// pitch and channel that did not start on the same tick, notes are listed in
/// the order they end, and parts are ordered by their first completed note.
/// Pitch bends and control changes before a part's first note are kept as
/// <c>pretty_midi</c> keeps them (its "straggler" instruments).
/// </summary>
public sealed class MidiFile
{
    private MidiFile(int ticksPerBeat, List<MidiInstrument> instruments)
    {
        TicksPerBeat = ticksPerBeat;
        Instruments = instruments;
    }

    public int TicksPerBeat { get; }
    public IReadOnlyList<MidiInstrument> Instruments { get; }

    public static MidiFile Load(string path) => Parse(File.ReadAllBytes(path));

    public static MidiFile Parse(byte[] data)
    {
        var tracks = new List<List<MidiEvent>>();
        int pos = 0;
        int ticksPerBeat = 0;
        bool sawHeader = false;
        while (pos + 8 <= data.Length)
        {
            var id = System.Text.Encoding.ASCII.GetString(data, pos, 4);
            int length = checked((int)BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(pos + 4)));
            int body = pos + 8;
            if (body + length > data.Length)
                throw new InvalidDataException($"MIDI chunk '{id}' runs past the end of the file.");
            if (id == "MThd")
            {
                short division = BinaryPrimitives.ReadInt16BigEndian(data.AsSpan(body + 4));
                if (division <= 0)
                    throw new NotSupportedException("SMPTE time division is not supported.");
                ticksPerBeat = division;
                sawHeader = true;
            }
            else if (id == "MTrk")
            {
                tracks.Add(ReadTrack(data.AsSpan(body, length)));
            }
            pos = body + length;
        }
        if (!sawHeader)
            throw new InvalidDataException("Not a MIDI file (no MThd header).");

        var clock = TempoMap.FromFirstTrack(tracks.Count > 0 ? tracks[0] : [], ticksPerBeat);
        return new MidiFile(ticksPerBeat, ReadInstruments(tracks, clock));
    }

    private static List<MidiInstrument> ReadInstruments(List<List<MidiEvent>> tracks, TempoMap clock)
    {
        var instruments = new Dictionary<(int Program, int Channel, int Track), MidiInstrument>();
        var stragglers = new Dictionary<(int Channel, int Track), MidiInstrument>();
        var order = new List<MidiInstrument>();

        // pretty_midi's __get_instrument: notes create instruments; pitch bends and
        // control changes go to an existing one, or to a per-(channel, track)
        // straggler whose event lists a later instrument on that channel shares.
        MidiInstrument GetInstrument(int program, int channel, int track, bool createNew)
        {
            if (instruments.TryGetValue((program, channel, track), out var existing))
                return existing;
            stragglers.TryGetValue((channel, track), out var straggler);
            if (!createNew && straggler is not null)
                return straggler;

            var instrument = new MidiInstrument(program, channel, track);
            if (createNew)
            {
                if (straggler is not null)
                {
                    instrument.ControlChanges = straggler.ControlChanges;
                    instrument.PitchBends = straggler.PitchBends;
                }
                instruments[(program, channel, track)] = instrument;
                order.Add(instrument);
            }
            else
            {
                stragglers[(channel, track)] = instrument;
            }
            return instrument;
        }

        for (int trackIndex = 0; trackIndex < tracks.Count; trackIndex++)
        {
            var openNotes = new Dictionary<(int Channel, int Pitch), List<(long Tick, int Velocity)>>();
            var programs = new int[16];
            foreach (var e in tracks[trackIndex])
            {
                switch (e.Kind)
                {
                    case EventKind.ProgramChange:
                        programs[e.Channel] = e.Data1;
                        break;
                    case EventKind.NoteOn when e.Data2 > 0:
                        var key = (e.Channel, e.Data1);
                        if (!openNotes.TryGetValue(key, out var starts))
                            openNotes[key] = starts = [];
                        starts.Add((e.Tick, e.Data2));
                        break;
                    case EventKind.NoteOn:
                    case EventKind.NoteOff:
                        var offKey = (e.Channel, e.Data1);
                        if (!openNotes.TryGetValue(offKey, out var open))
                            break;
                        var toClose = open.Where(n => n.Tick != e.Tick).ToList();
                        var toKeep = open.Where(n => n.Tick == e.Tick).ToList();
                        foreach (var (start, velocity) in toClose)
                        {
                            GetInstrument(programs[e.Channel], e.Channel, trackIndex, createNew: true).Notes.Add(
                                new MidiNote(clock.TickToTime(start), clock.TickToTime(e.Tick), e.Data1, velocity));
                        }
                        if (toClose.Count > 0 && toKeep.Count > 0)
                            openNotes[offKey] = toKeep;
                        else
                            openNotes.Remove(offKey);
                        break;
                    case EventKind.PitchBend:
                        GetInstrument(programs[e.Channel], e.Channel, trackIndex, createNew: false).PitchBends.Add(
                            (clock.TickToTime(e.Tick), ((e.Data2 << 7) | e.Data1) - 8192));
                        break;
                    case EventKind.ControlChange:
                        GetInstrument(programs[e.Channel], e.Channel, trackIndex, createNew: false).ControlChanges.Add(
                            (clock.TickToTime(e.Tick), e.Data1, e.Data2));
                        break;
                }
            }
        }
        return order;
    }

    private enum EventKind { Other, NoteOff, NoteOn, ControlChange, ProgramChange, PitchBend, SetTempo }

    private readonly record struct MidiEvent(long Tick, EventKind Kind, int Channel, int Data1, int Data2);

    /// <summary>Reads one track's events with absolute ticks, handling running status like mido.</summary>
    private static List<MidiEvent> ReadTrack(ReadOnlySpan<byte> track)
    {
        var events = new List<MidiEvent>();
        int pos = 0;
        long tick = 0;
        int runningStatus = -1;
        while (pos < track.Length)
        {
            tick += ReadVarLen(track, ref pos);
            if (pos >= track.Length)
                break;
            int status = track[pos];
            if (status == 0xFF)
            {
                int type = track[pos + 1];
                pos += 2;
                int length = ReadVarLen(track, ref pos);
                if (type == 0x51 && length == 3)
                {
                    int tempo = (track[pos] << 16) | (track[pos + 1] << 8) | track[pos + 2];
                    events.Add(new MidiEvent(tick, EventKind.SetTempo, 0, tempo, 0));
                }
                pos += length;
                if (type == 0x2F)
                    break;
                continue;
            }
            if (status is 0xF0 or 0xF7)
            {
                pos++;
                pos += ReadVarLen(track, ref pos);
                continue;
            }

            if (status < 0x80)
            {
                if (runningStatus < 0)
                    throw new InvalidDataException("MIDI data byte without a status byte.");
                status = runningStatus;
            }
            else
            {
                pos++;
                runningStatus = status;
            }

            int command = status & 0xF0, channel = status & 0x0F;
            int dataBytes = command is 0xC0 or 0xD0 ? 1 : 2;
            if (pos + dataBytes > track.Length)
                throw new InvalidDataException("MIDI event runs past the end of its track.");
            int data1 = track[pos];
            int data2 = dataBytes == 2 ? track[pos + 1] : 0;
            pos += dataBytes;

            var kind = command switch
            {
                0x80 => EventKind.NoteOff,
                0x90 => EventKind.NoteOn,
                0xB0 => EventKind.ControlChange,
                0xC0 => EventKind.ProgramChange,
                0xE0 => EventKind.PitchBend,
                _ => EventKind.Other,
            };
            if (kind != EventKind.Other)
                events.Add(new MidiEvent(tick, kind, channel, data1, data2));
        }
        return events;
    }

    private static int ReadVarLen(ReadOnlySpan<byte> data, ref int pos)
    {
        int value = 0;
        for (int i = 0; i < 4; i++)
        {
            if (pos >= data.Length)
                throw new InvalidDataException("Truncated MIDI variable-length number.");
            byte b = data[pos++];
            value = (value << 7) | (b & 0x7F);
            if ((b & 0x80) == 0)
                return value;
        }
        throw new InvalidDataException("MIDI variable-length number is too long.");
    }

    /// <summary><c>pretty_midi</c>'s tick-to-seconds conversion (<c>_load_tempo_changes</c>, <c>_update_tick_to_time</c>).</summary>
    private sealed class TempoMap
    {
        private readonly List<(long Tick, double Scale, double Time)> _segments;

        private TempoMap(List<(long Tick, double Scale, double Time)> segments) => _segments = segments;

        public static TempoMap FromFirstTrack(List<MidiEvent> track, int ticksPerBeat)
        {
            var scales = new List<(long Tick, double Scale)> { (0, 60.0 / (120.0 * ticksPerBeat)) };
            foreach (var e in track.Where(e => e.Kind == EventKind.SetTempo))
            {
                double scale = 60.0 / (6e7 / e.Data1 * ticksPerBeat);
                if (e.Tick == 0)
                    scales = [(0, scale)];
                else if (scale != scales[^1].Scale)
                    scales.Add((e.Tick, scale));
            }

            var segments = new List<(long, double, double)>();
            double time = 0;
            for (int i = 0; i < scales.Count; i++)
            {
                segments.Add((scales[i].Tick, scales[i].Scale, time));
                if (i + 1 < scales.Count)
                    time += scales[i].Scale * (scales[i + 1].Tick - scales[i].Tick);
            }
            return new TempoMap(segments);
        }

        public double TickToTime(long tick)
        {
            var segment = _segments[0];
            foreach (var s in _segments)
                if (s.Tick <= tick)
                    segment = s;
            return segment.Time + segment.Scale * (tick - segment.Tick);
        }
    }
}
