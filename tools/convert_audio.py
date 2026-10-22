#!/usr/bin/env python3
"""
Converts the title's sounds for the original Xbox: the Pentium III cannot
decode XMA (WMA Pro) in real time, so every sound is decoded here (ffmpeg's
xma2 decoder) and stored as 4:1 IMA ADPCM, in blocks of one XMA frame (512
samples per channel) that the runtime's XMA decoder (src/runtime/xma.cpp)
plays on an APU voice of its own in place of the packets the title feeds it.

Usage: python tools/convert_audio.py <package folder> <update folder> <output folder>

Each sound is a RIFF file of 2048-byte XMA packets (a data chunk, after a
format chunk in some of the update's). Sounds\\*.xma (the package's and the update's) go to Sounds.jpa,
music\\*.xma to Music.jpa:

  header  "JPA2", u32 entries, u32 0, u32 data offset   (little-endian)
  entries (entries x 40 bytes): u32 streams, u32 channels per stream (0:
          not decoded, silent), u32 frames per stream, u32 data offset (from
          the data offset above) of stream 0, u32 bytes per stream, u32
          packets, u32 packet table offset, u32 name offset (in the file),
          u32 offset of the packets in the sound's file, u32 0
  names   the title's path of each sound below D:\\, lower case, NUL ended
          ("sounds\\sound_cling3.xma", "update\\sounds\\...")
  packets per entry, packets x 8 bytes: u8 stream, u8 frames, u16 0, u32
          first frame (in its stream)
  data    per entry, per stream, per frame, per channel: s16 predictor, u8
          step index, u8 0, 256 bytes of nibbles (512 samples, low nibble
          first)

The runtime knows a sound by the file the title reads it from, so neither
where it lands in memory nor the title's voice set-up matters.
"""
import multiprocessing
import os
import struct
import subprocess
import sys

PACKET = 2048
FRAME = 512

STEPS = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97,
         107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
         876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428,
         4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350,
         22385, 24623, 27086, 29794, 32767]
INDEX = [-1, -1, -1, -1, 2, 4, 6, 8]


def adpcm_encode(samples, state):
    """IMA ADPCM of one channel's frame from state (predictor, index): the
    frame's bytes and the state after it."""
    out = bytearray(FRAME // 2)
    pred, index = state
    for i, s in enumerate(samples):
        step = STEPS[index]
        diff = s - pred
        code = 0
        if diff < 0:
            code = 8
            diff = -diff
        delta = step >> 3
        if diff >= step:
            code |= 4
            diff -= step
            delta += step
        if diff >= step >> 1:
            code |= 2
            diff -= step >> 1
            delta += step >> 1
        if diff >= step >> 2:
            code |= 1
            delta += step >> 2
        pred = pred - delta if code & 8 else pred + delta
        pred = max(-32768, min(32767, pred))
        index = max(0, min(88, index + INDEX[code & 7]))
        out[i >> 1] |= code << ((i & 1) * 4)
    return bytes(out), (pred, index)


def xma_data(path):
    """The sound's packets, and where they start in its file."""
    d = open(path, "rb").read()
    if d[:4] != b"RIFF" or d[8:12] != b"WAVE":
        raise ValueError("%s: not a RIFF WAVE file" % path)
    at = 12
    while at + 8 <= len(d):
        tag, size = d[at:at + 4], struct.unpack("<I", d[at + 4:at + 8])[0]
        if tag == b"data":
            return d[at + 8:at + 8 + size], at + 8
        at += 8 + size + (size & 1)
    raise ValueError("%s: no data chunk" % path)


def streams_of(data):
    """Each packet's stream and the frames that start in it, following the
    packets' skip counts as the hardware does (stream n starts at packet n)."""
    count = len(data) // PACKET
    owner = [None] * count
    stream = 0
    for start in range(count):
        if owner[start] is not None:
            continue
        p = start
        while p < count and owner[p] is None:
            owner[p] = stream
            p += 1 + data[p * PACKET + 3]
        stream += 1
    frames = [data[p * PACKET] >> 2 for p in range(count)]
    return owner, frames, stream


def ffmpeg_decode(data, streams, channels, rate, work):
    """The data as s16 samples, one list per channel, or None if ffmpeg's
    xma2 decoder does not take it with that many channels at that rate."""
    samples = sum(data[p * PACKET] >> 2 for p in range(len(data) // PACKET)) * FRAME // streams
    xma2 = struct.pack("<HIIIIIIIBBH", streams, {1: 0x4, 2: 0x3, 4: 0x33}.get(channels, 0x3), samples, PACKET, 0,
                       samples, 0, 0, 0, 4, 1)
    fmt = struct.pack("<HHIIHHH", 0x0166, channels, rate, rate * channels * 2, channels * 2, 16, len(xma2)) + xma2
    riff = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    src, dst = os.path.join(work, "in.wav"), os.path.join(work, "out.raw")
    open(src, "wb").write(b"RIFF" + struct.pack("<I", len(riff)) + riff)
    if os.path.exists(dst):
        os.remove(dst)
    run = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-f", "s16le", "-acodec", "pcm_s16le", dst],
                         capture_output=True, text=True)
    raw = open(dst, "rb").read() if os.path.exists(dst) else b""
    # A wrong channel count fails at the first packet or so; the right one
    # decodes all but the decoder's start-up delay.
    if run.returncode or "Error" in run.stderr or len(raw) // 2 // channels < samples - 4 * FRAME:
        return None
    pcm = struct.unpack("<%dh" % (len(raw) // 2), raw)
    # ffmpeg drops the start-up delay; the hardware outputs every frame.
    # Padded at the front so each packet's frames line up.
    return [[0] * max(0, samples - len(pcm) // channels) + list(pcm[c::channels])[:samples] for c in range(channels)]


def convert_one(job):
    """One sound: (name, entry fields, packet table bytes, ADPCM blocks). A
    sound that cannot be read is an empty entry (silent), and says why."""
    name, path, work = job
    try:
        return convert_sound(name, path, work)
    except Exception as e:  # a worker must return: the pool waits for it
        print("  %s: %s" % (name, e))
        return name, (0, 0, 0, 0, 0, 0), b"", b""


def convert_sound(name, path, work):
    os.makedirs(work, exist_ok=True)
    data, data_start = xma_data(path)
    owner, frames, streams = streams_of(data)
    # Mono or stereo streams, and the rate: a frame's transform sizes
    # depend on it, the same up to 32 kHz and above (the sound plays at the
    # rate the title gives its XMA context).
    pcm = None
    for channels in (streams, 2 * streams):
        for rate in (48000, 24000):
            pcm = ffmpeg_decode(data, streams, channels, rate, work)
            if pcm:
                break
        if pcm:
            break
    stream_frames = [0] * streams
    table = bytearray()
    for p, f in enumerate(frames):
        table += struct.pack("<BBHI", owner[p], f, 0, stream_frames[owner[p]])
        stream_frames[owner[p]] += f
    blob = bytearray()
    per_stream = len(pcm) // streams if pcm else 0
    if pcm:
        for s in range(streams):
            chans = pcm[s * per_stream:(s + 1) * per_stream]
            states = [(0, 0)] * per_stream
            for f in range(stream_frames[s]):
                for c, channel in enumerate(chans):
                    piece = channel[f * FRAME:(f + 1) * FRAME]
                    piece = piece + [0] * (FRAME - len(piece))
                    pred, index = states[c]
                    nibbles, states[c] = adpcm_encode(piece, states[c])
                    blob += struct.pack("<hBB", pred, index, 0) + nibbles
    return name, (streams, per_stream, max(stream_frames), len(blob) // streams if streams else 0,
                  len(frames), data_start), bytes(table), bytes(blob)


def write_bank(out_path, results):
    count = len(results)
    names = b"".join(name.encode() + b"\0" for name, *_ in results)
    tables = b"".join(table for _, _, table, _ in results)
    header_size = 16 + 40 * count
    names_at = header_size
    tables_at = names_at + len(names)
    data_at = (tables_at + len(tables) + 15) & ~15
    entries, name_at, table_at, data_offset = b"", names_at, tables_at, 0
    for name, (streams, channels, frames, stream_bytes, packets, data_start), table, blob in results:
        entries += struct.pack("<IIIIIIIIII", streams, channels, frames, data_offset, stream_bytes, packets, table_at,
                               name_at, data_start, 0)
        name_at += len(name) + 1
        table_at += len(table)
        data_offset += len(blob)
    with open(out_path, "wb") as f:
        f.write(b"JPA2" + struct.pack("<III", count, 0, data_at) + entries + names + tables)
        f.write(b"\0" * (data_at - f.tell()))
        for *_, blob in results:
            f.write(blob)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    package, update, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    # ffmpeg (a WinGet alias) fails on files in the system temp folder: a
    # work folder of our own, one per worker.
    work = os.path.abspath(os.path.join(out, os.pardir, "audio_work"))
    banks = {"Sounds.jpa": [(package, "sounds"), (update, "sounds")], "Music.jpa": [(package, "music")]}
    with multiprocessing.Pool() as pool:
        for bank, folders in banks.items():
            jobs = []
            for root, folder in folders:
                prefix = "update\\" if root == update else ""
                src = os.path.join(root, folder)
                for name in sorted(os.listdir(src)):
                    if name.lower().endswith(".xma"):
                        key = (prefix + folder + "\\" + name).lower()
                        jobs.append((key, os.path.join(src, name), os.path.join(work, str(len(jobs)))))
            print("convert_audio: %s, %d sounds" % (bank, len(jobs)))
            results = pool.map(convert_one, jobs, chunksize=1)
            silent = [name for name, fields, _, _ in results if not fields[1]]
            for name in silent:
                print("  %s: ffmpeg cannot decode it: silent" % name)
            write_bank(os.path.join(out, bank), results)
            print("  %s: %d KB of ADPCM, %d silent" % (bank, sum(len(r[3]) for r in results) >> 10, len(silent)))


if __name__ == "__main__":
    main()
