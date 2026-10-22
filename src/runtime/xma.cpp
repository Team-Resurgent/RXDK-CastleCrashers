/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Copyright 2025 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// The XMA decoder. The Pentium III cannot decode XMA in real time, so the
// title's sounds are decoded when the title is built
// (tools/convert_audio.py: D:\audio\Sounds.jpa and Music.jpa, IMA ADPCM in
// blocks of one XMA frame). A context here consumes its input packets as the
// hardware does (frames per packet, skip counts, loops) and outputs silence
// for XAudio, so its voices last as long as on the 360. Each packet is found
// in the sound the title read it from (sound_read), and the context's voice
// on the APU plays that sound from there, read ahead of where it plays.
//
// Ported from Xenia (src/xenia/apu/xma_decoder.cc for the registers,
// src/xenia/apu/xma_context.h for the context layout and
// src/xenia/apu/xma_context_fake.cc for the decoder).
//
// The decoder's registers (0x7FEA0000-0x7FEAFFFF) are plain memory here, so
// writes to them are invisible: a runtime thread polls the context-group
// registers every millisecond, acts on their set bits as Xenia's
// XmaDecoder::WriteRegister does, and works the kicked contexts as Xenia's
// worker thread does.
#include "kernel.h"

#include "ppc_recomp_shared.h"

#include <dsound.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace audio {
LPDIRECTSOUND device();  // audio.cpp
}  // namespace audio

namespace {

// Registers (dword index); register r is at kRegisters + r * 4. They are
// little-endian hardware registers the title reads and writes with lwbrx and
// stwbrx, which see the host's native dword: they are accessed natively here,
// not with guest::load32/store32.
constexpr uint32_t kRegisters = 0x7FEA0000;
enum : uint32_t {
    kContextArrayAddress = 0x600,
    kCurrentContextIndex = 0x606,
    kNextContextIndex = 0x607,
    // Groups of 10: register n of a group is a mask of contexts n*32..n*32+31.
    kKick = 0x650,
    kLock = 0x690,
    kClear = 0x6A0,
    kGroupRegisters = 10,
};

volatile LONG* reg(uint32_t r) { return reinterpret_cast<volatile LONG*>(guest::host(kRegisters + r * 4)); }

constexpr uint32_t kContexts = 320, kContextSize = 64;

constexpr uint32_t kBytesPerPacket = 2048;
constexpr uint32_t kBitsPerPacket = kBytesPerPacket * 8;
constexpr uint32_t kBitsPerPacketHeader = 32;
constexpr uint32_t kBytesPerFrameChannel = 512 * 2;  // 512 16-bit samples
constexpr uint32_t kOutputBytesPerBlock = 256;

// A context as the hardware keeps it: 16 big-endian dwords in guest memory,
// loaded and swapped whole so the bitfields can be used (xma_context.h).
struct ContextData {
    // DWORD 0
    uint32_t input_buffer_0_packet_count : 12;
    uint32_t loop_count : 8;
    uint32_t input_buffer_0_valid : 1;
    uint32_t input_buffer_1_valid : 1;
    uint32_t output_buffer_block_count : 5;  // 256-byte blocks
    uint32_t output_buffer_write_offset : 5;
    // DWORD 1
    uint32_t input_buffer_1_packet_count : 12;
    uint32_t loop_subframe_start : 2;
    uint32_t loop_subframe_end : 3;
    uint32_t loop_subframe_skip : 3;
    uint32_t subframe_decode_count : 4;
    uint32_t output_buffer_padding : 3;
    uint32_t sample_rate : 2;
    uint32_t is_stereo : 1;
    uint32_t unk_dword_1_c : 1;
    uint32_t output_buffer_valid : 1;
    // DWORD 2
    uint32_t input_buffer_read_offset : 26;  // bits
    uint32_t error_status : 5;
    uint32_t error_set : 1;
    // DWORD 3
    uint32_t loop_start : 26;
    uint32_t parser_error_status : 5;
    uint32_t parser_error_set : 1;
    // DWORD 4
    uint32_t loop_end : 26;
    uint32_t packet_metadata : 5;
    uint32_t current_buffer : 1;
    // DWORD 5-8: physical addresses
    uint32_t input_buffer_0_ptr;
    uint32_t input_buffer_1_ptr;
    uint32_t output_buffer_ptr;
    uint32_t work_buffer_ptr;
    // DWORD 9
    uint32_t output_buffer_read_offset : 5;
    uint32_t : 25;
    uint32_t stop_when_done : 1;
    uint32_t interrupt_when_done : 1;
    // DWORD 10-15
    uint32_t unk_dwords_10_15[6];

    void load(uint32_t address) {
        uint32_t* d = reinterpret_cast<uint32_t*>(this);
        for (uint32_t i = 0; i < kContextSize / 4; i++)
            d[i] = guest::load32(address + i * 4);
    }
    void store(uint32_t address) const {
        const uint32_t* d = reinterpret_cast<const uint32_t*>(this);
        for (uint32_t i = 0; i < kContextSize / 4; i++)
            guest::store32(address + i * 4, d[i]);
    }

    bool any_input_buffer_valid() const { return input_buffer_0_valid || input_buffer_1_valid; }
    uint32_t current_input_buffer_packet_count() const {
        return current_buffer == 0 ? input_buffer_0_packet_count : input_buffer_1_packet_count;
    }
};
static_assert(sizeof(ContextData) == kContextSize, "the hardware's context is 64 bytes");

// The output ring buffer (Xenia's RingBuffer, with only what the fake decoder
// uses): read == write means empty, and all of it is free to write.
struct Ring {
    uint8_t* buffer;
    uint32_t capacity, read, write;
    uint32_t write_count() const {
        if (read == write)
            return capacity;
        return write < read ? read - write : capacity - write + read;
    }
    void write_data(const uint8_t* data, uint32_t count) {
        if (count > capacity)
            count = capacity;
        if (write + count < capacity) {
            memcpy(buffer + write, data, count);
            write += count;
        } else {
            uint32_t left = capacity - write;
            memcpy(buffer + write, data, left);
            memcpy(buffer, data + left, count - left);
            write = count - left;
        }
    }
};

// --- the converted sounds -------------------------------------------------

constexpr uint32_t kFrameSamples = 512;
constexpr uint32_t kBlockBytes = 4 + kFrameSamples / 2;  // one channel's frame: predictor, index, nibbles

struct BankEntry {
    uint32_t streams, channels, frames, offset, stream_bytes, packets, table, name, data_start, reserved;
};
struct PacketInfo {
    uint8_t stream, frames;
    uint16_t reserved;
    uint32_t first_frame;
};
static_assert(sizeof(BankEntry) == 40 && sizeof(PacketInfo) == 8, "convert_audio.py's layout");

// A converted bank (tools/convert_audio.py): its entries by the title's path
// of each sound, the blocks read from the file as they are needed.
struct Bank {
    std::vector<BankEntry> entries;
    std::map<std::string, uint32_t> by_name;
    uint32_t data_offset = 0;
    HANDLE file = INVALID_HANDLE_VALUE;
    bool keep = false;  // effects: a sound's blocks are read whole when the title loads it
    CRITICAL_SECTION file_lock;  // the decoder thread and the title's loads read it
};
Bank banks[2];  // effects, music

DWORD longest_read;  // statistics: the longest read_blocks, in ms

// Reads [at, at + size) of the bank's blocks.
void read_blocks(Bank& bank, uint32_t at, uint8_t* out, uint32_t size) {
    DWORD start = GetTickCount();
    EnterCriticalSection(&bank.file_lock);
    DWORD read = 0;
    SetFilePointer(bank.file, LONG(bank.data_offset + at), nullptr, FILE_BEGIN);
    if (!ReadFile(bank.file, out, size, &read, nullptr) || read != size)
        memset(out + read, 0, size - read);
    LeaveCriticalSection(&bank.file_lock);
    DWORD took = GetTickCount() - start;
    longest_read = took > longest_read ? took : longest_read;
}

// A sound the title has read into memory: from start, its packets from
// first_packet on, to end. Its blocks are kept here (effects, within
// kKeptBytes) or read as its packets are decoded.
struct Loaded {
    uint32_t end, first_packet;
    Bank* bank;
    const BankEntry* entry;
    std::vector<PacketInfo> packets;  // the entry's
    std::vector<uint8_t> blocks;      // all the entry's, when kept
    uint32_t sequence;                // when it was loaded (the oldest are dropped first)
};
std::map<uint32_t, Loaded> loaded;  // by guest address
uint32_t loaded_sequence, kept_bytes;
constexpr uint32_t kKeptBytes = 1 << 20;
CRITICAL_SECTION loaded_lock;
struct LoadedLockInit {
    LoadedLockInit() { InitializeCriticalSection(&loaded_lock); }
} loaded_lock_init;

// The headers (first 4 bytes) of packets whose memory was given back
// (runtime/io.cpp releases the sounds' XMA data as it is read): what a
// context needs of a packet to advance as the hardware would.
std::map<uint32_t, uint32_t> saved_headers;  // packet's guest address -> its header, as in memory
CRITICAL_SECTION saved_headers_lock;
struct SavedHeadersInit {
    SavedHeadersInit() { InitializeCriticalSection(&saved_headers_lock); }
} saved_headers_init;

bool load_bank(Bank& bank, const char* path, bool keep) {
    InitializeCriticalSection(&bank.file_lock);
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        log("xma: no %s: those sounds are silent\n", path);
        return false;
    }
    uint32_t header[4];
    DWORD read = 0;
    if (!ReadFile(f, header, sizeof(header), &read, nullptr) || read != sizeof(header) || header[0] != 0x3241504A) {
        log("xma: %s is not a converted bank\n", path);
        CloseHandle(f);
        return false;
    }
    bank.entries.resize(header[1]);
    bank.data_offset = header[3];
    ReadFile(f, bank.entries.data(), header[1] * sizeof(BankEntry), &read, nullptr);
    // The names: from the first entry's to the first packet table.
    uint32_t names = bank.entries.empty() ? 0 : bank.entries[0].name;
    uint32_t names_end = bank.entries.empty() ? 0 : bank.entries[0].table;
    std::vector<char> text(names_end - names);
    ReadFile(f, text.data(), DWORD(text.size()), &read, nullptr);
    for (uint32_t i = 0; i < bank.entries.size(); i++)
        bank.by_name[&text[bank.entries[i].name - names]] = i;
    bank.file = f;
    bank.keep = keep;
    log("xma: %s: %u sounds\n", path, header[1]);
    return true;
}

const int16_t kSteps[89] = {7,     8,     9,     10,    11,    12,    13,    14,    16,    17,    19,    21,    23,
                            25,    28,    31,    34,    37,    41,    45,    50,    55,    60,    66,    73,    80,
                            88,    97,    107,   118,   130,   143,   157,   173,   190,   209,   230,   253,   279,
                            307,   337,   371,   408,   449,   494,   544,   598,   658,   724,   796,   876,   963,
                            1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,  2272,  2499,  2749,  3024,  3327,
                            3660,  4026,  4428,  4871,  5358,  5894,  6484,  7132,  7845,  8630,  9493,  10442, 11487,
                            12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767};
const int8_t kIndexStep[8] = {-1, -1, -1, -1, 2, 4, 6, 8};

// One channel's frame (a 260-byte block) into 512 samples.
void decode_block(const uint8_t* block, int16_t* out) {
    int pred = int16_t(block[0] | (block[1] << 8)), index = block[2];
    for (uint32_t i = 0; i < kFrameSamples; i++) {
        int code = (block[4 + (i >> 1)] >> ((i & 1) * 4)) & 15;
        int step = kSteps[index], delta = step >> 3;
        if (code & 4)
            delta += step;
        if (code & 2)
            delta += step >> 1;
        if (code & 1)
            delta += step >> 2;
        pred = code & 8 ? pred - delta : pred + delta;
        pred = pred < -32768 ? -32768 : pred > 32767 ? 32767 : pred;
        index += kIndexStep[code & 7];
        index = index < 0 ? 0 : index > 88 ? 88 : index;
        out[i] = int16_t(pred);
    }
}

// A context's voice on the APU: a looping DirectSound buffer the context's
// sound is written into ahead of where it plays, straight from the
// converted sound (its Feed), so it plays smoothly at the rate and volume
// XAudio gives the context's voice (xaudio.cpp) whether or not XAudio keeps
// up with real time. XAudio takes the context's output (silence) as it
// would the sound, which keeps its voices' timing for the title.
constexpr uint32_t kVoiceFrames = 32;
// Frames kept written ahead of the play position. A voice starts once that
// many are written (or all of a shorter sound), and a frame half a buffer
// ahead of the writes is silenced, so a voice that runs out is quiet.
constexpr uint32_t kLeadFrames = 12;

// What a voice plays: a converted sound's stream from frame next on, looping
// from loop_end back to loop_start when the context loops (loop_end 0: it
// does not).
struct Feed {
    Bank* bank = nullptr;
    const BankEntry* entry = nullptr;  // null: none
    uint32_t key = 0;                  // the loaded sound's guest address (for its kept blocks)
    uint32_t stream = 0, next = 0, loop_start = 0, loop_end = 0, wraps = 0;
    uint32_t loops = 255;  // passes of the loop left (255: for ever); the last plays on to the end
    uint32_t out_channels = 0, rate = 0;
    // Blocks read ahead from the bank: frames cache_first on.
    std::vector<uint8_t> cache;
    uint32_t cache_first = 0, cache_count = 0;
};

struct Voice {
    LPDIRECTSOUNDBUFFER buffer = nullptr;
    uint32_t channels = 0, rate = 0, frame_bytes = 0, write = 0;
    uint32_t primed = 0;  // frames written before it plays
    bool playing = false;
    // When it last had a frame written, and when the title last took a
    // frame of its context.
    DWORD last_write = 0, last_taken = 0;
    DWORD checked = 0;  // when its play position was last read
    Feed feed;
    // From XAudio, and what was last applied.
    volatile float ratio = 0, gain = 1;
    float applied_ratio = -1, applied_gain = -1;
    // The pan (the front speakers' gains), and what was last applied.
    float left = 1, right = 1, applied_left = -1, applied_right = -1;
};

struct Context {
    bool allocated = false;
    bool enabled = false;  // kicked and not yet worked
    uint8_t frame_remaining_subframes = 0;
    Voice voice;
    uint32_t output_start = 0, output_end = 0;  // guest addresses of the output buffer
    // The packet being output: the frames of it still to output. And the
    // first frame of the last packet found in a converted sound, with the
    // feed's loops seen then (to tell its loops from the title's seeks).
    uint32_t frames_left = 0;
    uint32_t taken_frame = 0, wraps_seen = 0;
    // The frame being copied out (silence), as the hardware writes it.
    uint8_t output[kFrameSamples * 2 * 2];
    uint32_t output_size = 0, output_done = 0;
    uint32_t pending_skip = 0;  // packets still to skip in the next buffer
};

uint32_t voices_started, voice_matches, voice_misses;  // statistics
// Statistics: voices found run out, and the fewest frames a voice's writes
// were ahead of its play position.
uint32_t voices_dry, least_lead = ~0u;

void stop_voice(Voice& v) {
    if (v.buffer && v.playing)
        v.buffer->Stop();
    v.playing = false;
    v.primed = 0;
}

void end_feed(Voice& v) {
    stop_voice(v);
    v.feed.entry = nullptr;
    std::vector<uint8_t>().swap(v.feed.cache);
    v.feed.cache_count = 0;
}

void start_voice(Voice& v) {
    v.buffer->SetCurrentPosition(0);
    v.buffer->Play(0, 0, DSBPLAY_LOOPING);
    v.playing = true;
    voices_started++;
}

// A decoded frame (512 samples x channels, little-endian) onto the voice.
void voice_write(Voice& v, const int16_t* samples, uint32_t channels, uint32_t rate) {
    uint32_t frame_bytes = kFrameSamples * channels * 2;
    if (!v.buffer || v.channels != channels || v.rate != rate) {
        if (v.buffer) {
            stop_voice(v);
            v.buffer->Release();
            v.buffer = nullptr;
        }
        WAVEFORMATEX format = {};
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nChannels = WORD(channels);
        format.nSamplesPerSec = rate;
        format.wBitsPerSample = 16;
        format.nBlockAlign = WORD(channels * 2);
        format.nAvgBytesPerSec = rate * channels * 2;
        DSBUFFERDESC desc = {};
        desc.dwSize = sizeof(desc);
        desc.dwBufferBytes = kVoiceFrames * frame_bytes;
        desc.lpwfxFormat = &format;
        LPDIRECTSOUND sound = audio::device();
        if (!sound || FAILED(sound->CreateSoundBuffer(&desc, &v.buffer, nullptr))) {
            v.buffer = nullptr;
            return;
        }
        v.channels = channels;
        v.rate = rate;
        v.frame_bytes = frame_bytes;
        v.applied_ratio = v.applied_gain = v.applied_left = v.applied_right = -1;
    }
    uint32_t size = kVoiceFrames * frame_bytes;
    auto put = [&](uint32_t at, const void* data) {
        void *p1, *p2;
        DWORD n1, n2;
        if (SUCCEEDED(v.buffer->Lock(at, frame_bytes, &p1, &n1, &p2, &n2, 0))) {
            if (data) {
                __builtin_memcpy(p1, data, n1);
                if (p2)
                    __builtin_memcpy(p2, static_cast<const uint8_t*>(data) + n1, n2);
            } else {
                __builtin_memset(p1, 0, n1);
                if (p2)
                    __builtin_memset(p2, 0, n2);
            }
            v.buffer->Unlock(p1, n1, p2, n2);
        }
    };
    if (!v.playing && !v.primed) {
        // From the start of the buffer, the rest silent.
        for (uint32_t f = 1; f < kVoiceFrames; f++)
            put(f * frame_bytes, nullptr);
        v.write = 0;
    }
    put(v.write, samples);
    v.write = (v.write + frame_bytes) % size;
    put((v.write + size / 2) % size, nullptr);
    v.last_write = GetTickCount();
    if (!v.playing && ++v.primed >= kLeadFrames)
        start_voice(v);
}

// Frames the voice's writes are ahead of where it plays. One that ran out
// (the play position passed the writes) is written to just ahead of it.
uint32_t voice_lead(Voice& v) {
    uint32_t size = kVoiceFrames * v.frame_bytes;
    DWORD play = 0, unused = 0;
    v.buffer->GetCurrentPosition(&play, &unused);
    uint32_t lead = (v.write + size - play) % size;
    if (lead > size / 2) {
        v.write = (play / v.frame_bytes + 1) % kVoiceFrames * v.frame_bytes;
        voices_dry++;
        return 1;
    }
    lead /= v.frame_bytes;
    least_lead = lead < least_lead ? lead : least_lead;
    return lead;
}

// The feed's frame next onto its voice, read from its sound.
void feed_frame(Voice& v) {
    Feed& f = v.feed;
    const BankEntry& e = *f.entry;
    uint32_t block_bytes = e.channels * kBlockBytes;
    if (f.next < f.cache_first || f.next >= f.cache_first + f.cache_count) {
        // Sixteen frames on, from the kept blocks or the disc.
        uint32_t count = e.frames - f.next < 16 ? e.frames - f.next : 16;
        uint32_t at = e.offset + f.stream * e.stream_bytes + f.next * block_bytes;
        f.cache.resize(count * block_bytes);
        bool kept = false;
        EnterCriticalSection(&loaded_lock);
        auto it = loaded.find(f.key);
        if (it != loaded.end() && it->second.entry == f.entry && !it->second.blocks.empty()) {
            memcpy(f.cache.data(), &it->second.blocks[at - e.offset], f.cache.size());
            kept = true;
        }
        LeaveCriticalSection(&loaded_lock);
        if (!kept)
            read_blocks(*f.bank, at, f.cache.data(), uint32_t(f.cache.size()));
        f.cache_first = f.next;
        f.cache_count = count;
    }
    static int16_t pcm[2][kFrameSamples], native[kFrameSamples * 2];
    const uint8_t* block = &f.cache[(f.next - f.cache_first) * block_bytes];
    uint32_t channels = e.channels < 2 ? e.channels : 2;
    for (uint32_t ch = 0; ch < channels; ch++)
        decode_block(block + ch * kBlockBytes, pcm[ch]);
    for (uint32_t i = 0; i < kFrameSamples; i++) {
        int left = pcm[0][i], right = channels > 1 ? pcm[1][i] : left;
        if (f.out_channels == 1) {
            native[i] = int16_t(channels > 1 ? (left + right) / 2 : left);
        } else {
            native[i * 2] = int16_t(left);
            native[i * 2 + 1] = int16_t(right);
        }
    }
    voice_write(v, native, f.out_channels, f.rate);
    if (++f.next >= f.loop_end && f.loop_end) {
        f.next = f.loop_start;
        f.wraps++;
        if (f.loops != 255 && --f.loops == 0)
            f.loop_end = 0;
        static int said;
        if (f.bank == &banks[1] && said++ < 30)
            log("xma: music loops back to frame %u\n", f.loop_start);
    }
}

// A decoder pass's writes to the voice: kLeadFrames ahead of where it plays.
// It stops when the title stops taking its context's output (the sound was
// stopped or paused) or when its sound has played out.
void feed_voice(Voice& v) {
    Feed& f = v.feed;
    if (!f.entry)
        return;
    DWORD now = GetTickCount();
    if (now - v.last_taken > 750) {
        end_feed(v);
        return;
    }
    auto more = [&] { return f.next < f.entry->frames; };
    if (!v.playing) {
        while (!v.playing && more())
            feed_frame(v);
        if (!v.playing && v.primed && v.buffer)
            start_voice(v);  // a sound shorter than the lead: all written
        return;
    }
    if (!more()) {
        if (now - v.last_write > 200)
            end_feed(v);  // played out
        return;
    }
    // The play position is costly to read (the APU's): every 40 ms, well
    // within the lead (about 130 ms).
    if (now - v.checked < 40)
        return;
    v.checked = now;
    for (uint32_t lead = voice_lead(v); lead < kLeadFrames && more(); lead++)
        feed_frame(v);
}

// XAudio's pitch and volume onto the APU.
void update_voice(Voice& v) {
    if (!v.playing)
        return;
    float ratio = v.ratio, gain = v.gain;
    if (ratio > 0 && (ratio < v.applied_ratio * 0.998f || ratio > v.applied_ratio * 1.002f)) {
        DWORD hz = DWORD(ratio * 48000.0f);
        v.buffer->SetFrequency(hz < 1000 ? 1000 : hz > 192000 ? 192000 : hz);
        v.applied_ratio = ratio;
    }
    // A gain in hundredths of a decibel.
    auto millibels = [](float gain) {
        LONG mb = gain > 0.0001f ? LONG(2000.0f * std::log10(gain)) : DSBVOLUME_MIN;
        return mb < DSBVOLUME_MIN ? DSBVOLUME_MIN : mb > 0 ? 0 : mb;
    };
    if (gain != v.applied_gain) {
        v.buffer->SetVolume(millibels(gain));
        v.applied_gain = gain;
    }
    float left = v.left, right = v.right;
    if (left != v.applied_left || right != v.applied_right) {
        // A mono voice goes to both front speakers, a stereo one's channels
        // one to each: the pan is their volumes.
        DSMIXBINVOLUMEPAIR pairs[2] = {{DSMIXBIN_FRONT_LEFT, millibels(left)}, {DSMIXBIN_FRONT_RIGHT, millibels(right)}};
        DSMIXBINS bins = {2, pairs};
        v.buffer->SetMixBinVolumes(&bins);
        v.applied_left = left;
        v.applied_right = right;
    }
}

// The current packet's next frame of output: silence, which XAudio takes as
// it would the sound (the voice is heard from the APU).
void decode_frame(Context& c, uint32_t out_channels) {
    c.output_size = kFrameSamples * out_channels * 2;
    memset(c.output, 0, c.output_size);
    c.output_done = 0;
    c.frames_left--;
    c.voice.last_taken = GetTickCount();
}

uint32_t contexts;  // guest address of the array
Context state[kContexts];
// Held by the decoder thread for a whole pass, and by XMACreateContext and
// XMAReleaseContext: what Xenia's per-context locks are for.
CRITICAL_SECTION lock;

// An XAudio source voice (the overrides at the end): it plays the converted
// sound its buffer holds on an APU voice of its own, at XAudio's volume, pan
// and pitch. XAudio never starts it, so XAudio mixes nothing. One whose
// buffer is not a converted sound is left to XAudio (native false).
struct Source {
    Voice voice;
    bool native = false;   // its buffer is a converted sound
    bool started = false;  // between Start and Stop
    uint32_t first_frame = 0, context = 0;  // where its buffer starts; the buffer's pContext
    float volume = 1, ratio = 1;
};
std::map<uint32_t, Source> sources;  // by the voice's guest address, under lock
uint32_t native_buffers, xaudio_buffers;  // statistics

// Statistics for the log: worked counts passes that decoded something.
uint32_t kicks, locks, clears, worked, stray_kicks, unknown_packets;

uint32_t context_address(uint32_t id) { return contexts + id * kContextSize; }

// A physical address (or a virtual one: MmGetPhysicalAddress returns its
// argument) as a guest address.
uint32_t translate_physical(uint32_t p) { return (p & 0x1FFFFFFF) | 0x80000000; }

void swap_input_buffer(ContextData& data) {
    // No more frames.
    if (data.current_buffer == 0)
        data.input_buffer_0_valid = 0;
    else
        data.input_buffer_1_valid = 0;
    data.current_buffer ^= 1;
    data.input_buffer_read_offset = kBitsPerPacketHeader;
}

// A loop's end reached: back to its start (at packet granularity: the packet
// a loop point falls in is decoded whole).
void update_loop_status(ContextData& data) {
    if (data.loop_count == 0)
        return;
    uint32_t loop_end = data.loop_end > kBitsPerPacketHeader ? data.loop_end : kBitsPerPacketHeader;
    if (data.input_buffer_read_offset < loop_end || data.loop_end <= data.loop_start)
        return;
    data.input_buffer_read_offset = data.loop_start / kBitsPerPacket * kBitsPerPacket + kBitsPerPacketHeader;
    if (data.loop_count != 255)
        data.loop_count = data.loop_count - 1;
}

// A packet of a sound the title loaded: the sound, its packet's stream,
// first frame and frames, and the loaded sound's guest address.
struct PacketRef {
    Bank* bank;
    const BankEntry* entry;
    uint32_t key, stream, first_frame, frames;
};

bool find_packet(uint32_t address, PacketRef& out) {
    EnterCriticalSection(&loaded_lock);
    auto it = loaded.upper_bound(address);
    bool found = false;
    if (it != loaded.begin() && address < (--it)->second.end) {
        const Loaded& l = it->second;
        uint32_t index = l.first_packet + (address - it->first) / kBytesPerPacket;
        if (index < l.packets.size()) {
            const PacketInfo& p = l.packets[index];
            out = {l.bank, l.entry, it->first, p.stream, p.first_frame, p.frames};
            found = true;
        }
    }
    LeaveCriticalSection(&loaded_lock);
    return found;
}

// The context's voice plays its sound from the packet p on (the title
// started it, or moved it elsewhere). When the context loops, so does the
// voice: between the frames its loop points' packets start and end at.
void start_feed(Context& c, const ContextData& data, uint32_t buffer, const PacketRef& p) {
    static const uint32_t kRates[4] = {24000, 32000, 44100, 48000};
    Voice& v = c.voice;
    end_feed(v);
    Feed& f = v.feed;
    f.bank = p.bank;
    f.entry = p.entry;
    f.key = p.key;
    f.stream = p.stream;
    f.next = p.first_frame;
    f.loop_start = f.loop_end = 0;
    f.out_channels = data.is_stereo ? 2 : 1;
    f.rate = kRates[data.sample_rate];
    PacketRef start, end;
    if (data.loop_count && data.loop_end > data.loop_start &&
        find_packet(buffer + data.loop_start / kBitsPerPacket * kBytesPerPacket, start) &&
        find_packet(buffer + data.loop_end / kBitsPerPacket * kBytesPerPacket, end) && start.entry == p.entry &&
        end.entry == p.entry) {
        f.loop_start = start.first_frame;
        f.loop_end = end.first_frame + end.frames;
    }
    v.last_taken = GetTickCount();
    c.wraps_seen = f.wraps;
    // Diagnostic: the music's voices, whether they loop on their own.
    static int said;
    if (p.bank == &banks[1] && said++ < 60) {
        const char* name = "?";
        for (const auto& [n, i] : p.bank->by_name)
            if (&p.bank->entries[i] == p.entry)
                name = n.c_str();
        log("xma: music %s from frame %u of %u; loop count %u, bits %u-%u, frames %u-%u\n", name, p.first_frame,
            p.entry->frames, data.loop_count, data.loop_start, data.loop_end, f.loop_start, f.loop_end);
    }
}

// Starts the next frame: of the current packet while it has frames, else of
// the next packet, which is looked up in the converted banks. Packets are
// stepped as the hardware steps them: past the skip count's packets of the
// other streams, into the other input buffer when this one runs out.
void process_packet(Context& c, ContextData& data) {
    if (c.output_done < c.output_size)
        return;  // the current frame is not all out yet
    uint32_t out_channels = data.is_stereo ? 2 : 1;
    if (c.frames_left) {
        decode_frame(c, out_channels);
        return;
    }
    // No available data.
    if (!data.any_input_buffer_valid())
        return;
    update_loop_status(data);
    uint32_t packets = data.current_input_buffer_packet_count();
    if (c.pending_skip) {
        // The skip left over from the other buffer.
        uint32_t skip = c.pending_skip < packets ? c.pending_skip : packets;
        c.pending_skip -= skip;
        data.input_buffer_read_offset = skip * kBitsPerPacket + kBitsPerPacketHeader;
    }
    uint32_t packet = data.input_buffer_read_offset / kBitsPerPacket;
    // Every packet of the current buffer done: switch to the other.
    if (packet >= packets) {
        swap_input_buffer(data);
        return;
    }
    uint32_t buffer = data.current_buffer == 0 ? data.input_buffer_0_ptr : data.input_buffer_1_ptr;
    uint32_t address = translate_physical(buffer) + packet * kBytesPerPacket;
    const uint8_t* bytes = guest::host(address);
    // A released packet: its saved header. Released pages read as zeros
    // (and one not touched since is not touched now).
    uint8_t header[4];
    bool saved = false;
    uint32_t zero = 0;
    if (!guest::resident(address) || memcmp(bytes, &zero, 4) == 0) {
        EnterCriticalSection(&saved_headers_lock);
        auto it = saved_headers.find(address);
        if (it != saved_headers.end()) {
            memcpy(header, &it->second, 4);
            saved = true;
        }
        LeaveCriticalSection(&saved_headers_lock);
    }
    uint32_t header_frames = (saved ? header[0] : bytes[0]) >> 2, skip = saved ? header[3] : bytes[3];

    c.frames_left = header_frames;
    PacketRef p;
    if (find_packet(address, p)) {
        c.frames_left = p.frames;
        // The voice is ahead of the title. It starts over where the title is
        // for another sound, when the title is ahead of it (it ran out), or
        // when the title went back other than by one of the voice's loops.
        const Feed& f = c.voice.feed;
        bool looped = p.first_frame < c.taken_frame && f.wraps > c.wraps_seen;
        if (looped)
            c.wraps_seen++;
        uint32_t ahead = f.next + (f.wraps - c.wraps_seen) * (f.loop_end - f.loop_start);
        if (p.entry->channels && (f.entry != p.entry || f.stream != p.stream || p.first_frame > ahead + 2 ||
                                  (p.first_frame < c.taken_frame && !looped)))
            start_feed(c, data, translate_physical(buffer), p);
        c.taken_frame = p.first_frame;
    } else {
        unknown_packets++;
    }

    uint32_t next = packet + 1 + skip;
    if (next >= packets) {
        c.pending_skip = next - packets;
        swap_input_buffer(data);
    } else {
        data.input_buffer_read_offset = next * kBitsPerPacket + kBitsPerPacketHeader;
    }
    if (c.frames_left)
        decode_frame(c, out_channels);
}

// Writes the current frame's subframes (256-byte blocks) to the output, at
// most the context's subframe_decode_count a pass.
void consume(Context& c, Ring& output, ContextData& data, int32_t& remaining_blocks) {
    if (c.output_done >= c.output_size)
        return;
    uint32_t bytes = c.output_size - c.output_done;
    uint32_t most = data.subframe_decode_count * kOutputBytesPerBlock;
    if (bytes > most)
        bytes = most;
    output.write_data(c.output + c.output_done, bytes);
    c.output_done += bytes;
    remaining_blocks -= int32_t(bytes / kOutputBytesPerBlock);
}

// One kick's work (XmaContextFake::Work): decode into the free part of the
// output buffer, as far as the input lasts.
void work(uint32_t id) {
    Context& c = state[id];
    if (!c.enabled || !c.allocated)
        return;
    c.enabled = false;

    uint32_t address = context_address(id);
    ContextData data;
    data.load(address);
    if (!data.output_buffer_valid)
        return;

    Ring output;
    output.buffer = guest::host(translate_physical(data.output_buffer_ptr));
    output.capacity = data.output_buffer_block_count * kOutputBytesPerBlock;
    c.output_start = translate_physical(data.output_buffer_ptr);
    c.output_end = c.output_start + output.capacity;
    if (!output.capacity) {
        data.store(address);
        return;
    }
    output.read = data.output_buffer_read_offset * kOutputBytesPerBlock % output.capacity;
    output.write = data.output_buffer_write_offset * kOutputBytesPerBlock % output.capacity;
    int32_t remaining_blocks = int32_t(output.write_count() / kOutputBytesPerBlock);

    // Not enough space for even one pass.
    int32_t minimum = int32_t(data.subframe_decode_count) * 2 - 1;
    if (minimum > remaining_blocks) {
        data.store(address);
        return;
    }
    while (remaining_blocks >= minimum) {
        // A subframe_decode_count of 0 consumes nothing: stop rather than
        // spin (Xenia would loop forever).
        int32_t blocks_before = remaining_blocks;
        uint32_t offset_before = data.input_buffer_read_offset, buffer_before = data.current_buffer;
        process_packet(c, data);
        consume(c, output, data, remaining_blocks);
        if (!data.any_input_buffer_valid() || data.error_status == 4)
            break;
        if (remaining_blocks == blocks_before && data.input_buffer_read_offset == offset_before &&
            data.current_buffer == buffer_before)
            break;
    }
    if (output.write != data.output_buffer_write_offset * kOutputBytesPerBlock % output.capacity)
        worked++;
    data.output_buffer_write_offset = output.write / kOutputBytesPerBlock;
    // Read == write means nothing written or the buffer full: Xenia marks the
    // output buffer done either way.
    if (output.read == output.write)
        data.output_buffer_valid = 0;
    data.store(address);
}

// A kick. The title kicks a voice's contexts one store each, so of a group
// register only its last kick is seen (take_group): a kick seen in a group
// stands for the group's other contexts that are ready to decode (allocated,
// with a valid output buffer) too. One the title set up but meant to hold
// back is decoded early: being silent, it only finishes sooner.
void enable(uint32_t id) {
    if (!state[id].allocated)
        stray_kicks++;
    state[id].enabled = true;
    uint32_t first = id & ~31u;
    for (uint32_t n = first; n < first + 32 && n < kContexts; n++)
        if (state[n].allocated && guest::load32(context_address(n) + 4) >> 31)
            state[n].enabled = true;
}

void disable(uint32_t id) { state[id].enabled = false; }

// Resets the context's buffers (XmaContextFake::Clear).
void clear(uint32_t id) {
    uint32_t address = context_address(id);
    ContextData data;
    data.load(address);
    data.input_buffer_0_valid = 0;
    data.input_buffer_1_valid = 0;
    data.output_buffer_valid = 0;
    data.output_buffer_read_offset = 0;
    data.output_buffer_write_offset = 0;
    data.input_buffer_read_offset = kBitsPerPacketHeader;
    Context& c = state[id];
    c.frame_remaining_subframes = 0;
    c.frames_left = 0;
    c.output_size = c.output_done = 0;
    c.pending_skip = 0;
    end_feed(c.voice);
    data.store(address);
}

// Takes a group's set bits and clears them, as the hardware does once it has
// acted on them: atomically, so a bit the title sets meanwhile is kept for
// the next pass. The title writes one context's bit per store, so when it
// kicks (or locks) several contexts of a group between two passes only the
// last store is seen: the others' commands are lost.
template <typename Action>
uint32_t take_group(uint32_t first, Action action) {
    uint32_t count = 0;
    for (uint32_t n = 0; n < kGroupRegisters; n++) {
        uint32_t mask = uint32_t(InterlockedExchange(reg(first + n), 0));
        while (mask) {
            uint32_t id = n * 32 + uint32_t(__builtin_ctz(mask));
            if (id < kContexts)
                action(id);
            mask &= mask - 1;
            count++;
        }
    }
    return count;
}

DWORD WINAPI decoder_thread(void*) {
    uint32_t next = 1, passes = 0;
    uint32_t logged_kicks = 0, logged_stray = 0;
    // Statistics: the longest pass and the longest time between two, in ms.
    DWORD last_end = GetTickCount(), longest_pass = 0, longest_gap = 0, last_stats = last_end;
    uint32_t dry_logged = 0;
    for (;;) {
        DWORD pass_start = GetTickCount();
        longest_gap = pass_start - last_end > longest_gap ? pass_start - last_end : longest_gap;
        EnterCriticalSection(&lock);
        // CurrentContextIndex rotates on read on the hardware: the context
        // being processed. The title's lock waits while it names a context
        // it holds, so it must not stick (and never sets 0x200, which makes
        // the title's check collide).
        *reg(kCurrentContextIndex) = LONG(next);
        next = (next + 1) % kContexts;
        *reg(kNextContextIndex) = LONG(next);
        // Within one pass the title can have locked, rewritten and kicked a
        // context (its audio frame does all three): locks, then clears, then
        // kicks, so the kick wins.
        locks += take_group(kLock, disable);
        clears += take_group(kClear, clear);
        kicks += take_group(kKick, enable);
        for (uint32_t n = 0; n < kContexts; n++) {
            work(n);
            feed_voice(state[n].voice);
            update_voice(state[n].voice);
        }
        DWORD now = GetTickCount();
        for (auto& [address, s] : sources) {
            if (!s.native || !s.started)
                continue;
            s.voice.last_taken = now;  // no context to take its output: it ends with its sound
            feed_voice(s.voice);
            update_voice(s.voice);
        }
        uint32_t playing = 0;
        for (auto& [address, s] : sources)
            playing += s.voice.playing;
        LeaveCriticalSection(&lock);
        last_end = GetTickCount();
        longest_pass = last_end - pass_start > longest_pass ? last_end - pass_start : longest_pass;
        if (last_end - last_stats >= 5000) {
            // Every 5 s: whether the voices kept ahead of the APU.
            log("xma: %u voices playing, %u ran dry, writes at least %d frames ahead; longest pass %u ms, gap %u "
                "ms, read %u ms\n",
                playing, voices_dry - dry_logged, least_lead == ~0u ? -1 : int(least_lead), longest_pass,
                longest_gap, longest_read);
            dry_logged = voices_dry;
            least_lead = ~0u;
            longest_pass = longest_gap = longest_read = 0;
            last_stats = last_end;
        }

        if (stray_kicks != logged_stray) {
            log("xma: %u kicks of contexts not allocated\n", stray_kicks);
            logged_stray = stray_kicks;
        }
        static uint32_t logged_buffers;
        if ((logged_kicks == 0 && kicks) || (logged_buffers == 0 && native_buffers) ||
            (++passes % 10000 == 0 && (kicks != logged_kicks || native_buffers != logged_buffers))) {
            logged_buffers = native_buffers;
            uint32_t buffers = 0;
            for (const Context& c : state)
                buffers += c.voice.buffer != nullptr;
            log("xma: %u kicks, %u locks, %u clears, %u worked, %u packets not in the banks, %u voices started, "
                "%u voice matches, %u misses; %u voice buffers, %u sounds loaded (%u KB kept); %u voices ran dry, "
                "writes at least %d frames ahead; %u buffers on the APU, %u to XAudio, %u source voices\n",
                kicks, locks, clears, worked, unknown_packets, voices_started, voice_matches, voice_misses, buffers,
                unsigned(loaded.size()), kept_bytes >> 10, voices_dry, least_lead == ~0u ? -1 : int(least_lead),
                native_buffers, xaudio_buffers, unsigned(sources.size()));
            least_lead = ~0u;
            logged_kicks = kicks;
        }
        Sleep(1);
    }
}

// A packet's header (its first 4 bytes, the first in the top byte) without
// touching a released page: the saved header then, and the zeros an
// untouched page holds.
uint32_t packet_header(uint32_t address) {
    if (guest::resident(address)) {
        uint32_t header = guest::load32(address);
        if (header)
            return header;
    }
    uint32_t header = 0;
    EnterCriticalSection(&saved_headers_lock);
    auto it = saved_headers.find(translate_physical(address));
    if (it != saved_headers.end())
        header = __builtin_bswap32(it->second);
    LeaveCriticalSection(&saved_headers_lock);
    return header;
}

// A byte of XMA data as it reads, without touching a page that has no
// memory (it reads as zeros, as a released one does).
uint32_t peek(uint32_t address) { return guest::resident(address) ? guest::load8(address) : 0; }

// The bit offset in the packet at packet (with that header) past its first
// frame and count more: the frames' lengths are their first 15 bits.
uint32_t frame_bits(uint32_t packet, uint32_t header, uint32_t count) {
    uint32_t bit = ((((header >> 24) & 3) << 8 | ((header >> 16) & 0xFF)) << 5 | ((header >> 8) & 0xFF) >> 3) + 32;
    for (; count; count--) {
        uint32_t byte = packet + (bit >> 3), shift = bit & 7, v;
        if (shift < 2) {
            v = peek(byte) << 8 | peek(byte + 1);
            shift = 1 - shift;
        } else {
            v = (peek(byte) << 16) | (peek(byte + 1) << 8) | peek(byte + 2);
            shift = 9 - shift;
        }
        bit += (v >> shift) & 0x7FFF;
    }
    return bit;
}

}  // namespace

namespace xma {
// From XAudio's resampler (xaudio.cpp): the voice reading at input.
void voice_params(uint32_t input, float ratio, float gain) {
    static uint32_t last;
    Context* c = &state[last];
    if (!(c->allocated && input >= c->output_start && input < c->output_end)) {
        c = nullptr;
        for (uint32_t n = 0; n < kContexts; n++) {
            if (state[n].allocated && input >= state[n].output_start && input < state[n].output_end) {
                c = &state[n];
                last = n;
                break;
            }
        }
    }
    if (!c) {
        voice_misses++;
        return;
    }
    voice_matches++;
    c->voice.ratio = ratio;
    c->voice.gain = gain;
}

// From runtime/io.cpp, before it releases [address, address + size): the
// packets' headers, for process_packet once their pages are gone.
void save_packet_headers(uint32_t address, uint32_t size) {
    EnterCriticalSection(&saved_headers_lock);
    for (uint32_t offset = 0; offset + kBytesPerPacket <= size; offset += kBytesPerPacket) {
        uint32_t header;
        memcpy(&header, guest::host(address + offset), 4);
        saved_headers[translate_physical(address + offset)] = header;
    }
    LeaveCriticalSection(&saved_headers_lock);
}

// From runtime/io.cpp: the title has read size bytes of the sound at path
// (D:\...) from offset into address. The whole packets read are the
// sound's from then on, so the decoder knows them wherever the title put
// them; an effect's blocks are read now (the title loads a level's effects
// together, and they must start at once).
void sound_read(const char* path, uint32_t address, uint32_t offset, uint32_t size) {
    std::string name = path;
    if (name.size() < 3 || _strnicmp(name.c_str(), "D:\\", 3) != 0)
        return;
    name = name.substr(3);
    for (char& ch : name)
        ch = char(tolower(uint8_t(ch)));
    Bank* bank = nullptr;
    const BankEntry* entry = nullptr;
    for (Bank& b : banks) {
        auto it = b.by_name.find(name);
        if (it != b.by_name.end()) {
            bank = &b;
            entry = &b.entries[it->second];
        }
    }
    if (!entry)
        return;
    uint32_t kDataStart = entry->data_start;  // where its packets start in the file
    uint32_t data = offset > kDataStart ? offset - kDataStart : 0;
    uint32_t first = (data + kBytesPerPacket - 1) / kBytesPerPacket;  // the first whole packet
    uint32_t start = address + (kDataStart + first * kBytesPerPacket - offset);
    if (offset + size < kDataStart + (first + 1) * kBytesPerPacket)
        return;  // not a whole packet
    uint32_t end = start + (offset + size - kDataStart - first * kBytesPerPacket) / kBytesPerPacket * kBytesPerPacket;
    Loaded l;
    l.end = end;
    l.first_packet = first;
    l.bank = bank;
    l.entry = entry;
    l.packets.resize(entry->packets);
    EnterCriticalSection(&bank->file_lock);
    DWORD read = 0;
    SetFilePointer(bank->file, LONG(entry->table), nullptr, FILE_BEGIN);
    ReadFile(bank->file, l.packets.data(), entry->packets * sizeof(PacketInfo), &read, nullptr);
    LeaveCriticalSection(&bank->file_lock);
    uint32_t bytes = entry->streams * entry->stream_bytes;
    if (bank->keep && entry->channels && bytes <= kKeptBytes / 4) {
        l.blocks.resize(bytes);
        read_blocks(*bank, entry->offset, l.blocks.data(), bytes);
    }
    EnterCriticalSection(&loaded_lock);
    // What the title loaded there before is gone.
    for (auto it = loaded.lower_bound(start); it != loaded.end() && it->first < end;) {
        kept_bytes -= uint32_t(it->second.blocks.size());
        it = loaded.erase(it);
    }
    auto before = loaded.lower_bound(start);
    if (before != loaded.begin() && (--before)->second.end > start) {
        kept_bytes -= uint32_t(before->second.blocks.size());
        loaded.erase(before);
    }
    l.sequence = ++loaded_sequence;
    kept_bytes += uint32_t(l.blocks.size());
    loaded[start] = std::move(l);
    // Over the budget: the oldest effects' blocks go (read as they play).
    while (kept_bytes > kKeptBytes) {
        Loaded* oldest = nullptr;
        for (auto& [at, s] : loaded)
            if (!s.blocks.empty() && (!oldest || s.sequence < oldest->sequence))
                oldest = &s;
        if (!oldest)
            break;
        kept_bytes -= uint32_t(oldest->blocks.size());
        std::vector<uint8_t>().swap(oldest->blocks);
    }
    LeaveCriticalSection(&loaded_lock);
}

void init() {
    InitializeCriticalSection(&lock);
    load_bank(banks[0], "D:\\audio\\Sounds.jpa", true);
    load_bank(banks[1], "D:\\audio\\Music.jpa", false);
    contexts = guest::alloc(kContexts * kContextSize, 256);
    *reg(kContextArrayAddress) = LONG(contexts & 0x1FFFFFFF);
    *reg(kNextContextIndex) = 1;
    kernel::start_runtime_thread(decoder_thread, nullptr, 0x10000);
}
}  // namespace xma

IMPORT(XMACreateContext) {
    // (context*)
    uint32_t context = 0;
    EnterCriticalSection(&lock);
    for (uint32_t i = 0; i < kContexts; i++) {
        if (!state[i].allocated) {
            state[i] = Context();
            state[i].allocated = true;
            context = context_address(i);
            memset(guest::host(context), 0, kContextSize);
            break;
        }
    }
    LeaveCriticalSection(&lock);
    guest::store32(ctx.r3.u32, context);
    ctx.r3.u64 = context ? kSuccess : kNoMemory;
}

IMPORT(XMAReleaseContext) {
    uint32_t i = (ctx.r3.u32 - contexts) / kContextSize;
    if (i < kContexts) {
        EnterCriticalSection(&lock);
        state[i].allocated = false;
        state[i].enabled = false;
        // Its voice's buffer goes with it: the title creates far more
        // contexts over time than play at once.
        Voice& v = state[i].voice;
        end_feed(v);
        if (v.buffer)
            v.buffer->Release();
        v.buffer = nullptr;
        memset(guest::host(context_address(i)), 0, kContextSize);
        LeaveCriticalSection(&lock);
    }
    ctx.r3.u64 = 0;
}

// The title's XMA seek (8210C5B0): the bit offset in the data (r3, r4 bytes)
// of frame r6, counting from packet r5; 0 past the end. It walks the packets'
// headers, then the frame lengths (15 bits each) in the packet the frame is
// in. Here released pages are not touched (runtime/io.cpp releases sounds'
// data): the walk would bring the whole sound back.
PPC_FUNC(sub_8210C5B0) {
    uint32_t start = ctx.r3.u32, end = start + ctx.r4.u32, target = ctx.r6.u32;
    uint32_t at = start + (ctx.r5.u32 << 11), frames = 0, header = 0;
    for (;; at += ((header & 0xFF) + 1) << 11) {
        if (at + 2048 > end) {
            ctx.r3.u64 = 0;
            return;
        }
        header = packet_header(at);
        uint32_t through = (header >> 26) + frames;
        if (through > target)
            break;
        frames = through;
    }
    ctx.r3.u64 = ((at - start) << 3) + frame_bits(at, header, target - frames);
}

// The title's other XMA seek (8210C690): the bit offset in the data (r3, r4
// bytes) of the last frame of the stream whose first packet is r5. The last
// packet with frames is taken to be one stride before a last packet with
// none. As 8210C5B0, without touching released pages.
PPC_FUNC(sub_8210C690) {
    uint32_t start = ctx.r3.u32, end = start + ctx.r4.u32;
    uint32_t last = start + (ctx.r5.u32 << 11);
    uint32_t stride = ((packet_header(last) & 0xFF) + 1) << 11;
    for (uint32_t at = last + stride; at + 2048 <= end; at += stride) {
        last = at;
        stride = ((packet_header(at) & 0xFF) + 1) << 11;
    }
    if (!(packet_header(last) >> 26))
        last -= stride;
    uint32_t header = packet_header(last);
    ctx.r3.u64 = ((last - start) << 3) + frame_bits(last, header, (header >> 26) - 1);
}

// --- XAudio's source voices on the APU ------------------------------------------
//
// The source voice's methods (its vtable is 82008D40): every sound the title
// plays is a whole .xma file in one buffer (its audio manager, 820C0BB0 on),
// so each voice plays its converted sound on an APU voice (Source) instead of
// XAudio decoding and mixing it. The rest of its methods (details, effects,
// filters, routing) stay XAudio's. The voice object: +64 its channels, +212
// its sample rate.

extern "C" PPC_FUNC(__imp__sub_8210D6D8);
extern "C" PPC_FUNC(__imp__sub_82109DA0);
extern "C" PPC_FUNC(__imp__sub_82109EF8);
extern "C" PPC_FUNC(__imp__sub_8210C890);
extern "C" PPC_FUNC(__imp__sub_8210C990);
extern "C" PPC_FUNC(__imp__sub_821087F8);
extern "C" PPC_FUNC(__imp__sub_8210F7A8);
extern "C" PPC_FUNC(__imp__sub_82108A88);
extern "C" PPC_FUNC(__imp__sub_82108B10);
extern "C" PPC_FUNC(__imp__sub_821097B0);

namespace {

// The voice's Source when its buffer is a converted sound (lock held).
Source* native_source(uint32_t voice) {
    auto it = sources.find(voice);
    return it != sources.end() && it->second.native ? &it->second : nullptr;
}

float float_at(uint32_t address) {
    uint32_t bits = guest::load32(address);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

const char* sound_name(const Bank* bank, const BankEntry* entry) {
    for (const auto& [name, i] : bank->by_name)
        if (&bank->entries[i] == entry)
            return name.c_str();
    return "?";
}

}  // namespace

// SubmitSourceBuffer(voice, XAUDIO2_BUFFER*: +8 the data, +20 loop begin and
// +24 loop length in samples, +28 loop count (255 for ever), +32 context).
PPC_FUNC(sub_8210D6D8) {
    uint32_t voice = ctx.r3.u32, buffer = ctx.r4.u32;
    uint32_t data = guest::load32(buffer + 8);
    PacketRef p;
    bool found = data && find_packet(data, p) && p.entry->channels;
    EnterCriticalSection(&lock);
    Source& s = sources[voice];
    s.native = found;
    if (found) {
        Voice& v = s.voice;
        end_feed(v);
        Feed& f = v.feed;
        f.bank = p.bank;
        f.entry = p.entry;
        f.key = p.key;
        f.stream = p.stream;
        f.next = p.first_frame;
        f.wraps = 0;
        f.out_channels = p.entry->channels < 2 ? 1 : 2;
        uint32_t rate = guest::load32(voice + 212);
        f.rate = rate >= 8000 && rate <= 96000 ? rate : 48000;
        uint32_t loop_begin = guest::load32(buffer + 20), loop_length = guest::load32(buffer + 24);
        uint32_t loop_count = guest::load32(buffer + 28);
        f.loop_start = f.loop_end = 0;
        f.loops = 255;
        if (loop_count) {
            f.loop_start = p.first_frame + loop_begin / kFrameSamples;
            f.loop_end = loop_length ? p.first_frame + (loop_begin + loop_length + kFrameSamples - 1) / kFrameSamples
                                     : p.entry->frames;
            if (f.loop_end > p.entry->frames)
                f.loop_end = p.entry->frames;
            if (f.loop_end <= f.loop_start)
                f.loop_end = 0;
            f.loops = loop_count;
        }
        s.first_frame = p.first_frame;
        s.context = guest::load32(buffer + 32);
        v.ratio = float(f.rate) * s.ratio / 48000.0f;
        v.gain = s.volume;
        v.last_taken = GetTickCount();
        native_buffers++;
        static int said;  // a few: each line over the serial port stalls the title
        if (said++ < 3)
            log("xma: voice %08X plays %s at %u Hz, %u channels, frames %u of %u, loop %u-%u x%u\n", voice,
                sound_name(p.bank, p.entry), f.rate, f.out_channels, f.next, p.entry->frames, f.loop_start, f.loop_end,
                loop_count);
    } else {
        xaudio_buffers++;
    }
    LeaveCriticalSection(&lock);
    if (found) {
        ctx.r3.u64 = 0;
        return;
    }
    static int said;
    if (said++ < 10)
        log("xma: voice %08X: buffer %08X (%u bytes) is not a converted sound: XAudio plays it\n", voice, data,
            guest::load32(buffer + 4));
    __imp__sub_8210D6D8(ctx, base);
}

// Start(voice, flags, operation set).
PPC_FUNC(sub_82109DA0) {
    EnterCriticalSection(&lock);
    Source& s = sources[ctx.r3.u32];
    s.started = true;
    bool native = s.native;
    if (native && s.voice.playing && s.voice.buffer)
        s.voice.buffer->Play(0, 0, DSBPLAY_LOOPING);  // after a Stop: where it was
    LeaveCriticalSection(&lock);
    if (native) {
        ctx.r3.u64 = 0;
        return;
    }
    __imp__sub_82109DA0(ctx, base);
}

// Stop(voice, flags, operation set): paused where it is.
PPC_FUNC(sub_82109EF8) {
    EnterCriticalSection(&lock);
    Source& s = sources[ctx.r3.u32];
    s.started = false;
    bool native = s.native;
    if (native && s.voice.playing && s.voice.buffer)
        s.voice.buffer->Stop();
    LeaveCriticalSection(&lock);
    if (native) {
        ctx.r3.u64 = 0;
        return;
    }
    __imp__sub_82109EF8(ctx, base);
}

// FlushSourceBuffers(voice): its buffer is done with.
PPC_FUNC(sub_8210C890) {
    EnterCriticalSection(&lock);
    Source* s = native_source(ctx.r3.u32);
    if (s)
        end_feed(s->voice);
    LeaveCriticalSection(&lock);
    if (s) {
        ctx.r3.u64 = 0;
        return;
    }
    __imp__sub_8210C890(ctx, base);
}

// GetState(voice, XAUDIO2_VOICE_STATE*: +0 the buffer's context, +4 buffers
// queued, +8 samples played unless flags has 0x100). Its buffer is queued
// until its sound has played out (or was flushed).
PPC_FUNC(sub_8210C990) {
    uint32_t state = ctx.r4.u32, flags = ctx.r5.u32;
    EnterCriticalSection(&lock);
    Source* s = native_source(ctx.r3.u32);
    if (s) {
        const Feed& f = s->voice.feed;
        bool queued = f.entry != nullptr;
        guest::store32(state, queued ? s->context : 0);
        guest::store32(state + 4, queued ? 1 : 0);
        if (!(flags & 0x100)) {
            uint32_t frames = f.next >= s->first_frame ? f.next - s->first_frame : 0;
            guest::store64(state + 8, uint64_t(frames + f.wraps * (f.loop_end - f.loop_start)) * kFrameSamples);
        }
    }
    LeaveCriticalSection(&lock);
    if (!s)
        __imp__sub_8210C990(ctx, base);
}

// SetVolume(voice, f1 the gain, operation set).
PPC_FUNC(sub_821087F8) {
    EnterCriticalSection(&lock);
    Source& s = sources[ctx.r3.u32];
    s.volume = float(ctx.f1.f64);
    s.voice.gain = s.volume;
    bool native = s.native;
    LeaveCriticalSection(&lock);
    if (native) {
        ctx.r3.u64 = 0;
        return;
    }
    __imp__sub_821087F8(ctx, base);
}

// SetOutputMatrix(voice, destination, r5 source by r6 destination channels,
// r7 the gains, operation set): the pan. The title's audio manager sets a
// mono sound's front left and right gains (0 and 1), a stereo one's left
// gain at 0 and its right at 10.
PPC_FUNC(sub_8210F7A8) {
    uint32_t channels = ctx.r5.u32, outputs = ctx.r6.u32, matrix = ctx.r7.u32;
    EnterCriticalSection(&lock);
    Source* s = native_source(ctx.r3.u32);
    if (s && matrix && outputs >= 2) {
        uint32_t right = channels == 1 ? 1 : 10;
        s->voice.left = float_at(matrix);
        s->voice.right = right < channels * outputs ? float_at(matrix + right * 4) : s->voice.left;
    }
    LeaveCriticalSection(&lock);
    if (s) {
        ctx.r3.u64 = 0;
        return;
    }
    __imp__sub_8210F7A8(ctx, base);
}

// ExitLoop(voice, operation set): it plays on to the end.
PPC_FUNC(sub_82108A88) {
    EnterCriticalSection(&lock);
    Source* s = native_source(ctx.r3.u32);
    if (s)
        s->voice.feed.loop_end = 0;
    LeaveCriticalSection(&lock);
    if (s) {
        ctx.r3.u64 = 0;
        return;
    }
    __imp__sub_82108A88(ctx, base);
}

// SetFrequencyRatio(voice, f1 the ratio, operation set).
PPC_FUNC(sub_82108B10) {
    EnterCriticalSection(&lock);
    Source& s = sources[ctx.r3.u32];
    float ratio = float(ctx.f1.f64);
    s.ratio = ratio < 1.0f / 1024 ? 1.0f / 1024 : ratio > 8 ? 8 : ratio;
    bool native = s.native;
    if (native)
        s.voice.ratio = float(s.voice.feed.rate) * s.ratio / 48000.0f;
    LeaveCriticalSection(&lock);
    if (native) {
        ctx.r3.u64 = 0;
        return;
    }
    __imp__sub_82108B10(ctx, base);
}

// DestroyVoice(voice): its APU voice goes with it.
PPC_FUNC(sub_821097B0) {
    EnterCriticalSection(&lock);
    auto it = sources.find(ctx.r3.u32);
    if (it != sources.end()) {
        Voice& v = it->second.voice;
        end_feed(v);
        if (v.buffer)
            v.buffer->Release();
        sources.erase(it);
    }
    LeaveCriticalSection(&lock);
    __imp__sub_821097B0(ctx, base);
}
