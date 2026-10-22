// XAudio's render driver.
//
// The title's XAudio registers a client: a callback the system calls every
// audio frame (256 samples at 48 kHz, 5.33 ms) on its own thread, which mixes
// a frame and hands it back with XAudioSubmitRenderDriverFrame (six channels
// of big-endian floats), which go to the speakers through DirectSound as
// stereo. The XMA decoder (xma.cpp) feeds the title's voices the converted
// wave banks. Follows Xenia's src/xenia/apu and xboxkrnl_audio*.cc.
#include "kernel.h"

#include <dsound.h>

namespace xaudio {
void log_profile(uint32_t seconds);  // xaudio.cpp
}  // namespace xaudio

namespace {

constexpr uint32_t kDriverTag = 0x41550000;
constexpr int kClients = 8;
constexpr uint32_t kFrameSamples = 256, kRate = 48000;

struct Client {
    uint32_t callback = 0, argument = 0;  // argument: guest word holding the title's argument
    GuestThread* thread = nullptr;
    volatile bool running = false;
    volatile uint32_t submitted = 0;
};
Client clients[kClients];
uint64_t output_cycles;  // in play_frame (statistics)

// The speakers: a looping DirectSound buffer the submitted frames are written
// into (48 kHz stereo), a few frames ahead of where it plays. Without one the
// frames are only counted.
constexpr uint32_t kOutputFrameBytes = kFrameSamples * 2 * 2;
constexpr uint32_t kOutputFrames = 32;        // 170 ms of buffer
constexpr uint32_t kOutputAheadFrames = 8;    // 43 ms queued: what the client keeps
struct Output {
    LPDIRECTSOUND ds = nullptr;
    LPDIRECTSOUNDBUFFER buffer = nullptr;
    uint32_t write = 0;  // byte offset of the next frame
} output;

bool open_output() {
    if (FAILED(DirectSoundCreate(nullptr, &output.ds, nullptr))) {
        log("audio: no DirectSound: silent\n");
        return false;
    }
    WAVEFORMATEX format = {};
    format.wFormatTag = WAVE_FORMAT_PCM;
    format.nChannels = 2;
    format.nSamplesPerSec = kRate;
    format.wBitsPerSample = 16;
    format.nBlockAlign = 4;
    format.nAvgBytesPerSec = kRate * 4;
    DSBUFFERDESC desc = {};
    desc.dwSize = sizeof(desc);
    desc.dwBufferBytes = kOutputFrames * kOutputFrameBytes;
    desc.lpwfxFormat = &format;
    if (FAILED(output.ds->CreateSoundBuffer(&desc, &output.buffer, nullptr))) {
        log("audio: no DirectSound buffer: silent\n");
        return false;
    }
    void* p;
    DWORD n;
    if (SUCCEEDED(output.buffer->Lock(0, desc.dwBufferBytes, &p, &n, nullptr, nullptr, 0))) {
        memset(p, 0, n);
        output.buffer->Unlock(p, n, nullptr, 0);
    }
    output.buffer->Play(0, 0, DSBPLAY_LOOPING);
    log("audio: DirectSound output, 48 kHz stereo\n");
    return true;
}

// Bytes written ahead of the play cursor.
uint32_t queued() {
    DWORD play = 0, write = 0;
    output.buffer->GetCurrentPosition(&play, &write);
    constexpr uint32_t size = kOutputFrames * kOutputFrameBytes;
    return (output.write + size - play) % size;
}

// A submitted frame (six channels of 256 big-endian floats). XAudio's mix
// is not heard: the voices play on the APU (runtime/xma.cpp), and XAudio's
// resamplers do not fill its input (runtime/xaudio.cpp). The speakers'
// buffer stays silent; the frame only moves the write position, which
// paces XAudio.
void play_frame(uint32_t samples) {
    (void)samples;
    output.write = (output.write + kOutputFrameBytes) % (kOutputFrames * kOutputFrameBytes);
}

DWORD WINAPI client_thread(void* parameter) {
    guest::GuestFaults faults;
    Client& c = *static_cast<Client*>(parameter);
    set_current_thread(c.thread);
    LARGE_INTEGER frequency, start, now;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    uint64_t frames = 0, last_frames = 0, mix_cycles = 0;
    uint32_t dropped = 0, owed = 0;  // owed: frames to mix before reading the play position again
    DWORD last_log = GetTickCount();
    while (c.running) {
        if (output.buffer) {
            // Paced by the speakers: a frame whenever fewer than
            // kOutputAheadFrames are queued. Fallen behind (the play cursor
            // overtook the frames written): start again just ahead of it.
            // The play position is costly to read (the APU's): once per
            // run of the frames the queue is short of.
            if (!owed) {
                DirectSoundDoWork();
                uint32_t ahead = queued();
                if (ahead > (kOutputFrames - 2) * kOutputFrameBytes) {
                    DWORD play = 0, write = 0;
                    output.buffer->GetCurrentPosition(&play, &write);
                    output.write = (play / kOutputFrameBytes + 2) % kOutputFrames * kOutputFrameBytes;
                    ahead = 2 * kOutputFrameBytes;
                    dropped++;
                } else if (ahead >= kOutputAheadFrames * kOutputFrameBytes) {
                    // Until the queue is down to the mark.
                    uint32_t spare = ahead - kOutputAheadFrames * kOutputFrameBytes;
                    uint32_t ms = spare / (kRate * 4 / 1000);
                    Sleep(ms ? ms : 1);
                    continue;
                }
                owed = (kOutputAheadFrames * kOutputFrameBytes - ahead) / kOutputFrameBytes + 1;
            }
            owed--;
            PPCContext& ctx = c.thread->ctx;
            ctx.r1.u64 = c.thread->stack + c.thread->stack_size - 0x100;
            ctx.r3.u64 = c.argument;
            uint64_t t0 = __builtin_ia32_rdtsc();
            uint32_t submitted = c.submitted;
            guest::call(ctx, c.callback);
            uint64_t mixed = __builtin_ia32_rdtsc() - t0;
            mix_cycles += mixed;
            // A frame that takes longer than real time (5.3 ms) would starve
            // the title's threads, this one running above them: then at most
            // half the processor (the sound breaks up instead).
            uint32_t mixed_ms = uint32_t(mixed / 733000);
            if (mixed_ms > 5)
                Sleep(mixed_ms);
            if (c.submitted == submitted) {
                // No frame (XAudio not started yet): this thread runs above
                // the title's, which must get to start it.
                Sleep(1);
                continue;
            }
            frames++;
            // Every 5 s: mixed frames a second (187.5 is real time), the
            // title's mixing time per frame, and underruns.
            DWORD now_ms = GetTickCount();
            if (now_ms - last_log >= 5000) {
                uint64_t n = frames - last_frames;
                log("audio: %u frames/s (187 needed), %u us mixing each (%u us of it output), %u underruns\n",
                    uint32_t(n / 5), uint32_t(n ? mix_cycles / n / 733 : 0), uint32_t(n ? output_cycles / n / 733 : 0),
                    dropped);
                output_cycles = 0;
                xaudio::log_profile(5);
                last_log = now_ms;
                last_frames = frames;
                mix_cycles = 0;
            }
            continue;
        }
        // One callback per frame of real time.
        QueryPerformanceCounter(&now);
        uint64_t due = uint64_t(now.QuadPart - start.QuadPart) * kRate / kFrameSamples / uint64_t(frequency.QuadPart);
        if (frames >= due) {
            Sleep(1);
            continue;
        }
        // Behind (mixing is slower than real time): drop frames rather than
        // catch up, which would take all of the processor from the title.
        if (due - frames > 4) {
            dropped += uint32_t(due - frames - 1);
            frames = due - 1;
        }
        PPCContext& ctx = c.thread->ctx;
        ctx.r1.u64 = c.thread->stack + c.thread->stack_size - 0x100;
        ctx.r3.u64 = c.argument;
        guest::call(ctx, c.callback);
        frames++;
        if (frames == 1 || frames % (kRate / kFrameSamples * 10) == 0)
            log("audio: %u frames called, %u submitted, %u dropped\n", uint32_t(frames), c.submitted, dropped);
    }
    return 0;
}

int client_of(uint32_t driver) {
    int index = int(driver & 0xFFFF);
    return (driver & 0xFFFF0000) == kDriverTag && index < kClients ? index : -1;
}

}  // namespace

IMPORT(XAudioRegisterRenderDriverClient) {
    // ({callback, argument}*, driver*)
    uint32_t callback = guest::load32(ctx.r3.u32), argument = guest::load32(ctx.r3.u32 + 4);
    if (!callback) {
        ctx.r3.u64 = 0x80070057;  // E_INVALIDARG
        return;
    }
    for (int i = 0; i < kClients; i++) {
        Client& c = clients[i];
        if (c.running)
            continue;
        c.callback = callback;
        c.argument = guest::alloc(guest::kPage);
        guest::store32(c.argument, argument);
        if (!c.thread)
            c.thread = guest_thread_state(0x10000, callback);
        c.running = true;
        // Above the title's other threads: a frame is due every 5.3 ms, and
        // at their priority it waits its turn behind them.
        HANDLE thread = CreateThread(nullptr, kernel::kGuestHostStack, client_thread, &c, CREATE_SUSPENDED, nullptr);
        if (!thread)
            fatal("audio: cannot start the audio thread\n");
        SetThreadPriority(thread, THREAD_PRIORITY_ABOVE_NORMAL);
        ResumeThread(thread);
        guest::store32(ctx.r4.u32, kDriverTag | uint32_t(i));
        log("audio: client %d, callback %08X\n", i, callback);
        ctx.r3.u64 = 0;
        return;
    }
    ctx.r3.u64 = 0x8007000E;  // E_OUTOFMEMORY
}

IMPORT(XAudioUnregisterRenderDriverClient) {
    int i = client_of(ctx.r3.u32);
    if (i >= 0)
        clients[i].running = false;
    ctx.r3.u64 = 0;
}

IMPORT(XAudioSubmitRenderDriverFrame) {
    // (driver, samples)
    int i = client_of(ctx.r3.u32);
    if (i >= 0) {
        clients[i].submitted = clients[i].submitted + 1;
        uint64_t t0 = __builtin_ia32_rdtsc();
        play_frame(ctx.r4.u32);
        output_cycles += __builtin_ia32_rdtsc() - t0;
    }
    ctx.r3.u64 = 0;
}

IMPORT(XAudioGetVoiceCategoryVolumeChangeMask) {
    guest::store32(ctx.r4.u32, 0);
    ctx.r3.u64 = 0;
}

IMPORT(XAudioGetVoiceCategoryVolume) {
    float volume = 1.0f;
    uint32_t bits;
    memcpy(&bits, &volume, 4);
    guest::store32(ctx.r4.u32, bits);
    ctx.r3.u64 = 0;
}

// The XMA decoder (xma.cpp).
namespace xma {
void init();
}  // namespace xma


namespace audio {
// The device, for the XMA decoder's voices.
LPDIRECTSOUND device() { return output.ds; }

void init() {
    open_output();
    xma::init();
}
}  // namespace audio
