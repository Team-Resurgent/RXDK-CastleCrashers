// xam.xex: the controller, memory, and the system UI and Live services an
// offline original Xbox does not have (no profile is ever signed in).
#include "kernel.h"
#include "gpu/renderer.h"

#include <cstdlib>
#include <map>
#include <vector>

namespace {

// XAM error codes the title tests.
enum : uint32_t {
    kErrorSuccess = 0,
    kErrorDeviceNotConnected = 0x48F,
    kErrorNoSuchUser = 0x525,
    kErrorNoMoreFiles = 0x12,
    kErrorNotFound = 0x490,
    kErrorCancelled = 0x4C7,
    kErrorFunctionFailed = 0x65B,
};

// The original Xbox controller as an Xbox 360 one.
class Pads {
public:
    void init() {
        polling_.fAutoPoll = TRUE;
        polling_.fInterruptOut = TRUE;
        polling_.bInputInterval = 8;
        polling_.bOutputInterval = 8;
        XInitDevices(0, nullptr);
        // USB devices enumerate after XInitDevices: wait for a controller
        // (the title quits at start when it finds none).
        for (int waited = 0; waited < 3000 && !(connected_ = XGetDevices(XDEVICE_TYPE_GAMEPAD)); waited += 50)
            Sleep(50);
        log("xam: controllers %08X\n", unsigned(connected_));
        open_all();
    }

    // The first connected controller as the 360 user 0's; false if none.
    bool state(uint32_t& packet, uint16_t& buttons, uint8_t& lt, uint8_t& rt, int16_t sticks[4]) {
        if (script_.size()) {
            // A test script plays instead of the controller.
            DWORD now = GetTickCount() - script_start_;
            while (script_next_ < script_.size() && script_[script_next_].ms <= now) {
                const Step& step = script_[script_next_++];
                if (step.command)
                    run_command(step);
                else
                    script_step_ = step;
            }
            if (holding_)  // "hold": the cheat, every poll
                guest::poke_candidates(hold_value_);
            buttons = script_step_.buttons;
            lt = script_step_.lt;
            rt = script_step_.rt;
            sticks[0] = script_step_.lx;
            sticks[1] = script_step_.ly;
            sticks[2] = sticks[3] = 0;
            packet = ++script_packet_;
            return true;
        }
        DWORD inserted = 0, removed = 0;
        if (XGetDeviceChanges(XDEVICE_TYPE_GAMEPAD, &inserted, &removed)) {
            for (int port = 0; port < 4; ++port)
                if ((removed & (1u << port)) && handles_[port]) {
                    XInputClose(handles_[port]);
                    handles_[port] = nullptr;
                }
            connected_ = (connected_ & ~removed) | inserted;
            open_all();
        }
        for (int port = 0; port < 4; ++port) {
            XINPUT_STATE s;
            if (!handles_[port] || XInputGetState(handles_[port], &s) != ERROR_SUCCESS)
                continue;
            const XINPUT_GAMEPAD& pad = s.Gamepad;
            buttons = uint16_t(pad.wButtons & 0xFF);  // D-pad, start, back, thumbs: the same bits
            auto press = [&](int analog, uint16_t bit) {
                if (pad.bAnalogButtons[analog] > 30)
                    buttons |= bit;
            };
            press(XINPUT_GAMEPAD_A, 0x1000);
            press(XINPUT_GAMEPAD_B, 0x2000);
            press(XINPUT_GAMEPAD_X, 0x4000);
            press(XINPUT_GAMEPAD_Y, 0x8000);
            press(XINPUT_GAMEPAD_WHITE, 0x0100);  // left shoulder
            press(XINPUT_GAMEPAD_BLACK, 0x0200);  // right shoulder
            lt = pad.bAnalogButtons[XINPUT_GAMEPAD_LEFT_TRIGGER];
            rt = pad.bAnalogButtons[XINPUT_GAMEPAD_RIGHT_TRIGGER];
            sticks[0] = pad.sThumbLX;
            sticks[1] = pad.sThumbLY;
            sticks[2] = pad.sThumbRX;
            sticks[3] = pad.sThumbRY;
            packet = s.dwPacketNumber;
            return true;
        }
        return false;
    }

    // D:\inputs.txt, when there is one: lines of "<milliseconds> <buttons in
    // hex> [<left trigger> <right trigger> <left stick x> <left stick y>]"
    // (the 360's XINPUT bits, then decimal), held from that time on. For
    // tests: remove it from the title's folder to play.
    void load_script() {
        HANDLE f = CreateFileA("D:\\inputs.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (f == INVALID_HANDLE_VALUE)
            return;
        static char text[16384];
        DWORD read = 0;
        ReadFile(f, text, sizeof(text) - 1, &read, nullptr);
        CloseHandle(f);
        text[read] = 0;
        for (char* line = text; *line;) {
            char* end;
            unsigned long ms = strtoul(line, &end, 10);
            if (end != line) {
                // The optional fields: only within this line.
                char* eol = end;
                while (*eol && *eol != '\n')
                    eol++;
                char saved = *eol;
                *eol = 0;
                Step step = {uint32_t(ms), 0};
                while (*end == ' ' || *end == '\t')
                    end++;
                if (*end == 's' || *end == 'p' || *end == 'h') {  // scan, poke, hold <value>
                    step.command = *end;
                    while (*end && *end != ' ' && *end != '\t')
                        end++;
                    step.value = uint32_t(strtoul(end, &end, 0));
                } else {
                    step.buttons = uint16_t(strtoul(end, &end, 16));
                    step.lt = uint8_t(strtoul(end, &end, 10));
                    step.rt = uint8_t(strtoul(end, &end, 10));
                    step.lx = int16_t(strtol(end, &end, 10));
                    step.ly = int16_t(strtol(end, &end, 10));
                }
                script_.push_back(step);
                *eol = saved;
                end = eol;
            }
            while (*end && *end != '\n')
                end++;
            line = *end ? end + 1 : end;
        }
        script_start_ = GetTickCount();
        log("xam: playing %u input script steps from D:\\inputs.txt\n", unsigned(script_.size()));
    }

private:
    struct Step {
        uint32_t ms;
        uint16_t buttons;
        uint8_t lt = 0, rt = 0;
        int16_t lx = 0, ly = 0;
        char command = 0;  // 's'can, 'p'oke or 'h'old guest dwords (memory.cpp scan)
        uint32_t value = 0;
    };
    bool holding_ = false;
    uint32_t hold_value_ = 0;

    void run_command(const Step& step) {
        if (step.command == 's') {
            uint32_t left = guest::scan(step.value);
            log("xam: scan for %u: %u candidates\n", unsigned(step.value), unsigned(left));
            const auto& c = guest::scan_candidates();
            for (size_t i = 0; i < c.size() && i < 16; i++)
                log("xam:   candidate %08X\n", unsigned(c[i]));
        } else if (step.command == 'p') {
            guest::poke_candidates(step.value);
            log("xam: poked %u into %u candidates\n", unsigned(step.value), unsigned(guest::scan_candidates().size()));
        } else if (step.command == 'h') {
            holding_ = true;
            hold_value_ = step.value;
            log("xam: holding %u in %u candidates\n", unsigned(step.value), unsigned(guest::scan_candidates().size()));
        }
    }
    std::vector<Step> script_;
    size_t script_next_ = 0;
    Step script_step_ = {0, 0};
    DWORD script_start_ = 0;
    uint32_t script_packet_ = 0;
    HANDLE handles_[4] = {};
    DWORD connected_ = 0;
    XINPUT_POLLING_PARAMETERS polling_ = {};

    void open_all() {
        for (DWORD port = 0; port < 4; ++port)
            if ((connected_ & (1u << port)) && !handles_[port])
                handles_[port] = XInputOpen(XDEVICE_TYPE_GAMEPAD, port, XDEVICE_NO_SLOT, &polling_);
    }
} pads;

// An XOVERLAPPED the title passed: completed at once with result.
uint32_t complete(uint32_t overlapped, uint32_t result, uint32_t extended_error = 0) {
    if (!overlapped)
        return result;
    // XOVERLAPPED: InternalLow (result), InternalHigh (length),
    // InternalContext, hEvent, completion routine and context, extended error
    guest::store32(overlapped, result);
    guest::store32(overlapped + 4, 0);
    guest::store32(overlapped + 24, extended_error);
    uint32_t event = guest::load32(overlapped + 12);
    if (event)
        SetEvent(kernel::host_handle(event));
    return 0x3E5;  // ERROR_IO_PENDING: the result is in the XOVERLAPPED
}

}  // namespace

namespace xam {
void init() {
    pads.init();
    pads.load_script();
}
}  // namespace xam

IMPORT(XamInputGetState) {
    // (user, flags, XINPUT_STATE*)
    uint32_t packet;
    uint16_t buttons = 0;
    uint8_t lt = 0, rt = 0;
    int16_t sticks[4] = {};
    if (ctx.r3.u32 != 0 || !pads.state(packet, buttons, lt, rt, sticks)) {
        ctx.r3.u64 = kErrorDeviceNotConnected;
        return;
    }
    // Logged on change: buttons, triggers and the left stick's direction.
    auto direction = [](int16_t v) { return v > 8000 ? 1 : v < -8000 ? -1 : 0; };
    uint32_t summary = buttons | (lt > 30 ? 0x10000 : 0) | (rt > 30 ? 0x20000 : 0) |
                       ((direction(sticks[0]) & 3) << 18) | ((direction(sticks[1]) & 3) << 20);
    static uint32_t last_summary;
    if (summary != last_summary)
        log("xam: buttons %04X triggers %u/%u stick %d,%d at %lu ms\n", buttons, lt, rt, direction(sticks[0]),
            direction(sticks[1]), (unsigned long)GetTickCount());
    last_summary = summary;
    // Diagnostic: A held for 300 ms logs one frame's draws (what firing draws).
    static DWORD a_since;
    static bool a_logged;
    if (buttons & 0x1000) {
        DWORD now = GetTickCount();
        if (!a_since)
            a_since = now;
        if (!a_logged && now - a_since >= 300) {
            gpu::renderer::request_debug_frame();
            a_logged = true;
        }
    } else {
        a_since = 0;
        a_logged = false;
    }
    uint32_t s = ctx.r5.u32;
    guest::store32(s, packet);
    guest::store16(s + 4, buttons);
    guest::store8(s + 6, lt);
    guest::store8(s + 7, rt);
    for (int i = 0; i < 4; i++)
        guest::store16(s + 8 + i * 2, uint16_t(sticks[i]));
    ctx.r3.u64 = kErrorSuccess;
}

IMPORT(XamAlloc) {
    // (flags, size, address*)
    uint32_t address = guest::alloc(ctx.r4.u32 ? ctx.r4.u32 : 1);
    guest::store32(ctx.r5.u32, address);
    ctx.r3.u64 = address ? kSuccess : kNoMemory;
}

IMPORT(XamFree) {
    guest::free(ctx.r3.u32);
    ctx.r3.u64 = kSuccess;
}

IMPORT(XamGetSystemVersion) { ctx.r3.u64 = 0x20044C80; }  // 2.0.17559.0

PPC_EXTERN_FUNC(__imp__RtlImageXexHeaderField);

IMPORT(XamGetExecutionId) {
    // (XEX_EXECUTION_ID**): the one in the XEX header (key 0x40006).
    PPCContext query = ctx;
    query.r3.u64 = 0;
    query.r4.u64 = 0x00040006;
    __imp__RtlImageXexHeaderField(query, base);
    guest::store32(ctx.r3.u32, query.r3.u32);
    ctx.r3.u64 = query.r3.u32 ? kSuccess : kNotFound;
}

// No launch data: the title was started from the dashboard.
IMPORT(XamLoaderGetLaunchDataSize) { ctx.r3.u64 = kErrorNotFound; }
IMPORT(XamLoaderGetLaunchData) { ctx.r3.u64 = kErrorNotFound; }
IMPORT(XamLoaderSetLaunchData) { ctx.r3.u64 = kErrorSuccess; }
IMPORT(XamLoaderLaunchTitle) { fatal("xam: the title launched another title\n"); }
IMPORT(XamLoaderTerminateTitle) { fatal("xam: the title exited\n"); }

// One profile, signed in locally (offline) as user 0: the original Xbox has
// no sign-in screen to show, and the title wants a profile to play with.
constexpr uint64_t kXuid = 0xE000000000000001ull;  // an offline XUID
constexpr uint32_t kSignedInLocally = 1;

IMPORT(XamUserGetSigninState) { ctx.r3.u64 = ctx.r3.u32 == 0 ? kSignedInLocally : 0; }

IMPORT(XamUserGetXUID) {
    // (user, type mask, XUID*)
    if (ctx.r3.u32 != 0) {
        ctx.r3.u64 = kErrorNoSuchUser;
        return;
    }
    guest::store64(ctx.r5.u32, kXuid);
    ctx.r3.u64 = kErrorSuccess;
}

IMPORT(XamUserGetName) {
    // (user, buffer, size)
    if (ctx.r3.u32 != 0) {
        ctx.r3.u64 = kErrorNoSuchUser;
        return;
    }
    static const char name[] = "Player";
    uint32_t n = ctx.r5.u32 < sizeof(name) ? ctx.r5.u32 : sizeof(name);
    memcpy(guest::host(ctx.r4.u32), name, n);
    if (n)
        guest::store8(ctx.r4.u32 + n - 1, 0);
    ctx.r3.u64 = kErrorSuccess;
}

IMPORT(XamUserCheckPrivilege) {
    // (user, privilege, result*): no Live privileges offline
    guest::store32(ctx.r5.u32, 0);
    ctx.r3.u64 = ctx.r3.u32 == 0 ? kErrorSuccess : kErrorNoSuchUser;
}

IMPORT(XamUserAreUsersFriends) {
    // (user, XUIDs*, count, result*, overlapped)
    if (ctx.r6.u32)
        guest::store32(ctx.r6.u32, 0);
    ctx.r3.u64 = complete(ctx.r7.u32, kErrorSuccess);
}

IMPORT(XamUserReadProfileSettings) {
    // (title, user, XUID count, XUIDs*, setting count, setting ids*, buffer
    // size*, buffer, overlapped): every setting at its default, the title's
    // own (where saves live) not set yet.
    uint32_t user = ctx.r4.u32, count = ctx.r7.u32, ids = ctx.r8.u32, size_pointer = ctx.r9.u32;
    uint32_t buffer = ctx.r10.u32, overlapped = kernel::arg(ctx, 8);
    if (count < 1 || count > 32 || !size_pointer) {
        ctx.r3.u64 = 0x57;  // ERROR_INVALID_PARAMETER
        return;
    }
    // X_USER_READ_PROFILE_SETTINGS, then a 40-byte X_USER_PROFILE_SETTING
    // each, then the strings' and binaries' data.
    uint32_t needed = 8 + 40 * count;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t id = guest::load32(ids + i * 4);
        uint32_t type = id >> 28;
        if (type == 4 || type == 6)  // WSTRING, BINARY: their size is in the id
            needed += (id >> 16) & 0xFFF;
    }
    uint32_t size = guest::load32(size_pointer);
    if (!buffer || size < needed) {
        if (!size)
            guest::store32(size_pointer, needed);
        ctx.r3.u64 = 0x7A;  // ERROR_INSUFFICIENT_BUFFER
        return;
    }
    if (user != 0) {
        ctx.r3.u64 = complete(overlapped, kErrorFunctionFailed, 0x80070525);
        return;
    }
    memset(guest::host(buffer), 0, needed);
    guest::store32(buffer, count);
    guest::store32(buffer + 4, buffer + 8);
    uint32_t data = buffer + 8 + 40 * count;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t id = guest::load32(ids + i * 4);
        uint32_t s = buffer + 8 + 40 * i;
        uint32_t type = id >> 28, data_size = (id >> 16) & 0xFFF;
        bool title_specific = (id & 0xFFFF) >= 0x3FFD;  // XPROFILE_TITLE_SPECIFIC1..3
        guest::store32(s, title_specific ? 0 : 1);       // source: not set / default
        guest::store32(s + 8, 0xFFFFFFFF);               // XUID: none, the user index then
        guest::store32(s + 12, user);
        guest::store32(s + 16, id);
        guest::store8(s + 24, uint8_t(type));
        if (type == 4 || type == 6) {
            guest::store32(s + 32, title_specific ? 0 : data_size);
            guest::store32(s + 36, data);
            data += data_size;
        }
    }
    ctx.r3.u64 = complete(overlapped, kErrorSuccess);
}

IMPORT(XamUserWriteProfileSettings) {
    // (title, user, count, settings*, overlapped): accepted, not kept yet
    ctx.r3.u64 = complete(ctx.r7.u32, kErrorSuccess);
}
IMPORT(XamUserCreateAchievementEnumerator) { ctx.r3.u64 = kErrorNoSuchUser; }
IMPORT(XamUserCreateStatsEnumerator) { ctx.r3.u64 = kErrorNoSuchUser; }
IMPORT(XamWriteGamerTile) { ctx.r3.u64 = kErrorNoSuchUser; }

IMPORT(XamEnumerate) { ctx.r3.u64 = complete(ctx.r8.u32, kErrorNoMoreFiles); }

// Notifications: each listener hears once that user 0 signed in, as on a
// console that just signed someone in; nothing after that.
namespace {
constexpr uint32_t kSigninChanged = 0x0000000A;  // XN_SYS_SIGNINCHANGED
std::map<uint32_t, bool> told;  // listener -> sign-in already reported
}  // namespace

IMPORT(XamNotifyCreateListener) {
    HANDLE h = CreateEvent(nullptr, TRUE, TRUE, nullptr);  // signalled: something to read
    uint32_t handle = kernel::add_handle(h);
    told[handle] = false;
    ctx.r3.u64 = handle;
}

IMPORT(XNotifyGetNext) {
    // (listener, id to match, id*, parameter*)
    uint32_t listener = ctx.r3.u32, match = ctx.r4.u32;
    auto it = told.find(listener);
    if (it != told.end() && !it->second && (!match || match == kSigninChanged)) {
        it->second = true;
        guest::store32(ctx.r5.u32, kSigninChanged);
        if (ctx.r6.u32)
            guest::store32(ctx.r6.u32, 1);  // signed-in users: user 0
        ResetEvent(kernel::host_handle(listener));
        ctx.r3.u64 = 1;
        return;
    }
    ctx.r3.u64 = 0;
}

// The system UI is not there: cancelled.
IMPORT(XamShowSigninUI) { ctx.r3.u64 = kErrorCancelled; }
IMPORT(XamShowAchievementsUI) { ctx.r3.u64 = kErrorCancelled; }
IMPORT(XamShowDeviceSelectorUI) { ctx.r3.u64 = complete(ctx.r8.u32, kErrorCancelled); }
IMPORT(XamShowGamerCardUIForXUID) { ctx.r3.u64 = kErrorCancelled; }
IMPORT(XamShowMarketplaceUI) { ctx.r3.u64 = kErrorCancelled; }
IMPORT(XamShowMessageBoxUIEx) { ctx.r3.u64 = kErrorCancelled; }
IMPORT(XamShowDirtyDiscErrorUI) { fatal("xam: the title reported a dirty disc\n"); }

IMPORT(XamContentGetLicenseMask) {
    // (mask*, overlapped): the full game.
    guest::store32(ctx.r3.u32, 0xFFFFFFFF);
    ctx.r3.u64 = complete(ctx.r4.u32, kErrorSuccess);
}

// Voice: no headset, no voice (Xenia's answers).
IMPORT(XamVoiceCreate) {
    // (user, packets, voice*)
    guest::store32(ctx.r5.u32, 0);
    ctx.r3.u64 = 5;  // ERROR_ACCESS_DENIED
}

IMPORT(XamVoiceClose) { ctx.r3.u64 = 0; }
IMPORT(XamVoiceHeadsetPresent) { ctx.r3.u64 = 0; }
IMPORT(XamVoiceSubmitPacket) { ctx.r3.u64 = 0x800700AA; }  // busy

// Sessions (XSessionCreate and the rest, inside the title): a handle to a
// zeroed block the title may keep state in. Single player makes one too.
IMPORT(XamSessionCreateHandle) {
    // (handle*)
    uint32_t handle = kernel::add_handle(CreateEvent(nullptr, TRUE, FALSE, nullptr));
    guest::store32(ctx.r3.u32, handle);
    ctx.r3.u64 = kErrorSuccess;
}

IMPORT(XamSessionRefObjByHandle) {
    // (handle, object*)
    static std::map<uint32_t, uint32_t> objects;
    uint32_t& object = objects[ctx.r3.u32];
    if (!object)
        object = guest::alloc(0x100);
    guest::store32(ctx.r4.u32, object);
    ctx.r3.u64 = kErrorSuccess;
}

// Messages to XAM's apps: XGI (0xFB: sessions, contexts, properties,
// achievements, leaderboards) and XMP (0xFA: the music player, idle) are
// answered, all succeeding offline, as Xenia does.
namespace {
constexpr uint32_t kSOk = 0, kEFail = 0x80004005;

uint32_t dispatch_message(uint32_t app, uint32_t message, uint32_t buffer, uint32_t length);

uint32_t dispatch(uint32_t app, uint32_t message, uint32_t buffer, uint32_t length) {
    uint32_t result = dispatch_message(app, message, buffer, length);
    if (app == 0xFA)  // XMP: polled every frame
        return result;
    uint32_t w[8] = {};
    for (uint32_t i = 0; buffer && i < 8; ++i)
        w[i] = guest::load32(buffer + i * 4);
    log("xam: message %08X to app %02X (buffer %08X, %u bytes) -> %08X: %08X %08X %08X %08X %08X %08X %08X %08X\n",
        unsigned(message), unsigned(app), unsigned(buffer), unsigned(length), unsigned(result), unsigned(w[0]),
        unsigned(w[1]), unsigned(w[2]), unsigned(w[3]), unsigned(w[4]), unsigned(w[5]), unsigned(w[6]),
        unsigned(w[7]));
    return result;
}

uint32_t dispatch_message(uint32_t app, uint32_t message, uint32_t buffer, uint32_t length) {
    if (app == 0xFB) {
        switch (message) {
        case 0x000B0041: {  // XUserGetContext: none set
            uint32_t context = guest::load32(buffer + 16);
            if (context)
                guest::store32(context + 4, 0);
            return kEFail;
        }
        case 0x000B0006:  // XUserSetContext
        case 0x000B0007:  // XUserSetProperty
        case 0x000B0008:  // XUserWriteAchievements
        case 0x000B0010:  // XSessionCreate
        case 0x000B0011:  // XSessionDelete
        case 0x000B0012:  // XSessionJoinLocal
        case 0x000B0013:  // XSessionLeaveLocal
        case 0x000B0014:  // XSessionStart
        case 0x000B0015:  // XSessionEnd
        case 0x000B0016:  // XSessionWriteStats
        case 0x000B0019:  // XSessionFlushStats
        case 0x000B0071:
            return kSOk;
        }
    }
    if (app == 0xFA && (message >> 16) == 0x0007) {  // XMP: no user music
        switch (message) {
        case 0x00070009:  // XMPGetStatus: idle
            guest::store32(guest::load32(buffer + 4), 0);
            return kSOk;
        case 0x0007000B:  // XMPGetVolume: full
            guest::store32(guest::load32(buffer + 4), 0x3F800000);
            return kSOk;
        }
        return kSOk;
    }
    if (app == 0xFC) {  // XLiveBase
        switch (message) {
        case 0x00058004:  // logon id
        case 0x00058006:  // NAT type: open
            guest::store32(buffer, 1);
            return kSOk;
        case 0x00058007:  // service info
            return 0x80151802;  // connection invalid
        case 0x00058046:
            return kSOk;
        }
    }
    log("xam: message %08X to app %08X (buffer %08X, %u bytes) not handled\n", unsigned(message), unsigned(app),
        unsigned(buffer), unsigned(length));
    return kEFail;
}
}  // namespace

IMPORT(XMsgInProcessCall) {
    // (app, message, argument 1, argument 2)
    ctx.r3.u64 = dispatch(ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, ctx.r6.u32);
}

IMPORT(XMsgStartIORequest) {
    // (app, message, XOVERLAPPED*, buffer, length)
    uint32_t result = dispatch(ctx.r3.u32, ctx.r4.u32, ctx.r6.u32, ctx.r7.u32);
    // Without an XOVERLAPPED the title's XSession* calls take
    // GetLastError() (the KTHREAD's, at PCR+0x100 -> +0x160) as their
    // result: clear what earlier calls left there.
    if (result == kSOk)
        guest::store32(guest::load32(ctx.r13.u32 + 0x100) + 0x160, 0);
    ctx.r3.u64 = complete(ctx.r5.u32, result);
}

IMPORT(XMsgCancelIORequest) { ctx.r3.u64 = kErrorSuccess; }
