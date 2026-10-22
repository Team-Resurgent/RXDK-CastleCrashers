// The Xbox 360 kernel and XAM as the title's imports see them.
//
// IMPORT(Name) defines the import the generated code calls as __imp__Name:
// its arguments are in ctx.r3..r10 (then on the guest stack), its result goes
// in r3. Imports nothing here defines get a logging stub (tools/gen_stubs.py).
#pragma once

#include "guest.h"

// Each records itself on the calling guest thread for the watchdog.
#define IMPORT(name)                                                       \
    static void import_##name(PPCContext& __restrict ctx, PPCBase base);  \
    PPC_FUNC(__imp__##name) {                                              \
        kernel::ImportScope scope(#name, __builtin_return_address(0));    \
        import_##name(ctx, base);                                          \
    }                                                                      \
    static void import_##name([[maybe_unused]] PPCContext& __restrict ctx, [[maybe_unused]] PPCBase base)

// NTSTATUS values the title tests.
enum : uint32_t {
    kSuccess = 0,
    kAbandoned = 0x80,
    kUserApc = 0xC0,
    kAlerted = 0x101,
    kTimeout = 0x102,
    kPending = 0x103,
    kObjectNameExists = 0x40000000,
    kNoMoreFiles = 0x80000006,
    kUnsuccessful = 0xC0000001,
    kNotImplemented = 0xC0000002,
    kInvalidHandle = 0xC0000008,
    kInvalidParameter = 0xC000000D,
    kEndOfFile = 0xC0000011,
    kNoMemory = 0xC0000017,
    kObjectTypeMismatch = 0xC0000024,
    kObjectNameNotFound = 0xC0000034,
    kObjectPathNotFound = 0xC000003A,
    kNotFound = 0xC0000225,
};

// Dispatcher object types (DISPATCHER_HEADER.Type).
enum : uint8_t {
    kNotificationEvent = 0,
    kSynchronizationEvent = 1,
    kMutant = 2,
    kSemaphore = 5,
    kThread = 6,
    kNotificationTimer = 8,
    kSynchronizationTimer = 9,
};

namespace kernel {

// The runtime's own host threads run above every guest thread (kernel.cpp's
// host_priority).
constexpr int kRuntimePriority = THREAD_PRIORITY_HIGHEST;
// The host stack of a thread running guest code: the recompiled functions'
// frames, as deep as the guest's calls. Xbox stacks are committed memory:
// Castle Crashers' threads were seen using at most 6 KB (the watchdog logs
// it), so 64 KB.
constexpr uint32_t kGuestHostStack = 0x10000;
// A runtime thread at kRuntimePriority (fatal if it cannot start).
HANDLE start_runtime_thread(LPTHREAD_START_ROUTINE routine, void* parameter, uint32_t stack_size);
// Cycles (rdtsc) the CPU has had nothing else to run, so far.
uint64_t idle_cycles();

// Logs import calls while set (all but the most frequent): what a title
// asked for just before it went wrong.
extern volatile bool trace_imports;
void trace_import(const char* name, void* caller);

struct ImportScope {
    GuestThread* thread;
    ImportScope(const char* name, void* caller) : thread(current_thread()) {
        if (thread) {
            thread->import = name;
            thread->import_caller = caller;
            thread->in_import = true;
            uintptr_t sp = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
            if (sp < thread->host_stack_low)
                thread->host_stack_low = sp;
        }
        if (trace_imports)
            trace_import(name, caller);
    }
    ~ImportScope() {
        if (thread)
            thread->in_import = false;
    }
};

// A 64-bit guest argument: r3..r10, then the caller's parameter save area
// (eight bytes each from r1 + 0x54).
inline uint64_t arg64(PPCContext& ctx, int index) {
    switch (index) {
    case 0: return ctx.r3.u64;
    case 1: return ctx.r4.u64;
    case 2: return ctx.r5.u64;
    case 3: return ctx.r6.u64;
    case 4: return ctx.r7.u64;
    case 5: return ctx.r8.u64;
    case 6: return ctx.r9.u64;
    case 7: return ctx.r10.u64;
    default: return guest::load64(ctx.r1.u32 + 0x54 + (index - 8) * 8);
    }
}

// A 32-bit guest argument: on the stack, the first four bytes of its slot.
inline uint32_t arg(PPCContext& ctx, int index) {
    return index < 8 ? uint32_t(arg64(ctx, index)) : guest::load32(ctx.r1.u32 + 0x54 + (index - 8) * 8);
}

// The host handle behind a guest handle, a guest object, or 0.
HANDLE host_handle(uint32_t guest_handle);
HANDLE host_object(uint32_t guest_object);
// A guest handle for a new host handle.
uint32_t add_handle(HANDLE handle);

// Waits: guest timeouts are 100 ns units, negative relative, 0 polls.
DWORD timeout_ms(uint32_t guest_timeout_pointer);
uint32_t wait_status(DWORD result);

// image_header: the title's XEX header, copied into guest memory.
void init(uint32_t image_header);
// The variable a variable import's record points at (0 if unknown).
uint32_t import_variable(bool kernel_library, uint32_t ordinal);
// The main thread: what main() runs the entry point on.
GuestThread* main_thread();

// printf-style formatting of guest arguments (format.cpp): next(i) is the
// i-th 64-bit argument, wide formats a UTF-16 format string.
int format(char* out, int size, uint32_t format, bool wide, uint64_t (*next)(void*, int), void* state);

}  // namespace kernel

namespace xam {
void init();
}  // namespace xam

namespace audio {
// The XMA decoder's context array (audio.cpp).
void init();
}  // namespace audio
