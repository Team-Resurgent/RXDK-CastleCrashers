// xboxkrnl.exe: threads, dispatcher objects and waits, critical sections,
// TLS, memory, time and the loader's queries.
//
// Guest handles are the host's handles (both are 32-bit values). Dispatcher
// objects the title builds in its own memory (KEVENT, KSEMAPHORE, KTHREAD)
// get a host object the first time a kernel call names them; handles the
// title asks to see as objects (ObReferenceObjectByHandle) get a guest
// DISPATCHER_HEADER standing for them.
#include "kernel.h"

#include <algorithm>
#include <map>
#include <vector>

#ifndef MAXIMUM_WAIT_OBJECTS
#define MAXIMUM_WAIT_OBJECTS 64
#endif

namespace kernel {
namespace {

CRITICAL_SECTION lock;
struct Lock {
    Lock() { EnterCriticalSection(&lock); }
    ~Lock() { LeaveCriticalSection(&lock); }
};

std::map<uint32_t, HANDLE> object_handles;  // guest object -> host handle
std::map<HANDLE, uint32_t> handle_objects;  // host handle -> guest object
std::map<HANDLE, int> duplicates;           // extra references NtDuplicateObject made
std::map<uint32_t, CRITICAL_SECTION*> rw_locks;

uint32_t tls_used[2];  // KeTlsAlloc's 64 slots
__thread GuestThread* this_thread;
GuestThread* first_thread;

// Kernel variables the title imports (PPC_IMPORT records with no thunk).
uint32_t xbox_krnl_version, time_stamp_bundle, loaded_command_line, executable_module, debug_monitor_data,
    cert_monitor_data, vd_global_device, vd_global_xam_device, vd_gpu_clock, vd_hsio_lock, event_object_type,
    thread_object_type, process, hardware_info;
uint32_t xex_header;  // the title's XEX header, in guest memory

uint8_t object_type(uint32_t object) { return guest::load8(object); }
int32_t signal_state(uint32_t object) { return int32_t(guest::load32(object + 4)); }

void write_header(uint32_t object, uint8_t type, uint8_t size_in_words, int32_t signal) {
    guest::store8(object, type);
    guest::store8(object + 1, 0);
    guest::store8(object + 2, size_in_words);
    guest::store8(object + 3, 0);
    guest::store32(object + 4, uint32_t(signal));
    // An empty wait list: both links point at the list head.
    guest::store32(object + 8, object + 8);
    guest::store32(object + 12, object + 8);
}

void bind(uint32_t object, HANDLE handle) {
    Lock held;
    object_handles[object] = handle;
    handle_objects[handle] = object;
}

DWORD WINAPI time_stamp_thread(void*) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    for (;;) {
        // InterruptTime and SystemTime in 100 ns units, TickCount in ms.
        LARGE_INTEGER count;
        QueryPerformanceCounter(&count);
        uint64_t ticks = uint64_t(count.QuadPart) / uint64_t(frequency.QuadPart) * 10000000ull +
                         uint64_t(count.QuadPart) % uint64_t(frequency.QuadPart) * 10000000ull / uint64_t(frequency.QuadPart);
        FILETIME now;
        GetSystemTimeAsFileTime(&now);
        guest::store64(time_stamp_bundle, ticks);
        guest::store64(time_stamp_bundle + 8, (uint64_t(now.dwHighDateTime) << 32) | now.dwLowDateTime);
        guest::store32(time_stamp_bundle + 16, GetTickCount());
        Sleep(1);
    }
}

// The CPU nothing else wants: a lowest-priority thread counts the cycles it
// runs (gaps of more than ~10 us between its readings are other threads').
volatile uint64_t idle_total;

DWORD WINAPI idle_thread(void*) {
    uint64_t last = __builtin_ia32_rdtsc();
    for (;;) {
        uint64_t now = __builtin_ia32_rdtsc();
        if (now - last < 8000)
            idle_total = idle_total + (now - last);
        last = now;
    }
}

GuestThread* all_threads;

// Every few seconds, where each guest thread is: a hung title's threads
// sit in the import they last called.
DWORD WINAPI watchdog_thread(void*) {
    for (;;) {
        Sleep(5000);
        // The console's memory, by the kernel's categories (pages, but the
        // virtual-memory byte count).
        MM_STATISTICS mm = {};
        mm.Length = sizeof(mm);
        MmQueryStatistics(&mm);
        log("watchdog: %u s up, %u KB free, %u KB of guest memory committed, %u KB of it touched; "
            "console: virtual %u KB, image %u KB, stacks %u KB, pool %u KB, cache %u KB\n",
            unsigned(GetTickCount() / 1000), unsigned(mm.AvailablePages * 4), guest::committed_bytes() >> 10,
            guest::resident_bytes() >> 10, unsigned(mm.VirtualMemoryBytesCommitted >> 10),
            unsigned(mm.ImagePagesCommitted * 4), unsigned(mm.StackPagesCommitted * 4),
            unsigned(mm.PoolPagesCommitted * 4), unsigned(mm.CachePagesCommitted * 4));
        // The threads every minute (each line is serial time).
        static unsigned rounds;
        if (rounds % 6 == 5)  // every 30 s: where the memory is
            guest::log_blocks(14);
        if (++rounds % 12 == 5) {
            // Every minute from 25 s: what each thread has been calling (a
            // title spinning on something that never comes; call trace
            // builds only).
            for (GuestThread* t = all_threads; t; t = t->next) {
                dump_calls(t, 40);
                profile_calls(t, 8);
            }
        }
        if (rounds % 12 != 1)
            continue;
        for (GuestThread* t = all_threads; t; t = t->next) {
            if (t->import)
                log("watchdog: thread %u (start %08X) %s %s, called from host %p, %u KB of host stack\n", t->id,
                    t->start, t->in_import ? "in" : "last called", t->import, t->import_caller,
                    t->host_stack_top && t->host_stack_low < t->host_stack_top
                        ? unsigned((t->host_stack_top - t->host_stack_low) >> 10)
                        : 0u);
        }
    }
}

// A guest thread that ended, in its KTHREAD: the title's GetExitCodeThread
// reads the object itself (signalled, then the exit status at +0x140), and
// waits for a loader thread that way before its intro moves on.
void mark_exited(GuestThread* t, uint32_t status) {
    if (!t)
        return;
    guest::store32(t->kthread + 0x140, status);
    guest::store32(t->kthread + 4, 1);  // DISPATCHER_HEADER.SignalState
}

DWORD WINAPI thread_entry(void* parameter) {
    guest::GuestFaults faults;
    GuestThread* thread = static_cast<GuestThread*>(parameter);
    thread->host_stack_top = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
    this_thread = thread;
    PPCContext& ctx = thread->ctx;
    ctx.fpscr.csr = ctx.fpscr.getcsr();
    if (thread->xapi_startup) {
        ctx.r3.u64 = thread->start;
        ctx.r4.u64 = thread->start_context;
        guest::call(ctx, thread->xapi_startup);
    } else {
        ctx.r3.u64 = thread->start_context;
        guest::call(ctx, thread->start);
    }
    mark_exited(thread, ctx.r3.u32);
    return DWORD(ctx.r3.u32);
}

uint32_t find_xex_header(uint32_t key) {
    uint32_t count = guest::load32(xex_header + 0x14);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t entry = xex_header + 0x18 + i * 8;
        if (guest::load32(entry) == key)
            return entry;
    }
    return 0;
}

// A guest object standing for a handle (ObReferenceObjectByHandle).
uint32_t object_for(HANDLE handle, uint8_t type) {
    {
        Lock held;
        auto it = handle_objects.find(handle);
        if (it != handle_objects.end())
            return it->second;
    }
    uint32_t object = guest::alloc(guest::kPage);
    write_header(object, type, 4, 0);
    bind(object, handle);
    return object;
}

void close(HANDLE handle) {
    {
        Lock held;
        auto d = duplicates.find(handle);
        if (d != duplicates.end()) {
            if (--d->second == 0)
                duplicates.erase(d);
            return;
        }
        auto it = handle_objects.find(handle);
        if (it != handle_objects.end()) {
            object_handles.erase(it->second);
            handle_objects.erase(it);
        }
    }
    CloseHandle(handle);
}

// The host has one processor where the 360 has six hardware threads, and the
// title's threads spin waiting for each other (XAudio for its audio thread):
// a spinning thread above the others would starve them. So all of them run
// at one priority, where time slices take turns, under the runtime's own
// threads (kRuntimePriority: the clock, the GPU, vblank, audio), which
// mostly sleep.
// The title's threads all run at one priority (a thread that spins at a
// higher one starves the rest on one processor), except the ones that ask
// for the 360's highest (15 and up): XAudio's workers, which the audio
// thread (audio.cpp) waits for every 5.3 ms.
int host_priority(int priority) { return priority >= 15 ? THREAD_PRIORITY_ABOVE_NORMAL : THREAD_PRIORITY_NORMAL; }

}  // namespace

HANDLE host_handle(uint32_t guest_handle) {
    if (guest_handle == 0xFFFFFFFE)  // NtCurrentThread()
        return this_thread->handle;
    return reinterpret_cast<HANDLE>(guest_handle);
}

uint32_t add_handle(HANDLE handle) { return uint32_t(reinterpret_cast<uintptr_t>(handle)); }

HANDLE host_object(uint32_t object) {
    {
        Lock held;
        auto it = object_handles.find(object);
        if (it != object_handles.end())
            return it->second;
    }
    HANDLE handle;
    switch (object_type(object)) {
    case kNotificationEvent:
    case kSynchronizationEvent:
        handle = CreateEvent(nullptr, object_type(object) == kNotificationEvent, signal_state(object) != 0, nullptr);
        break;
    case kSemaphore:
        handle = CreateSemaphore(nullptr, signal_state(object), int32_t(guest::load32(object + 16)), nullptr);
        break;
    case kMutant:
        handle = CreateMutex(nullptr, FALSE, nullptr);
        break;
    default:
        fatal("kernel: object %08X of type %u used before the kernel knew it\n", object, object_type(object));
    }
    bind(object, handle);
    return handle;
}

DWORD timeout_ms(uint32_t pointer) {
    if (!pointer)
        return INFINITE;
    int64_t t = int64_t(guest::load64(pointer));
    if (t == 0)
        return 0;
    if (t < 0)
        return DWORD((-t + 9999) / 10000);
    // An absolute time: from the system time now.
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    int64_t until = t - int64_t((uint64_t(now.dwHighDateTime) << 32) | now.dwLowDateTime);
    return until <= 0 ? 0 : DWORD((until + 9999) / 10000);
}

uint32_t wait_status(DWORD result) {
    if (result == WAIT_TIMEOUT)
        return kTimeout;
    if (result == WAIT_IO_COMPLETION)
        return kUserApc;
    if (result >= WAIT_ABANDONED_0 && result < WAIT_ABANDONED_0 + MAXIMUM_WAIT_OBJECTS)
        return kAbandoned + (result - WAIT_ABANDONED_0);
    if (result == WAIT_FAILED)
        return kUnsuccessful;
    return result - WAIT_OBJECT_0;
}

GuestThread* main_thread() { return first_thread; }

volatile bool trace_imports;

void trace_import(const char* name, void* caller) {
    static const char* quiet[] = {"RtlEnterCriticalSection", "RtlLeaveCriticalSection", "RtlTryEnterCriticalSection",
                                  "KeTlsGetValue", "KeTlsSetValue", "KeRaiseIrqlToDpcLevel", "KfLowerIrql",
                                  "KfAcquireSpinLock", "KfReleaseSpinLock", "KeAcquireSpinLockAtRaisedIrql",
                                  "KeReleaseSpinLockFromRaisedIrql", "XAudioSubmitRenderDriverFrame",
                                  "XAudioGetVoiceCategoryVolumeChangeMask"};
    for (const char* q : quiet)
        if (strcmp(name, q) == 0)
            return;
    GuestThread* t = current_thread();
    log("trace: thread %u %s from host %p\n", t ? t->id : 0, name, caller);
}

uint64_t idle_cycles() { return idle_total; }

namespace {
// A runtime thread's routine, run under a GuestFaults frame (they read and
// write guest memory).
struct RuntimeThread {
    LPTHREAD_START_ROUTINE routine;
    void* parameter;
};

DWORD WINAPI runtime_thread_entry(void* parameter) {
    guest::GuestFaults faults;
    RuntimeThread start = *static_cast<RuntimeThread*>(parameter);
    delete static_cast<RuntimeThread*>(parameter);
    return start.routine(start.parameter);
}
}  // namespace

HANDLE start_runtime_thread(LPTHREAD_START_ROUTINE routine, void* parameter, uint32_t stack_size) {
    HANDLE h = CreateThread(nullptr, stack_size, runtime_thread_entry, new RuntimeThread{routine, parameter},
                            CREATE_SUSPENDED, nullptr);
    if (!h)
        fatal("kernel: cannot start a runtime thread\n");
    SetThreadPriority(h, kRuntimePriority);
    ResumeThread(h);
    return h;
}

void init(uint32_t image_header) {
    InitializeCriticalSection(&lock);
    xex_header = image_header;

    uint32_t vars = guest::alloc(0x2000);
    uint32_t next = vars;
    auto var = [&](uint32_t size) {
        uint32_t at = next;
        next += (size + 15) & ~15u;
        return at;
    };
    xbox_krnl_version = var(8);
    guest::store16(xbox_krnl_version, 2);
    guest::store16(xbox_krnl_version + 4, 17559);
    time_stamp_bundle = var(24);
    loaded_command_line = var(16);
    memcpy(guest::host(loaded_command_line), "default.xex", 12);
    // XexExecutableModuleHandle points at the module's LDR_DATA_TABLE_ENTRY,
    // whose XexHeaderBase (+0x58) RtlImageXexHeaderField is given.
    uint32_t module = var(0x80);
    guest::store32(module + 0x58, xex_header);
    executable_module = var(4);
    guest::store32(executable_module, module);
    debug_monitor_data = var(0x40);
    cert_monitor_data = var(0x40);
    vd_global_device = var(4);
    vd_global_xam_device = var(4);
    // XboxHardwareInfo: flags, then the number of hardware threads. Castle
    // Crashers starts a worker, each with its own 6 MB heap, per hardware
    // thread but one: two here (the original Xbox has one processor).
    hardware_info = var(16);
    guest::store8(hardware_info + 4, 2);
    vd_gpu_clock = var(4);
    guest::store32(vd_gpu_clock, 500);
    vd_hsio_lock = var(0x20);
    event_object_type = var(0x40);
    thread_object_type = var(0x40);
    process = var(0x60);
    guest::store8(process + 0x2F, 1);  // X_PROCTYPE_TITLE

    start_runtime_thread(time_stamp_thread, nullptr, 0x10000);
    start_runtime_thread(watchdog_thread, nullptr, 0x10000);
    // Lowered before it first runs: spinning at runtime priority it would
    // never give this thread back.
    HANDLE idle = CreateThread(nullptr, 0x4000, idle_thread, nullptr, CREATE_SUSPENDED, nullptr);
    if (idle) {
        SetThreadPriority(idle, THREAD_PRIORITY_IDLE);
        ResumeThread(idle);
    }
    first_thread = create_thread(0x40000, 0, 0, 0);
}

uint32_t import_variable(bool kernel_library, uint32_t ordinal) {
    if (!kernel_library)
        return 0;
    switch (ordinal) {
    case 0x00E: return event_object_type;
    case 0x01B: return thread_object_type;
    case 0x059: return debug_monitor_data;
    case 0x0AD: return time_stamp_bundle;
    case 0x156: return hardware_info;
    case 0x158: return xbox_krnl_version;
    case 0x193: return executable_module;
    case 0x1AE: return loaded_command_line;
    case 0x1BE: return vd_global_device;
    case 0x1BF: return vd_global_xam_device;
    case 0x1C0: return vd_gpu_clock;
    case 0x1C1: return vd_hsio_lock;
    case 0x266: return cert_monitor_data;
    }
    return 0;
}

}  // namespace kernel

using namespace kernel;

GuestThread* current_thread() { return this_thread; }

void set_current_thread(GuestThread* thread) { this_thread = thread; }

GuestThread* guest_thread_state(uint32_t stack_size, uint32_t start) {
    static volatile LONG next_id = 0;
    GuestThread* t = new GuestThread;
    {
        Lock held;
        t->next = all_threads;
        all_threads = t;
    }
    t->stack_size = stack_size < 0x10000 ? 0x10000 : (stack_size + 0xFFF) & ~0xFFFu;
    t->stack = guest::alloc(t->stack_size);
    t->pcr = guest::alloc(0x1000);
    t->kthread = guest::alloc(0x1000);
    if (!t->stack || !t->pcr || !t->kthread)
        fatal("kernel: no guest memory for a thread (%u KB stack)\n", t->stack_size >> 10);
    t->id = uint32_t(InterlockedIncrement(&next_id));
    t->start = start;

    uint32_t top = t->stack + t->stack_size;
    write_header(t->kthread, kThread, 0, 0);
    guest::store32(t->kthread + 0x5C, top);
    guest::store32(t->kthread + 0x60, t->stack);
    guest::store32(t->kthread + 0x84, process);
    guest::store32(t->kthread + 0x14C, t->id);
    guest::store32(t->kthread + 0x150, start);
    guest::store32(t->pcr + 0x00, t->pcr + 0x800);  // a TLS block in the PCR's page
    guest::store32(t->pcr + 0x70, top);
    guest::store32(t->pcr + 0x74, t->stack);
    guest::store32(t->pcr + 0x100, t->kthread);  // PRCB.CurrentThread
    guest::store32(t->pcr + 0x2A8, t->pcr + 0x100);

    t->ctx.r1.u64 = top - 0x100;
    t->ctx.r13.u64 = t->pcr;
#ifdef PPC_CALL_TRACE
    t->ctx.trace = new const void*[PPC_CALL_TRACE_SIZE]();
#endif
    return t;
}

void dump_calls(GuestThread* t, uint32_t count) {
#ifdef PPC_CALL_TRACE
    log("calls: thread %u, last %u of %u (oldest first)\n", t->id, count, t->ctx.trace_next);
    uint32_t end = t->ctx.trace_next, start = end > count ? end - count : 0;
    // Host addresses inside each function: symbolize them off the console.
    const void* previous = nullptr;
    uint32_t repeats = 0;
    for (uint32_t i = start; i < end; i++) {
        const void* at = t->ctx.trace[i & (PPC_CALL_TRACE_SIZE - 1)];
        if (at == previous) {
            repeats++;
            continue;
        }
        if (repeats)
            log("calls:   (x%u)\n", repeats + 1);
        repeats = 0;
        log("calls: %p\n", at);
        previous = at;
    }
    if (repeats)
        log("calls:   (x%u)\n", repeats + 1);
#else
    (void)t;
    (void)count;
#endif
}

void profile_calls(GuestThread* t, uint32_t top) {
#ifdef PPC_CALL_TRACE
    // The functions entered most often in the whole ring: where a busy
    // thread spends its calls.
    uint32_t end = t->ctx.trace_next, start = end > PPC_CALL_TRACE_SIZE ? end - PPC_CALL_TRACE_SIZE : 0;
    std::map<const void*, uint32_t> counts;
    for (uint32_t i = start; i < end; i++)
        counts[t->ctx.trace[i & (PPC_CALL_TRACE_SIZE - 1)]]++;
    std::vector<std::pair<uint32_t, const void*>> sorted;
    for (auto& [at, n] : counts)
        sorted.push_back({n, at});
    std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first > b.first; });
    log("calls: thread %u profile of %u calls\n", t->id, end - start);
    for (uint32_t i = 0; i < top && i < sorted.size(); i++)
        log("calls: %p\ncalls:   (x%u)\n", sorted[i].second, sorted[i].first);
#else
    (void)t;
    (void)top;
#endif
}

GuestThread* create_thread(uint32_t stack_size, uint32_t xapi_startup, uint32_t start, uint32_t context) {
    GuestThread* t = guest_thread_state(stack_size, start);
    t->xapi_startup = xapi_startup;
    t->start_context = context;
    DWORD host_id;
    // Recompiled functions keep the guest's registers in host locals: a deep
    // guest call chain needs a deep host stack.
    t->handle = CreateThread(nullptr, kGuestHostStack, thread_entry, t, CREATE_SUSPENDED, &host_id);
    if (!t->handle)
        fatal("kernel: CreateThread failed\n");
    bind(t->kthread, t->handle);
    return t;
}

// --- threads -----------------------------------------------------------------

static void set_cpu(GuestThread* t, uint32_t affinity);

IMPORT(ExCreateThread) {
    // (handle*, stack size, thread id*, XAPI startup, start, context, flags)
    GuestThread* t = create_thread(ctx.r4.u32 ? ctx.r4.u32 : 0x40000, ctx.r6.u32, ctx.r7.u32, ctx.r8.u32);
    set_cpu(t, ctx.r9.u32 >> 24);  // the creation flags' processor mask
    if (ctx.r3.u32)
        guest::store32(ctx.r3.u32, add_handle(t->handle));
    if (ctx.r5.u32)
        guest::store32(ctx.r5.u32, t->id);
    log("kernel: thread %u at %08X (context %08X)\n", t->id, t->start, t->start_context);
    if (!(ctx.r9.u32 & 1))  // CREATE_SUSPENDED
        ResumeThread(t->handle);
    ctx.r3.u64 = kSuccess;
}

IMPORT(ExTerminateThread) {
    mark_exited(current_thread(), ctx.r3.u32);
    ExitThread(ctx.r3.u32);
}

IMPORT(KeGetCurrentProcessType) { ctx.r3.u64 = 1; }  // a title

// The 360 hardware thread a guest thread is on, as the guest sees it (PCR and
// KTHREAD): all run on the one host processor, but the title's code
// rendezvous by CPU number (XAudio waits for each of its CPUs to check in).
static void set_cpu(GuestThread* t, uint32_t affinity) {
    uint32_t cpu = affinity ? uint32_t(__builtin_ctz(affinity)) : 0;
    if (cpu > 5)
        cpu = 0;
    guest::store8(t->pcr + 0x10C, uint8_t(cpu));
    guest::store8(t->kthread + 0xBF, uint8_t(cpu));
}

IMPORT(KeSetAffinityThread) {
    // (KTHREAD, affinity, previous affinity*)
    if (!ctx.r4.u32) {
        ctx.r3.u64 = kInvalidParameter;
        return;
    }
    for (GuestThread* t = all_threads; t; t = t->next) {
        if (t->kthread == ctx.r3.u32) {
            if (ctx.r5.u32)
                guest::store32(ctx.r5.u32, 1u << guest::load8(t->pcr + 0x10C));
            set_cpu(t, ctx.r4.u32);
        }
    }
    ctx.r3.u64 = kSuccess;
}

IMPORT(KeSetBasePriorityThread) {
    HANDLE thread = host_object(ctx.r3.u32);
    int old = GetThreadPriority(thread);
    log("kernel: a thread asks for priority %d\n", int(ctx.r4.s32));
    SetThreadPriority(thread, host_priority(ctx.r4.s32));
    ctx.r3.u64 = uint64_t(int64_t(old));
}

IMPORT(KeResumeThread) { ctx.r3.u64 = ResumeThread(host_object(ctx.r3.u32)); }

IMPORT(NtResumeThread) {
    DWORD previous = ResumeThread(host_handle(ctx.r3.u32));
    if (ctx.r4.u32)
        guest::store32(ctx.r4.u32, previous);
    ctx.r3.u64 = previous == DWORD(-1) ? kInvalidHandle : kSuccess;
}

IMPORT(KeDelayExecutionThread) {
    // (mode, alertable, interval*)
    DWORD ms = timeout_ms(ctx.r5.u32);
    // A title polling something that never happens sleeps over and over:
    // name the recompiled function calling (from the host return address).
    static __thread uint32_t delays;
    if (++delays % 500 == 0)
        log("kernel: thread %u slept %u times (%u ms), called from host %p\n", current_thread()->id, delays, ms,
            __builtin_return_address(0));
    ctx.r3.u64 = SleepEx(ms, ctx.r4.u32 != 0) == WAIT_IO_COMPLETION ? kUserApc : kSuccess;
}

IMPORT(KeEnterCriticalRegion) {}
IMPORT(KeLeaveCriticalRegion) {}
IMPORT(KeRaiseIrqlToDpcLevel) { ctx.r3.u64 = 0; }
IMPORT(KfLowerIrql) {}
IMPORT(KeLockL2) {}
IMPORT(KeUnlockL2) {}

// --- spin locks: guest words, spinning politely on one processor ------------

static void spin_acquire(uint32_t lock_word) {
    uint32_t* word = reinterpret_cast<uint32_t*>(guest::host(lock_word));
    uint32_t me = __builtin_bswap32(current_thread()->kthread);
    // Sleep(0) runs only threads of the same priority or above: the owner
    // may be below (an interrupt callback waiting for a guest thread).
    for (int spins = 0; !__sync_bool_compare_and_swap(word, 0u, me); spins++)
        Sleep(spins < 16 ? 0 : 1);
}

static void spin_release(uint32_t lock_word) {
    __sync_lock_release(reinterpret_cast<uint32_t*>(guest::host(lock_word)));
}

IMPORT(KfAcquireSpinLock) { spin_acquire(ctx.r3.u32); ctx.r3.u64 = 0; }
IMPORT(KfReleaseSpinLock) { spin_release(ctx.r3.u32); }
IMPORT(KeAcquireSpinLockAtRaisedIrql) { spin_acquire(ctx.r3.u32); }
IMPORT(KeReleaseSpinLockFromRaisedIrql) { spin_release(ctx.r3.u32); }

// --- critical sections ---------------------------------------------------------
// RTL_CRITICAL_SECTION: header (16), LockCount (+0x10), RecursionCount
// (+0x14), OwningThread (+0x18: the owner's KTHREAD).

static void cs_init(uint32_t cs) {
    write_header(cs, kSynchronizationEvent, 4, 0);
    guest::store32(cs + 0x10, uint32_t(-1));
    guest::store32(cs + 0x14, 0);
    guest::store32(cs + 0x18, 0);
}

IMPORT(RtlInitializeCriticalSection) { cs_init(ctx.r3.u32); }
IMPORT(RtlInitializeCriticalSectionAndSpinCount) { cs_init(ctx.r3.u32); ctx.r3.u64 = kSuccess; }

static bool cs_try(uint32_t cs) {
    uint32_t me = current_thread()->kthread;
    uint32_t* owner = reinterpret_cast<uint32_t*>(guest::host(cs + 0x18));
    if (__builtin_bswap32(*owner) == me) {
        guest::store32(cs + 0x14, guest::load32(cs + 0x14) + 1);
        return true;
    }
    if (__sync_bool_compare_and_swap(owner, 0u, __builtin_bswap32(me))) {
        guest::store32(cs + 0x14, 1);
        guest::store32(cs + 0x10, 0);
        return true;
    }
    return false;
}

IMPORT(RtlEnterCriticalSection) {
    for (int spins = 0; !cs_try(ctx.r3.u32); spins++) {
        Sleep(spins < 16 ? 0 : 1);
        if (spins == 3000)
            log("kernel: thread %u waiting more than 3 s for critical section %08X (owner %08X)\n", current_thread()->id,
                ctx.r3.u32, guest::load32(ctx.r3.u32 + 0x18));
    }
}

IMPORT(RtlTryEnterCriticalSection) { ctx.r3.u64 = cs_try(ctx.r3.u32) ? 1 : 0; }

IMPORT(RtlLeaveCriticalSection) {
    uint32_t cs = ctx.r3.u32;
    uint32_t count = guest::load32(cs + 0x14) - 1;
    guest::store32(cs + 0x14, count);
    if (count == 0) {
        guest::store32(cs + 0x10, uint32_t(-1));
        __sync_lock_release(reinterpret_cast<uint32_t*>(guest::host(cs + 0x18)));
    }
}

// --- read/write locks: one host critical section each (shared = exclusive) ---

static CRITICAL_SECTION* rw_lock(uint32_t address) {
    Lock held;
    CRITICAL_SECTION*& cs = rw_locks[address];
    if (!cs) {
        cs = new CRITICAL_SECTION;
        InitializeCriticalSection(cs);
    }
    return cs;
}

IMPORT(ExInitializeReadWriteLock) { rw_lock(ctx.r3.u32); }
IMPORT(ExAcquireReadWriteLockExclusive) { EnterCriticalSection(rw_lock(ctx.r3.u32)); }
IMPORT(ExAcquireReadWriteLockShared) { EnterCriticalSection(rw_lock(ctx.r3.u32)); }
IMPORT(ExReleaseReadWriteLock) { LeaveCriticalSection(rw_lock(ctx.r3.u32)); }

// --- TLS -------------------------------------------------------------------------

IMPORT(KeTlsAlloc) {
    Lock held;
    for (uint32_t slot = 0; slot < 64; slot++) {
        if (!(tls_used[slot / 32] & (1u << (slot % 32)))) {
            tls_used[slot / 32] |= 1u << (slot % 32);
            ctx.r3.u64 = slot;
            return;
        }
    }
    ctx.r3.u64 = 0xFFFFFFFF;
}

IMPORT(KeTlsFree) {
    uint32_t slot = ctx.r3.u32;
    Lock held;
    if (slot < 64)
        tls_used[slot / 32] &= ~(1u << (slot % 32));
    ctx.r3.u64 = 1;
}

IMPORT(KeTlsGetValue) { ctx.r3.u64 = ctx.r3.u32 < 64 ? current_thread()->tls[ctx.r3.u32] : 0; }

IMPORT(KeTlsSetValue) {
    if (ctx.r3.u32 < 64)
        current_thread()->tls[ctx.r3.u32] = ctx.r4.u32;
    ctx.r3.u64 = ctx.r3.u32 < 64;
}

// --- time and I/O space ----------------------------------------------------------

IMPORT(KeQuerySystemTime) {
    // (time*): 100-nanosecond units since 1601
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    guest::store64(ctx.r3.u32, (uint64_t(now.dwHighDateTime) << 32) | now.dwLowDateTime);
}

IMPORT(RtlTimeToTimeFields) {
    // (time*, TIME_FIELDS*: year, month, day, hour, minute, second,
    // milliseconds, weekday, 16 bits each)
    uint64_t t = guest::load64(ctx.r3.u32);
    FILETIME time = {DWORD(t), DWORD(t >> 32)};
    SYSTEMTIME s = {};
    FileTimeToSystemTime(&time, &s);
    const WORD fields[8] = {s.wYear, s.wMonth, s.wDay, s.wHour, s.wMinute, s.wSecond, s.wMilliseconds, s.wDayOfWeek};
    for (int i = 0; i < 8; i++)
        guest::store16(ctx.r4.u32 + i * 2, fields[i]);
}

IMPORT(MmMapIoSpace) {
    // (2, address, bytes, flags): titles map XMA contexts with it, and the
    // address itself serves (Xenia's answer).
    ctx.r3.u64 = ctx.r4.u32;
}

// --- events, semaphores, waits ---------------------------------------------------

IMPORT(NtCreateEvent) {
    // (handle*, object attributes*, event type, initial state)
    HANDLE h = CreateEvent(nullptr, ctx.r5.u32 == 0, ctx.r6.u32 != 0, nullptr);
    guest::store32(ctx.r3.u32, add_handle(h));
    ctx.r3.u64 = h ? kSuccess : kNoMemory;
}

IMPORT(NtSetEvent) {
    if (ctx.r4.u32)
        guest::store32(ctx.r4.u32, 0);
    ctx.r3.u64 = SetEvent(host_handle(ctx.r3.u32)) ? kSuccess : kInvalidHandle;
}

IMPORT(NtClearEvent) { ctx.r3.u64 = ResetEvent(host_handle(ctx.r3.u32)) ? kSuccess : kInvalidHandle; }

IMPORT(KeSetEvent) {
    SetEvent(host_object(ctx.r3.u32));
    ctx.r3.u64 = 0;
}

IMPORT(KeResetEvent) {
    ResetEvent(host_object(ctx.r3.u32));
    ctx.r3.u64 = 0;
}

IMPORT(KeInitializeSemaphore) {
    // (semaphore, count, limit)
    uint32_t s = ctx.r3.u32;
    write_header(s, kSemaphore, 5, ctx.r4.s32);
    guest::store32(s + 16, ctx.r5.u32);
    HANDLE h = CreateSemaphore(nullptr, ctx.r4.s32, ctx.r5.s32, nullptr);
    bind(s, h);
}

IMPORT(NtCreateSemaphore) {
    // (handle*, object attributes*, initial count, limit)
    HANDLE h = CreateSemaphore(nullptr, ctx.r5.s32, ctx.r6.s32, nullptr);
    guest::store32(ctx.r3.u32, add_handle(h));
    ctx.r3.u64 = h ? kSuccess : kNoMemory;
}

IMPORT(NtReleaseSemaphore) {
    // (handle, count, previous count*)
    LONG previous = 0;
    bool released = ReleaseSemaphore(host_handle(ctx.r3.u32), ctx.r4.s32, &previous);
    if (ctx.r5.u32)
        guest::store32(ctx.r5.u32, uint32_t(previous));
    ctx.r3.u64 = released ? kSuccess : 0xC0000047;  // STATUS_SEMAPHORE_LIMIT_EXCEEDED
}

IMPORT(KeReleaseSemaphore) {
    // (semaphore, increment, adjustment, wait)
    LONG previous = 0;
    ReleaseSemaphore(host_object(ctx.r3.u32), ctx.r5.s32, &previous);
    ctx.r3.u64 = uint64_t(int64_t(previous));
}

// A wait that logs when it lasts more than a few seconds: what a hung title
// is waiting for.
static DWORD watched_wait(DWORD count, const HANDLE* handles, bool all, DWORD ms, bool alertable, uint32_t what) {
    const DWORD kReport = 3000;
    if (ms <= kReport)
        return WaitForMultipleObjectsEx(count, handles, all, ms, alertable);
    DWORD r = WaitForMultipleObjectsEx(count, handles, all, kReport, alertable);
    if (r != WAIT_TIMEOUT)
        return r;
    log("kernel: thread %u waiting more than %u s on %u object(s), first %08X\n", current_thread()->id,
        unsigned(kReport / 1000), unsigned(count), what);
    DWORD rest = ms == INFINITE ? INFINITE : ms - kReport;
    r = WaitForMultipleObjectsEx(count, handles, all, rest, alertable);
    log("kernel: thread %u done waiting on %08X\n", current_thread()->id, what);
    return r;
}

IMPORT(KeWaitForSingleObject) {
    // (object, reason, mode, alertable, timeout*)
    HANDLE h = host_object(ctx.r3.u32);
    DWORD r = watched_wait(1, &h, false, timeout_ms(ctx.r7.u32), ctx.r6.u32 != 0, ctx.r3.u32);
    ctx.r3.u64 = wait_status(r);
}

IMPORT(NtWaitForSingleObjectEx) {
    // (handle, mode, alertable, timeout*)
    HANDLE h = host_handle(ctx.r3.u32);
    DWORD r = watched_wait(1, &h, false, timeout_ms(ctx.r6.u32), ctx.r5.u32 != 0, ctx.r3.u32);
    ctx.r3.u64 = wait_status(r);
}

IMPORT(KeWaitForMultipleObjects) {
    // (count, objects*, wait type, reason, mode, alertable, timeout*, wait blocks)
    uint32_t count = ctx.r3.u32;
    HANDLE handles[MAXIMUM_WAIT_OBJECTS];
    if (count > MAXIMUM_WAIT_OBJECTS)
        fatal("kernel: wait for %u objects\n", count);
    for (uint32_t i = 0; i < count; i++)
        handles[i] = host_object(guest::load32(ctx.r4.u32 + i * 4));
    DWORD r = watched_wait(count, handles, ctx.r5.u32 == 0, timeout_ms(ctx.r9.u32), ctx.r8.u32 != 0, guest::load32(ctx.r4.u32));
    ctx.r3.u64 = wait_status(r);
}

IMPORT(NtWaitForMultipleObjectsEx) {
    // (count, handles*, wait type, mode, alertable, timeout*)
    uint32_t count = ctx.r3.u32;
    HANDLE handles[MAXIMUM_WAIT_OBJECTS];
    if (count > MAXIMUM_WAIT_OBJECTS)
        fatal("kernel: wait for %u handles\n", count);
    for (uint32_t i = 0; i < count; i++)
        handles[i] = host_handle(guest::load32(ctx.r4.u32 + i * 4));
    DWORD r = watched_wait(count, handles, ctx.r5.u32 == 0, timeout_ms(ctx.r8.u32), ctx.r7.u32 != 0, guest::load32(ctx.r4.u32));
    ctx.r3.u64 = wait_status(r);
}

// --- timers: the host's waitable timers (the same 100 ns due times) ---------------

IMPORT(NtCreateTimer) {
    // (handle*, object attributes*, type: 0 notification, 1 synchronization)
    HANDLE h = CreateWaitableTimer(nullptr, ctx.r5.u32 == 0, nullptr);
    guest::store32(ctx.r3.u32, add_handle(h));
    ctx.r3.u64 = h ? kSuccess : kNoMemory;
}

IMPORT(NtSetTimerEx) {
    // (handle, due time*, APC routine, mode, APC argument, resume, period ms,
    // previous state*)
    if (ctx.r5.u32) {
        static bool said;
        if (!said)
            log("kernel: timer APC routine %08X is not run\n", ctx.r5.u32);
        said = true;
    }
    LARGE_INTEGER due;
    due.QuadPart = int64_t(guest::load64(ctx.r4.u32));
    ctx.r3.u64 = SetWaitableTimer(host_handle(ctx.r3.u32), &due, LONG(ctx.r9.u32), nullptr, nullptr, FALSE)
                     ? kSuccess
                     : kInvalidHandle;
}

IMPORT(NtCancelTimer) {
    if (ctx.r4.u32)
        guest::store32(ctx.r4.u32, 0);
    ctx.r3.u64 = CancelWaitableTimer(host_handle(ctx.r3.u32)) ? kSuccess : kInvalidHandle;
}

// --- object manager --------------------------------------------------------------

IMPORT(NtClose) {
    close(host_handle(ctx.r3.u32));
    ctx.r3.u64 = kSuccess;
}

IMPORT(NtDuplicateObject) {
    // (handle, new handle*, options): the same handle, counted.
    HANDLE h = host_handle(ctx.r3.u32);
    {
        Lock held;
        duplicates[h]++;
    }
    if (ctx.r4.u32)
        guest::store32(ctx.r4.u32, add_handle(h));
    ctx.r3.u64 = kSuccess;
}

IMPORT(ObReferenceObjectByHandle) {
    // (handle, object type, object*)
    uint32_t handle = ctx.r3.u32;
    uint32_t object;
    if (handle == 0xFFFFFFFE)
        object = current_thread()->kthread;
    else
        object = object_for(host_handle(handle), ctx.r4.u32 == thread_object_type ? kThread : kNotificationEvent);
    if (ctx.r5.u32)
        guest::store32(ctx.r5.u32, object);
    ctx.r3.u64 = kSuccess;
}

IMPORT(ObReferenceObject) {}
IMPORT(ObDereferenceObject) {}
IMPORT(ObCreateSymbolicLink) { ctx.r3.u64 = kSuccess; }
IMPORT(ObDeleteSymbolicLink) { ctx.r3.u64 = kSuccess; }

// --- memory ------------------------------------------------------------------------

IMPORT(ExAllocatePoolWithTag) { ctx.r3.u64 = guest::alloc(ctx.r3.u32); }
// (size, tag, pool type)
IMPORT(ExAllocatePoolTypeWithTag) { ctx.r3.u64 = guest::alloc(ctx.r3.u32); }
IMPORT(ExFreePool) { guest::free(ctx.r3.u32); }

IMPORT(NtAllocateVirtualMemory) {
    // (base*, size*, allocation type, protect, debug)
    uint32_t address = guest::load32(ctx.r3.u32);
    uint32_t size = guest::load32(ctx.r4.u32);
    uint32_t type = ctx.r5.u32;
    // 64 KB pages with MEM_LARGE_PAGES, else 4 KB; reservations are whole
    // 64 KB units, as on the 360.
    uint32_t page = (type & 0x20000000) ? 0x10000 : guest::kPage;
    if (!address) {
        size = (size + 0xFFFF) & ~0xFFFFu;
        address = guest::reserve(size, 0x10000);
        if (!address) {
            log("kernel: out of guest address space for %u KB\n", size >> 10);
            ctx.r3.u64 = kNoMemory;
            return;
        }
    } else {
        uint32_t end = (address + size + page - 1) & ~(page - 1);
        address &= ~(page - 1);
        uint32_t start, block;
        bool committed;
        if (guest::query(address, start, block, committed) && end > start + block)
            end = start + block;
        size = end - address;
    }
    if ((type & 0x1000) && size >= 0x40000) {
        log("kernel: commit %u KB at %08X (%u KB committed before)\n", size >> 10, address, guest::committed_bytes() >> 10);
        dump_calls(current_thread(), 12);
    }
    if ((type & 0x1000) && !guest::commit(address, size)) {  // MEM_COMMIT
        log("kernel: cannot commit %u KB at %08X (%u KB committed)\n", size >> 10, address,
            guest::committed_bytes() >> 10);
        ctx.r3.u64 = kNoMemory;
        return;
    }
    guest::store32(ctx.r3.u32, address);
    guest::store32(ctx.r4.u32, size);
    ctx.r3.u64 = kSuccess;
}

IMPORT(NtFreeVirtualMemory) {
    // (base*, size*, free type, debug)
    uint32_t address = guest::load32(ctx.r3.u32);
    uint32_t size = guest::load32(ctx.r4.u32);
    if (ctx.r5.u32 & 0x8000) {  // MEM_RELEASE
        guest::free(address);
    } else {
        uint32_t start, block;
        bool committed;
        if (!size && guest::query(address, start, block, committed))
            size = block - (address - start);
        guest::decommit(address, size);
    }
    ctx.r3.u64 = kSuccess;
}

IMPORT(NtQueryVirtualMemory) {
    // (address, MEMORY_BASIC_INFORMATION*)
    uint32_t start, size;
    bool committed;
    uint32_t info = ctx.r4.u32;
    if (!guest::query(ctx.r3.u32, start, size, committed)) {
        ctx.r3.u64 = kInvalidParameter;
        return;
    }
    guest::store32(info + 0, ctx.r3.u32 & ~(guest::kPage - 1));
    guest::store32(info + 4, start);
    guest::store32(info + 8, PAGE_READWRITE);
    guest::store32(info + 12, start + size - (ctx.r3.u32 & ~(guest::kPage - 1)));
    guest::store32(info + 16, committed ? MEM_COMMIT : MEM_RESERVE);
    guest::store32(info + 20, PAGE_READWRITE);
    guest::store32(info + 24, MEM_PRIVATE);
    ctx.r3.u64 = kSuccess;
}

IMPORT(MmAllocatePhysicalMemoryEx) {
    // (flags, size, protect, min address, max address, alignment)
    uint32_t align = ctx.r8.u32 > guest::kPage ? ctx.r8.u32 : guest::kPage;
    uint32_t size = ctx.r4.u32;
    ctx.r3.u64 = guest::alloc(size, align);
    if (size >= 0x40000) {
        log("kernel: physical %u KB at %08X (%u KB committed before)\n", size >> 10, ctx.r3.u32,
            guest::committed_bytes() >> 10);
        dump_calls(current_thread(), 12);
    }
    if (!ctx.r3.u32)
        log("kernel: no physical memory for %u KB (%u KB committed)\n", ctx.r4.u32 >> 10, guest::committed_bytes() >> 10);
}

IMPORT(MmFreePhysicalMemory) {
    // Diagnosing: the title's main thread frees physical memory first when it
    // starts quitting.
    static bool dumped;
    if (!dumped && current_thread() == main_thread()) {
        dumped = true;
        dump_calls(current_thread(), 300);
    }
    guest::free(ctx.r4.u32);
}
// Guest memory is one window: its "physical" address is its address.
IMPORT(MmGetPhysicalAddress) {}
IMPORT(MmQueryAddressProtect) { ctx.r3.u64 = PAGE_READWRITE; }

// --- time ------------------------------------------------------------------------

// mftb is the host's time stamp counter (ppc_context.h): its rate.
IMPORT(KeQueryPerformanceFrequency) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    ctx.r3.u64 = uint64_t(frequency.QuadPart);
}

// --- run-time library --------------------------------------------------------------

IMPORT(RtlInitAnsiString) {
    // ANSI_STRING: Length, MaximumLength, Buffer.
    uint32_t s = ctx.r3.u32, text = ctx.r4.u32;
    uint32_t length = text ? uint32_t(strlen(reinterpret_cast<const char*>(guest::host(text)))) : 0;
    guest::store16(s, uint16_t(length));
    guest::store16(s + 2, uint16_t(text ? length + 1 : 0));
    guest::store32(s + 4, text);
}

IMPORT(RtlMultiByteToUnicodeN) {
    // (unicode*, max bytes, bytes written*, multibyte*, multibyte bytes)
    uint32_t count = ctx.r7.u32 < ctx.r4.u32 / 2 ? ctx.r7.u32 : ctx.r4.u32 / 2;
    for (uint32_t i = 0; i < count; i++)
        guest::store16(ctx.r3.u32 + i * 2, guest::load8(ctx.r6.u32 + i));
    if (ctx.r5.u32)
        guest::store32(ctx.r5.u32, count * 2);
    ctx.r3.u64 = kSuccess;
}

IMPORT(RtlNtStatusToDosError) {
    switch (ctx.r3.u32) {
    case kSuccess: ctx.r3.u64 = 0; break;
    case kObjectNameNotFound: ctx.r3.u64 = ERROR_FILE_NOT_FOUND; break;
    case kObjectPathNotFound: ctx.r3.u64 = ERROR_PATH_NOT_FOUND; break;
    case kInvalidHandle: ctx.r3.u64 = ERROR_INVALID_HANDLE; break;
    case kNoMemory: ctx.r3.u64 = ERROR_NOT_ENOUGH_MEMORY; break;
    case kEndOfFile: ctx.r3.u64 = ERROR_HANDLE_EOF; break;
    case kInvalidParameter: ctx.r3.u64 = ERROR_INVALID_PARAMETER; break;
    case kTimeout: ctx.r3.u64 = ERROR_TIMEOUT; break;
    case kPending: ctx.r3.u64 = ERROR_IO_PENDING; break;
    case kNoMoreFiles: ctx.r3.u64 = ERROR_NO_MORE_FILES; break;
    default: ctx.r3.u64 = ERROR_GEN_FAILURE; break;
    }
}

IMPORT(RtlCompareMemoryUlong) {
    // (source, length, pattern): bytes matching from the start
    uint32_t n = 0;
    while (n + 4 <= ctx.r4.u32 && guest::load32(ctx.r3.u32 + n) == ctx.r5.u32)
        n += 4;
    ctx.r3.u64 = n;
}

IMPORT(RtlFillMemoryUlong) {
    for (uint32_t n = 0; n + 4 <= ctx.r4.u32; n += 4)
        guest::store32(ctx.r3.u32 + n, ctx.r5.u32);
}

IMPORT(RtlRaiseException) {
    uint32_t record = ctx.r3.u32;
    fatal("kernel: the title raised exception %08X at %08X\n", guest::load32(record), guest::load32(record + 12));
}

IMPORT(RtlUnwind) { fatal("kernel: RtlUnwind (structured exceptions are not supported)\n"); }
IMPORT(__C_specific_handler) { fatal("kernel: __C_specific_handler (structured exceptions are not supported)\n"); }

IMPORT(KeBugCheck) { fatal("kernel: KeBugCheck %08X\n", ctx.r3.u32); }
IMPORT(KeBugCheckEx) { fatal("kernel: KeBugCheckEx %08X (%08X %08X %08X %08X)\n", ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, ctx.r6.u32, ctx.r7.u32); }
IMPORT(DbgBreakPoint) { log("kernel: DbgBreakPoint\n"); }
IMPORT(HalReturnToFirmware) { fatal("kernel: HalReturnToFirmware %u\n", ctx.r3.u32); }

// --- the loader ----------------------------------------------------------------------

IMPORT(RtlImageXexHeaderField) {
    // (header base, key): small values come back as they are, others as
    // pointers into the header.
    uint32_t entry = find_xex_header(ctx.r4.u32);
    if (!entry) {
        ctx.r3.u64 = 0;
        return;
    }
    uint32_t value = guest::load32(entry + 4);
    switch (ctx.r4.u32 & 0xFF) {
    case 0x00: ctx.r3.u64 = value; break;
    case 0x01: ctx.r3.u64 = entry + 4; break;
    default: ctx.r3.u64 = xex_header + value; break;
    }
}

IMPORT(XexCheckExecutablePrivilege) { ctx.r3.u64 = 0; }

IMPORT(XexGetModuleHandle) {
    // (name, handle*): only the title itself.
    uint32_t name = ctx.r3.u32;
    if (name) {
        ctx.r3.u64 = kNotFound;
        return;
    }
    guest::store32(ctx.r4.u32, guest::load32(executable_module));
    ctx.r3.u64 = kSuccess;
}

IMPORT(XexGetModuleSection) {
    // (module, name, data*, size*): the title's resources (XEX header 0x2FF:
    // entries of an 8-character name, address and size).
    const char* name = reinterpret_cast<const char*>(guest::host(ctx.r4.u32));
    uint32_t entry = find_xex_header(0x000002FF);
    if (entry) {
        uint32_t table = xex_header + guest::load32(entry + 4);
        uint32_t count = (guest::load32(table) - 4) / 16;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t r = table + 4 + i * 16;
            if (strncmp(reinterpret_cast<const char*>(guest::host(r)), name, 8) == 0) {
                guest::store32(ctx.r5.u32, guest::load32(r + 8));
                guest::store32(ctx.r6.u32, guest::load32(r + 12));
                ctx.r3.u64 = kSuccess;
                return;
            }
        }
    }
    log("kernel: no module section %.8s\n", name);
    ctx.r3.u64 = kNotFound;
}

IMPORT(XexGetProcedureAddress) { ctx.r3.u64 = kNotFound; }
IMPORT(ExRegisterTitleTerminateNotification) {}

IMPORT(ExGetXConfigSetting) {
    // (category, setting, buffer, size, required size*): defaults, zeroed.
    if (ctx.r5.u32)
        memset(guest::host(ctx.r5.u32), 0, ctx.r6.u32 & 0xFFFF);
    if (ctx.r7.u32)
        guest::store16(ctx.r7.u32, uint16_t(ctx.r6.u32));
    ctx.r3.u64 = kSuccess;
}

IMPORT(XGetAVPack) { ctx.r3.u64 = 6; }      // standard AV cable
IMPORT(XGetLanguage) { ctx.r3.u64 = 1; }    // English
IMPORT(XGetGameRegion) { ctx.r3.u64 = 0xFF; }

IMPORT(XGetVideoMode) {
    // XVIDEO_MODE: 640x480, 4:3, 60 Hz, NTSC-M.
    uint32_t m = ctx.r3.u32;
    memset(guest::host(m), 0, 48);
    guest::store32(m + 0, 640);
    guest::store32(m + 4, 480);
    guest::store32(m + 8, 0);   // interlaced
    guest::store32(m + 12, 0);  // widescreen
    guest::store32(m + 16, 0);  // high definition
    float refresh = 60.0f;
    uint32_t bits;
    memcpy(&bits, &refresh, 4);
    guest::store32(m + 20, bits);
    guest::store32(m + 24, 1);  // XC_VIDEO_STANDARD_NTSC_M
}
