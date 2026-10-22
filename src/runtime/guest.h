// Guest memory and the per-thread guest state.
//
// The title's 32-bit address space is a window of host memory: guest
// addresses kWindowBase..kWindowBase+kWindowSize map to one reserved host
// range, and `base` is that range minus kWindowBase. The Pentium III's
// pointers are 32 bits too, so base + guest address wraps to the host address
// with no masking (the generated code computes exactly that). Everything the
// title can address lives in the window: its image at 0x82000000, the
// function table after it, heaps, stacks and the kernel's objects.
#pragma once

#include <xtl.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "ppc_config.h"
#include "ppc_context.h"

namespace guest {

// From the GPU's writeback page (0x7F000000) and registers (0x7FC80000) up.
constexpr uint32_t kWindowBase = 0x7F000000;
constexpr uint32_t kWindowSize = 0x10000000;
constexpr uint32_t kPage = 0x1000;

// base + guest address: the host address (PPCBase, ppc_context.h).
extern PPCBase base;

inline uint8_t* host(uint32_t address) { return base + address; }
inline uint32_t address_of(const void* p) { return uint32_t(static_cast<const uint8_t*>(p) - base.window) + kWindowBase; }

inline uint8_t load8(uint32_t a) { return *host(a); }
inline uint16_t load16(uint32_t a) { uint16_t v; __builtin_memcpy(&v, host(a), 2); return __builtin_bswap16(v); }
inline uint32_t load32(uint32_t a) { uint32_t v; __builtin_memcpy(&v, host(a), 4); return __builtin_bswap32(v); }
inline uint64_t load64(uint32_t a) { uint64_t v; __builtin_memcpy(&v, host(a), 8); return __builtin_bswap64(v); }
inline void store8(uint32_t a, uint8_t v) { *host(a) = v; }
inline void store16(uint32_t a, uint16_t v) { v = __builtin_bswap16(v); __builtin_memcpy(host(a), &v, 2); }
inline void store32(uint32_t a, uint32_t v) { v = __builtin_bswap32(v); __builtin_memcpy(host(a), &v, 4); }
inline void store64(uint32_t a, uint64_t v) { v = __builtin_bswap64(v); __builtin_memcpy(host(a), &v, 8); }

// Reserves the window and commits the image and function table.
bool init();

// Committed, zeroed guest memory from the heap region. size is rounded up to
// pages; align is a power of two of at least a page. 0 when out of memory.
uint32_t alloc(uint32_t size, uint32_t align = kPage);
void free(uint32_t address);
// Size of the block alloc returned at address (0 when it is not one).
uint32_t block_size(uint32_t address);

// NtAllocateVirtualMemory's reserve and commit: reserve takes address space
// only, commit backs pages of a reserved block with memory.
//
// Commit is lazy: the title commits far more than it touches (a 32 MB heap,
// a 6 MB arena per 360 processor), so a committed page gets its memory, zeroed,
// the first time it is touched. The touch faults, and the GuestFaults frame
// of the thread that touched it commits the page and goes on.
uint32_t reserve(uint32_t size, uint32_t align);
bool commit(uint32_t address, uint32_t size);
bool decommit(uint32_t address, uint32_t size);
// The reserved block containing address: its start and size (false if none).
bool query(uint32_t address, uint32_t& start, uint32_t& size, bool& committed);
// Gives committed pages in [address, address + size) their memory now: for
// buffers the kernel fills (file reads), which do not fault pages in.
void touch(uint32_t address, uint32_t size);

// The pages wholly inside [address, address + size) give up their memory
// and contents while staying committed: the renderer has taken the texture
// that was there. If the title touches one again it is a zeroed page, and the
// range's state says so.
void release(uint32_t address, uint32_t size);
enum class Released { kKept, kTouched, kGone };
// kKept: nothing touched since release; kTouched: the title wrote there
// again; kGone: the title freed the memory.
Released released_state(uint32_t address, uint32_t size);

// Committed heap memory (what the title asked for), and the part of it that
// has memory, for the log.
uint32_t committed_bytes();
uint32_t resident_bytes();
// Logs the count blocks with the most touched memory, and the rest's total.
void log_blocks(uint32_t count);

// Guest memory faults handled on this thread while this lives on its stack:
// an x86 structured exception handler frame (fs:[0]). Every thread that
// touches guest memory starts with one.
class GuestFaults {
public:
    GuestFaults();
    ~GuestFaults();
    GuestFaults(const GuestFaults&) = delete;

private:
    // The system's EXCEPTION_REGISTRATION_RECORD: the frame registered
    // before this one, and the handler.
    void* next_;
    void* handler_;
};

// Diagnostic: narrow a search for a dword holding value (memory.cpp scan);
// the number of candidates left, and the candidates.
uint32_t scan(uint32_t value);
const std::vector<uint32_t>& scan_candidates();
// Writes value to the candidates still in committed memory.
void poke_candidates(uint32_t value);

// A guest call: the function the guest address names, with ctx's registers.
void call(PPCContext& ctx, uint32_t function);

}  // namespace guest

// The guest thread running on this host thread.
struct GuestThread {
    PPCContext ctx;
    uint32_t pcr = 0;      // r13: the processor control region (kernel.cpp)
    uint32_t kthread = 0;  // its KTHREAD, the object handles name
    uint32_t stack = 0, stack_size = 0;
    uint32_t id = 0;
    uint32_t tls[64] = {};
    HANDLE handle = nullptr;
    uint32_t start = 0, start_context = 0, xapi_startup = 0;
    int priority = 0;
    // The last import it called, from where (a host address in the
    // recompiled code), and whether it is still in it: what the watchdog
    // (kernel.cpp) reports.
    const char* import = nullptr;
    void* import_caller = nullptr;
    volatile bool in_import = false;
    GuestThread* next = nullptr;  // all threads, for the watchdog
};

GuestThread* current_thread();
// Makes thread the guest thread of the calling host thread (for host
// threads that call into the guest, such as interrupt callbacks).
void set_current_thread(GuestThread* thread);
// A guest thread's guest state (stack, PCR, KTHREAD) with no host thread.
GuestThread* guest_thread_state(uint32_t stack_size, uint32_t start);
// Logs the last count recompiled functions the thread entered (with
// PPC_CALL_TRACE; nothing without).
void dump_calls(GuestThread* thread, uint32_t count);
// Creates a guest thread (suspended: resume with ResumeThread).
GuestThread* create_thread(uint32_t stack_size, uint32_t xapi_startup, uint32_t start, uint32_t context);

void log(const char* format, ...);
[[noreturn]] void fatal(const char* format, ...);
