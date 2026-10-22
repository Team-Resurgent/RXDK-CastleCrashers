// The guest memory window (guest.h): reservation, the heap region's page
// allocator and guest calls through the function table.
#include "guest.h"

#include <algorithm>
#include <map>
#include <vector>

namespace guest {

PPCBase base;

namespace {

// The image and its function table (one host pointer per guest instruction:
// PPC_LOOKUP_FUNC in ppc_context.h).
constexpr uint32_t kTable = uint32_t(PPC_IMAGE_BASE + PPC_IMAGE_SIZE);
constexpr uint32_t kTableSize = (uint32_t(PPC_CODE_SIZE) * (sizeof(PPCFunc*) / 4) + kPage - 1) & ~(kPage - 1);
// The GPU's writeback page and registers (gpu.cpp): plain memory here.
constexpr uint32_t kGpuWriteback = 0x7F000000, kGpuWritebackSize = 0x10000;
constexpr uint32_t kGpuRegisters = 0x7FC80000, kGpuRegistersSize = 0x20000;
// The XMA decoder's registers: plain memory, nothing decodes (audio is not
// done yet).
constexpr uint32_t kXmaRegisters = 0x7FEA0000, kXmaRegistersSize = 0x10000;
// The heap region: everything the kernel and the title allocate.
constexpr uint32_t kHeap = 0x83000000;
constexpr uint32_t kHeapEnd = kWindowBase + kWindowSize;
static_assert(kHeapEnd <= 0xA0000000, "GPU physical addresses map to 0x80000000 + address");
constexpr uint32_t kHeapPages = (kHeapEnd - kHeap) / kPage;

// kLazy: committed for the title, with no memory until it is touched.
enum : uint8_t { kFree, kReserved, kCommitted, kLazy };

uint8_t pages[kHeapPages];
// Pages release() took the memory of: 1 until the title touches one (2).
// Cleared when the title decommits or frees them.
uint8_t released[kHeapPages];
std::map<uint32_t, uint32_t> blocks;  // start -> size of each reservation
uint32_t committed, resident_total;
uint32_t first_free;  // no page below it is free (reserve's first fit starts there)
CRITICAL_SECTION lock;

struct Lock {
    Lock() { EnterCriticalSection(&lock); }
    ~Lock() { LeaveCriticalSection(&lock); }
};

bool host_commit(uint32_t address, uint32_t size) {
    return VirtualAlloc(host(address), size, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

// A lazily committed page's memory, zeroed (lock held). False when the
// console has none left.
bool make_resident(uint32_t p) {
    uint32_t address = kHeap + p * kPage;
    if (!host_commit(address, kPage))
        return false;
    memset(host(address), 0, kPage);
    pages[p] = kCommitted;
    resident_total += kPage;
    return true;
}

// The structured exception handler of GuestFaults' frames: an access
// violation on a lazily committed page commits it and resumes the
// instruction; anything else goes on to the next handler.
struct ExceptionRecord {
    DWORD code, flags;
    ExceptionRecord* record;
    void* address;
    DWORD parameters;
    ULONG_PTR information[15];
};
constexpr int kContinueExecution = 0, kContinueSearch = 1;

extern "C" int __cdecl guest_fault_handler(ExceptionRecord* record, void*, void*, void*) {
    if (record->code != 0xC0000005 || record->parameters < 2 || (record->flags & 6))  // access violation
        return kContinueSearch;  // flags 6: unwinding
    uint32_t guest = uint32_t(record->information[1] - reinterpret_cast<ULONG_PTR>(base.window)) + kWindowBase;
    if (guest < kHeap || guest >= kHeapEnd)
        return kContinueSearch;
    Lock held;
    uint32_t p = (guest - kHeap) / kPage;
    if (pages[p] == kCommitted)
        return kContinueExecution;  // another thread committed it first
    if (pages[p] != kLazy)
        return kContinueSearch;
    if (released[p])
        released[p] = 2;
    if (!make_resident(p)) {
        log("guest: no memory for the page at %08X (%u KB resident)\n", kHeap + p * kPage, resident_total >> 10);
        return kContinueSearch;
    }
    return kContinueExecution;
}

}  // namespace

GuestFaults::GuestFaults() : handler_(reinterpret_cast<void*>(&guest_fault_handler)) {
    void* frame = this;
    __asm__ volatile("movl %%fs:0, %0" : "=r"(next_));
    __asm__ volatile("movl %0, %%fs:0" : : "r"(frame) : "memory");
}

GuestFaults::~GuestFaults() { __asm__ volatile("movl %0, %%fs:0" : : "r"(next_) : "memory"); }

bool init() {
    InitializeCriticalSection(&lock);
    void* window = VirtualAlloc(nullptr, kWindowSize, MEM_RESERVE, PAGE_READWRITE);
    if (!window) {
        log("guest: cannot reserve the %u MB window\n", kWindowSize >> 20);
        return false;
    }
    static_assert(kWindowBase == PPCBase::PPC_WINDOW_BASE && kWindowSize == PPCBase::PPC_WINDOW_MASK + 1);
    base.window = static_cast<uint8_t*>(window);
    if (!host_commit(uint32_t(PPC_IMAGE_BASE), uint32_t(PPC_IMAGE_SIZE)) || !host_commit(kTable, kTableSize) ||
        !host_commit(kGpuWriteback, kGpuWritebackSize) || !host_commit(kGpuRegisters, kGpuRegistersSize) ||
        !host_commit(kXmaRegisters, kXmaRegistersSize)) {
        log("guest: cannot commit the image (%u KB)\n", (uint32_t(PPC_IMAGE_SIZE) + kTableSize) >> 10);
        return false;
    }
    for (PPCFuncMapping* m = PPCFuncMappings; m->host; m++)
        PPC_LOOKUP_FUNC(base, m->guest) = m->host;
    log("guest: window %08X at host %p, image %08X-%08X, table %u KB\n", kWindowBase, window,
        uint32_t(PPC_IMAGE_BASE), kTable, kTableSize >> 10);
    return true;
}

uint32_t reserve(uint32_t size, uint32_t align) {
    size = (size + kPage - 1) & ~(kPage - 1);
    if (align < kPage)
        align = kPage;
    uint32_t count = size / kPage, step = align / kPage;
    Lock held;
    // Every page below first_free is in use: first fit from there.
    while (first_free < kHeapPages && pages[first_free] != kFree)
        first_free++;
    for (uint32_t first = first_free / step * step; first + count <= kHeapPages; first += step) {
        uint32_t i = 0;
        while (i < count && pages[first + i] == kFree)
            i++;
        if (i == count) {
            memset(pages + first, kReserved, count);
            uint32_t address = kHeap + first * kPage;
            blocks[address] = size;
            return address;
        }
        // Skip past the page in use, keeping the alignment.
        first = ((first + i) / step) * step;
    }
    return 0;
}

bool commit(uint32_t address, uint32_t size) {
    if (address < kHeap || address >= kHeapEnd)
        return false;
    uint32_t first = (address - kHeap) / kPage;
    uint32_t last = (address + size - 1 - kHeap) / kPage;
    Lock held;
    for (uint32_t p = first; p <= last; p++)
        if (pages[p] == kFree)
            return false;
    for (uint32_t p = first; p <= last; p++) {
        if (pages[p] == kReserved) {
            pages[p] = kLazy;
            committed += kPage;
        }
    }
    return true;
}

bool resident(uint32_t address) {
    if (address < kHeap || address >= kHeapEnd)
        return true;  // the image and the registers are always there
    return pages[(address - kHeap) / kPage] == kCommitted;  // a byte: no lock needed to read it
}

void release(uint32_t address, uint32_t size) {
    if (address < kHeap || address >= kHeapEnd || size < kPage)
        return;
    uint32_t first = (address - kHeap + kPage - 1) / kPage;  // whole pages only
    uint32_t end = (address + size - kHeap) / kPage;
    Lock held;
    for (uint32_t p = first; p < end && p < kHeapPages; p++) {
        if (pages[p] == kCommitted) {
            VirtualFree(host(kHeap + p * kPage), kPage, MEM_DECOMMIT);
            resident_total -= kPage;
            pages[p] = kLazy;
        }
        if (pages[p] == kLazy)
            released[p] = 1;
    }
}

Released released_state(uint32_t address, uint32_t size) {
    if (address < kHeap || address >= kHeapEnd || size < kPage)
        return Released::kGone;
    uint32_t first = (address - kHeap + kPage - 1) / kPage;
    uint32_t end = (address + size - kHeap) / kPage;
    Released state = Released::kKept;
    Lock held;
    for (uint32_t p = first; p < end && p < kHeapPages; p++) {
        if (!released[p])
            return Released::kGone;
        if (released[p] == 2)
            state = Released::kTouched;
    }
    return state;
}

void touch(uint32_t address, uint32_t size) {
    if (!size || address < kHeap || address >= kHeapEnd)
        return;
    uint32_t first = (address - kHeap) / kPage;
    uint32_t last = (address + size - 1 - kHeap) / kPage;
    Lock held;
    for (uint32_t p = first; p <= last && p < kHeapPages; p++) {
        if (pages[p] != kLazy)
            continue;
        if (released[p])
            released[p] = 2;
        if (!make_resident(p))
            log("guest: no memory for the page at %08X (%u KB resident)\n", kHeap + p * kPage, resident_total >> 10);
    }
}

bool decommit(uint32_t address, uint32_t size) {
    if (address < kHeap || address >= kHeapEnd)
        return false;
    uint32_t first = (address - kHeap) / kPage;
    uint32_t last = (address + size - 1 - kHeap) / kPage;
    Lock held;
    for (uint32_t p = first; p <= last; p++) {
        if (pages[p] == kCommitted) {
            VirtualFree(host(kHeap + p * kPage), kPage, MEM_DECOMMIT);
            resident_total -= kPage;
        }
        if (pages[p] == kCommitted || pages[p] == kLazy) {
            pages[p] = kReserved;
            committed -= kPage;
        }
        released[p] = 0;
    }
    return true;
}

uint32_t alloc(uint32_t size, uint32_t align) {
    uint32_t address = reserve(size, align);
    if (address && !commit(address, size)) {
        free(address);
        return 0;
    }
    return address;
}

void free(uint32_t address) {
    uint32_t size;
    {
        Lock held;
        auto it = blocks.find(address);
        if (it == blocks.end())
            return;
        size = it->second;
        blocks.erase(it);
    }
    decommit(address, size);
    Lock held;
    uint32_t first = (address - kHeap) / kPage;
    memset(pages + first, kFree, size / kPage);
    first_free = first < first_free ? first : first_free;
}

uint32_t block_size(uint32_t address) {
    Lock held;
    auto it = blocks.find(address);
    return it == blocks.end() ? 0 : it->second;
}

bool query(uint32_t address, uint32_t& start, uint32_t& size, bool& is_committed) {
    Lock held;
    auto it = blocks.upper_bound(address);
    if (it == blocks.begin())
        return false;
    --it;
    if (address >= it->first + it->second)
        return false;
    start = it->first;
    size = it->second;
    uint8_t state = pages[(address - kHeap) / kPage];
    is_committed = state == kCommitted || state == kLazy;
    return true;
}

uint32_t committed_bytes() { return committed; }
uint32_t resident_bytes() { return resident_total; }

void log_blocks(uint32_t count) {
    struct Use {
        uint32_t start, size, touched;
    };
    std::vector<Use> uses;
    {
        Lock held;
        for (auto& [start, size] : blocks) {
            uint32_t touched = 0;
            for (uint32_t p = (start - kHeap) / kPage, n = size / kPage; n--; p++)
                touched += pages[p] == kCommitted ? kPage : 0;
            uses.push_back({start, size, touched});
        }
    }
    std::sort(uses.begin(), uses.end(), [](const Use& a, const Use& b) { return a.touched > b.touched; });
    uint32_t small = 0;
    for (size_t i = 0; i < uses.size(); i++) {
        if (i < count)
            log("guest: block %08X, %u KB, %u KB touched\n", uses[i].start, uses[i].size >> 10, uses[i].touched >> 10);
        else
            small += uses[i].touched;
    }
    log("guest: %u more blocks, %u KB touched\n", unsigned(uses.size() > count ? uses.size() - count : 0), small >> 10);
}

// Diagnostic (input script "scan" / "poke"): narrowing searches for a guest
// dword, as a cheat finder does. The first scan covers the image and the
// committed heap; each later one keeps the candidates that now hold value.
namespace {
std::vector<uint32_t> candidates;
bool scanned;
}  // namespace

uint32_t scan(uint32_t value) {
    if (!scanned) {
        scanned = true;
        auto sweep = [&](uint32_t start, uint32_t size) {
            for (uint32_t a = start; a + 4 <= start + size; a += 4)
                if (load32(a) == value)
                    candidates.push_back(a);
        };
        sweep(uint32_t(PPC_IMAGE_BASE), uint32_t(PPC_IMAGE_SIZE));
        Lock held;
        for (uint32_t p = 0; p < kHeapPages; p++)
            if (pages[p] == kCommitted)
                sweep(kHeap + p * kPage, kPage);
    } else {
        // Heap pages freed since go (reading them would fault).
        std::vector<uint32_t> kept;
        Lock held;
        for (uint32_t a : candidates) {
            bool heap = a >= kHeap && a < kHeapEnd;
            if (heap && pages[(a - kHeap) / kPage] != kCommitted)
                continue;
            if (load32(a) == value)
                kept.push_back(a);
        }
        candidates.swap(kept);
    }
    return uint32_t(candidates.size());
}

const std::vector<uint32_t>& scan_candidates() { return candidates; }

void poke_candidates(uint32_t value) {
    Lock held;
    for (uint32_t a : candidates) {
        bool heap = a >= kHeap && a < kHeapEnd;
        if (!heap || pages[(a - kHeap) / kPage] == kCommitted)
            store32(a, value);
    }
}

}  // namespace guest

void ppc_missing_function(uint32_t guest_address) {
    GuestThread* t = current_thread();
    log("guest: thread %u called %08X through a pointer: no recompiled function there\n", t ? t->id : 0, guest_address);
    if (t)
        dump_calls(t, 40);
    fatal("guest: call to %08X\n", guest_address);
}

void ppc_spin_yield() {
    // Every 16th time a millisecond, so threads of lower priority run too.
    GuestThread* t = current_thread();
    Sleep(t && ++t->spins % 16 == 0 ? 1 : 0);
}

void ppc_bad_switch(uint32_t first_case, uint32_t index) {
    GuestThread* t = current_thread();
    log("guest: thread %u: index %u past the cases of the jump table whose first case is %08X\n", t ? t->id : 0,
        index, first_case);
    if (t)
        dump_calls(t, 40);
    fatal("guest: jump table at %08X\n", first_case);
}

namespace guest {

void call(PPCContext& ctx, uint32_t function) {
    if (function < PPC_CODE_BASE || function >= PPC_CODE_BASE + PPC_CODE_SIZE)
        fatal("guest: call to %08X, outside the code\n", function);
    PPCFunc* f = PPC_LOOKUP_FUNC(base, function);
    if (!f)
        fatal("guest: call to %08X, which is not a recompiled function\n", function);
    f(ctx, base);
}

}  // namespace guest
