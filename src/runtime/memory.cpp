// The guest memory window (guest.h): reservation, the heap region's page
// allocator and guest calls through the function table.
#include "guest.h"

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

enum : uint8_t { kFree, kReserved, kCommitted };

uint8_t pages[kHeapPages];
std::map<uint32_t, uint32_t> blocks;  // start -> size of each reservation
uint32_t committed;
CRITICAL_SECTION lock;

struct Lock {
    Lock() { EnterCriticalSection(&lock); }
    ~Lock() { LeaveCriticalSection(&lock); }
};

bool host_commit(uint32_t address, uint32_t size) {
    return VirtualAlloc(host(address), size, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

}  // namespace

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
    for (uint32_t first = 0; first + count <= kHeapPages; first += step) {
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
    for (uint32_t p = first; p <= last; p++) {
        if (pages[p] == kFree)
            return false;
        if (pages[p] == kReserved) {
            if (!host_commit(kHeap + p * kPage, kPage))
                return false;
            memset(host(kHeap + p * kPage), 0, kPage);
            pages[p] = kCommitted;
            committed += kPage;
        }
    }
    return true;
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
            pages[p] = kReserved;
            committed -= kPage;
        }
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
    memset(pages + (address - kHeap) / kPage, kFree, size / kPage);
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
    is_committed = pages[(address - kHeap) / kPage] == kCommitted;
    return true;
}

uint32_t committed_bytes() { return committed; }

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
