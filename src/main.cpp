// Castle Crashers on the original Xbox.
//
// The title's PowerPC code, statically recompiled to C++ (XenonRecomp,
// build/ppc), runs against this runtime: its memory image is loaded into a
// guest memory window at its own addresses (runtime/memory.cpp), and its
// imports from the Xbox 360 kernel and XAM are implemented on the original
// Xbox's (runtime/kernel.cpp, io.cpp, xam.cpp). D:\ holds the user's own
// files: image.bin (tools/xexprep's dump of their default.xex), default.xex
// for its header, and the package's data (the title update's in D:\update).
#include "runtime/kernel.h"

#include <cstdarg>
#include <cstdio>

void log(const char* format, ...) {
    char line[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    OutputDebugStringA(line);
}

void fatal(const char* format, ...) {
    char line[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    OutputDebugStringA("*** FATAL: ");
    OutputDebugStringA(line);
    for (;;)
        Sleep(1000);
}

namespace {

bool read_file(const char* path, void* data, uint32_t size) {
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        log("loader: cannot open %s (%lu)\n", path, GetLastError());
        return false;
    }
    // In pieces: one read of megabytes from the DVD fails.
    uint8_t* to = static_cast<uint8_t*>(data);
    for (uint32_t done = 0; done < size;) {
        DWORD want = size - done < 0x10000 ? size - done : 0x10000, read = 0;
        if (!ReadFile(f, to + done, want, &read, nullptr) || read != want) {
            log("loader: read of %s failed at %u (%lu)\n", path, done, GetLastError());
            CloseHandle(f);
            return false;
        }
        done += read;
    }
    CloseHandle(f);
    return true;
}

struct Import {
    uint32_t slot, thunk;
    bool kernel;
    uint32_t ordinal;
};

#define PPC_KERNEL true
#define PPC_XAM false
#define PPC_IMPORT(slot, thunk, library, ordinal) {slot, thunk, library, ordinal},
const Import imports[] = {
#include "ppc_imports.inc"
};

}  // namespace

int main() {
    guest::GuestFaults faults;
    log("Castle Crashers: starting\n");
    if (!guest::init())
        fatal("no guest memory\n");
    if (!read_file("D:\\image.bin", guest::host(uint32_t(PPC_IMAGE_BASE)), uint32_t(PPC_IMAGE_SIZE)))
        fatal("cannot read D:\\image.bin\n");

    // The XEX header: its size is at offset 8.
    uint8_t start[16];
    if (!read_file("D:\\default.xex", start, sizeof(start)) || memcmp(start, "XEX2", 4) != 0)
        fatal("cannot read D:\\default.xex\n");
    uint32_t header_size = (uint32_t(start[8]) << 24) | (uint32_t(start[9]) << 16) | (uint32_t(start[10]) << 8) | start[11];
    uint32_t header = guest::alloc(header_size);
    guest::touch(header, header_size);
    if (!read_file("D:\\default.xex", guest::host(header), header_size))
        fatal("cannot read the XEX header\n");

    kernel::init(header);
    for (const Import& i : imports) {
        uint32_t value = i.thunk ? i.thunk : kernel::import_variable(i.kernel, i.ordinal);
        if (!value)
            log("loader: no variable for %s ordinal %03X\n", i.kernel ? "xboxkrnl" : "xam", i.ordinal);
        guest::store32(i.slot, value);
    }
    xam::init();
    audio::init();

    // XEX header 0x10100: the entry point.
    uint32_t entry = 0;
    for (uint32_t i = 0, count = guest::load32(header + 0x14); i < count; i++)
        if (guest::load32(header + 0x18 + i * 8) == 0x00010100)
            entry = guest::load32(header + 0x18 + i * 8 + 4);
    GuestThread* main = kernel::main_thread();
    main->start = entry;
    log("loader: entry %08X, %u imports, %u KB of guest memory committed\n", entry, unsigned(sizeof(imports) / sizeof(imports[0])),
        guest::committed_bytes() >> 10);
    ResumeThread(main->handle);
    WaitForSingleObject(main->handle, INFINITE);
    fatal("the title's main thread returned %08X\n", main->ctx.r3.u32);
}
