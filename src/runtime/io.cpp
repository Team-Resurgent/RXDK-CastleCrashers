// Files: the title's paths on the host's drives.
//
// game:\ and the DVD (\Device\Cdrom0) are D:\, the title's own folder on the
// Xbox, and update:\ (the title update) D:\update; anything else the title
// writes goes under T:\. libxapi's
// GetFileAttributesExA and FindFirstFileA fault, so only GetFileAttributesA,
// CreateFileA and handle calls are used. Directories open as handles the
// kernel only remembers (the title queries nothing in them).
#include "kernel.h"

#include <map>
#include <set>
#include <string>

// The XMA decoder (xma.cpp).
namespace xma {
void save_packet_headers(uint32_t address, uint32_t size);
void sound_read(const char* path, uint32_t address, uint32_t offset, uint32_t size);
}  // namespace xma

namespace {

std::set<HANDLE> directories;
std::map<HANDLE, std::string> file_paths;  // for the log (a reused handle gets its new path)
std::map<HANDLE, HANDLE> listings;          // a directory's kernel handle, for NtQueryDirectoryFile
CRITICAL_SECTION directories_lock;
struct Init {
    Init() { InitializeCriticalSection(&directories_lock); }
} init_lock;

std::string path_of(HANDLE h) {
    EnterCriticalSection(&directories_lock);
    auto it = file_paths.find(h);
    std::string path = it == file_paths.end() ? "?" : it->second;
    LeaveCriticalSection(&directories_lock);
    return path;
}

bool starts_with(const std::string& s, const char* prefix) {
    size_t n = strlen(prefix);
    return s.size() >= n && _strnicmp(s.c_str(), prefix, n) == 0;
}

std::string host_path(uint32_t attributes) {
    // OBJECT_ATTRIBUTES: RootDirectory, ObjectName (ANSI_STRING*), Attributes
    uint32_t name = guest::load32(attributes + 4);
    uint16_t length = guest::load16(name);
    std::string path(reinterpret_cast<const char*>(guest::host(guest::load32(name + 4))), length);
    for (char& c : path)
        if (c == '/')
            c = '\\';
    if (starts_with(path, "\\??\\"))
        path.erase(0, 4);
    static const char* dvd[] = {"game:", "d:", "\\Device\\Cdrom0", "\\Device\\Harddisk0\\Partition1"};
    for (const char* prefix : dvd) {
        if (starts_with(path, prefix)) {
            std::string rest = path.substr(strlen(prefix));
            if (rest.empty() || rest[0] != '\\')
                rest = "\\" + rest;
            return "D:" + rest;
        }
    }
    // The title update's files (tools/build.py deploys them to D:\update).
    if (starts_with(path, "update:")) {
        std::string rest = path.substr(7);
        if (rest.empty() || rest[0] != '\\')
            rest = "\\" + rest;
        return "D:\\update" + rest;
    }
    size_t colon = path.find(':');
    if (colon != std::string::npos) {
        std::string rest = path.substr(colon + 1);
        if (rest.empty() || rest[0] != '\\')
            rest = "\\" + rest;
        return "T:\\" + path.substr(0, colon) + rest;
    }
    return "D:\\" + path;
}

void io_status(uint32_t status_block, uint32_t status, uint32_t information) {
    if (status_block) {
        guest::store32(status_block, status);
        guest::store32(status_block + 4, information);
    }
}

void open(PPCContext& ctx, uint32_t disposition, uint32_t options) {
    // (handle*, access, attributes*, io status*, ...)
    std::string path = host_path(ctx.r5.u32);
    uint32_t access = ctx.r4.u32;
    DWORD attributes = GetFileAttributesA(path.c_str());
    bool exists = attributes != INVALID_FILE_ATTRIBUTES;
    if ((options & 1) || (exists && (attributes & FILE_ATTRIBUTE_DIRECTORY))) {  // FILE_DIRECTORY_FILE
        if (!exists && (disposition == 2 || disposition == 3))
            exists = CreateDirectoryA(path.c_str(), nullptr) != 0;
        if (!exists) {
            log("io: no directory %s\n", path.c_str());
            io_status(ctx.r6.u32, kObjectPathNotFound, 0);
            ctx.r3.u64 = kObjectPathNotFound;
            return;
        }
        HANDLE h = CreateEvent(nullptr, TRUE, FALSE, nullptr);
        // The kernel's own handle on the directory, for listing it.
        std::string kernel_path = "\\??\\" + path;
        ANSI_STRING name;
        RtlInitAnsiString(&name, kernel_path.c_str());
        OBJECT_ATTRIBUTES attributes_of;
        InitializeObjectAttributes(&attributes_of, &name, OBJ_CASE_INSENSITIVE, nullptr);
        IO_STATUS_BLOCK status;
        HANDLE listing = nullptr;
        if (!NT_SUCCESS(NtOpenFile(&listing, FILE_LIST_DIRECTORY | SYNCHRONIZE, &attributes_of, &status,
                                   FILE_SHARE_READ | FILE_SHARE_WRITE,
                                   FILE_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT)))
            listing = nullptr;
        EnterCriticalSection(&directories_lock);
        directories.insert(h);
        file_paths[h] = path;
        HANDLE& slot = listings[h];
        if (slot)
            NtClose(slot);  // a closed directory's, its handle come back
        slot = listing;
        LeaveCriticalSection(&directories_lock);
        guest::store32(ctx.r3.u32, kernel::add_handle(h));
        io_status(ctx.r6.u32, kSuccess, 1);
        ctx.r3.u64 = kSuccess;
        return;
    }
    static const DWORD creation[] = {CREATE_ALWAYS, OPEN_EXISTING, CREATE_NEW, OPEN_ALWAYS, TRUNCATE_EXISTING, CREATE_ALWAYS};
    DWORD how = disposition < 6 ? creation[disposition] : OPEN_EXISTING;
    bool write = (access & (GENERIC_WRITE | FILE_WRITE_DATA | FILE_APPEND_DATA)) != 0;
    HANDLE h = CreateFileA(path.c_str(), write ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ,
                           FILE_SHARE_READ | (write ? 0 : FILE_SHARE_WRITE), nullptr, how, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        uint32_t status = exists ? kUnsuccessful : kObjectNameNotFound;
        log("io: cannot open %s (%lu)\n", path.c_str(), GetLastError());
        io_status(ctx.r6.u32, status, 0);
        ctx.r3.u64 = status;
        return;
    }
    // Each file once (the title opens its sound pack for every sound).
    static std::set<std::string> opened;
    EnterCriticalSection(&directories_lock);
    bool first = opened.insert(path).second;
    file_paths[h] = path;
    // A closed directory's handle comes back for a file: it is not a
    // directory any more (the title lists a folder, closes it and opens a
    // file in it, which then was taken for an empty directory).
    directories.erase(h);
    auto listing = listings.find(h);
    if (listing != listings.end()) {
        if (listing->second)
            NtClose(listing->second);
        listings.erase(listing);
    }
    LeaveCriticalSection(&directories_lock);
    if (first || kernel::trace_imports)
        log("io: open %s (%lu bytes)\n", path.c_str(), GetFileSize(h, nullptr));
    guest::store32(ctx.r3.u32, kernel::add_handle(h));
    io_status(ctx.r6.u32, kSuccess, exists ? 1 : 2);  // FILE_OPENED / FILE_CREATED
    ctx.r3.u64 = kSuccess;
}

bool seek(HANDLE h, uint32_t offset_pointer) {
    if (!offset_pointer)
        return true;
    int64_t offset = int64_t(guest::load64(offset_pointer));
    if (offset < 0)  // FILE_USE_FILE_POINTER_POSITION
        return true;
    LONG high = LONG(offset >> 32);
    return SetFilePointer(h, LONG(offset), &high, FILE_BEGIN) != INVALID_SET_FILE_POINTER || GetLastError() == NO_ERROR;
}

}  // namespace

IMPORT(NtCreateFile) {
    // (handle*, access, attributes*, io status*, allocation size*, file
    // attributes, share, disposition, options)
    open(ctx, ctx.r10.u32, uint32_t(kernel::arg(ctx, 8)));
}

IMPORT(NtOpenFile) {
    // (handle*, access, attributes*, io status*, share, options)
    open(ctx, 1, ctx.r8.u32);
}

// Scatter/gather I/O: the buffer is a list of pages, each a 64-bit
// FILE_SEGMENT_ELEMENT holding the page's address. Castle Crashers reads its
// packs this way.
static uint32_t scatter_gather(HANDLE h, uint32_t segments, uint32_t length, uint32_t offset_pointer, bool write) {
    if (!seek(h, offset_pointer))
        return ~0u;
    uint32_t done = 0;
    for (uint32_t i = 0; done < length; i++) {
        uint32_t page = uint32_t(guest::load64(segments + i * 8));
        uint32_t chunk = length - done < guest::kPage ? length - done : guest::kPage;
        guest::touch(page, chunk);
        DWORD moved = 0;
        BOOL ok = write ? WriteFile(h, guest::host(page), chunk, &moved, nullptr)
                        : ReadFile(h, guest::host(page), chunk, &moved, nullptr);
        if (!ok)
            return done ? done : ~0u;
        done += moved;
        if (moved < chunk)
            break;  // the end of the file
    }
    return done;
}

IMPORT(NtReadFileScatter) {
    // (handle, event, apc routine, apc context, io status*, segments, length, offset*)
    HANDLE h = kernel::host_handle(ctx.r3.u32);
    uint32_t read = scatter_gather(h, ctx.r8.u32, ctx.r9.u32, ctx.r10.u32, false);
    uint32_t status = read == ~0u ? kUnsuccessful : read == 0 && ctx.r9.u32 ? kEndOfFile : kSuccess;
    io_status(ctx.r7.u32, status, read == ~0u ? 0 : read);
    if (ctx.r4.u32)
        SetEvent(kernel::host_handle(ctx.r4.u32));
    ctx.r3.u64 = status;
}

IMPORT(NtWriteFileGather) {
    HANDLE h = kernel::host_handle(ctx.r3.u32);
    uint32_t written = scatter_gather(h, ctx.r8.u32, ctx.r9.u32, ctx.r10.u32, true);
    uint32_t status = written == ~0u ? kUnsuccessful : kSuccess;
    io_status(ctx.r7.u32, status, written == ~0u ? 0 : written);
    if (ctx.r4.u32)
        SetEvent(kernel::host_handle(ctx.r4.u32));
    ctx.r3.u64 = status;
}

IMPORT(NtReadFile) {
    // (handle, event, apc routine, apc context, io status*, buffer, length, offset*)
    HANDLE h = kernel::host_handle(ctx.r3.u32);
    uint32_t status_block = ctx.r7.u32;
    DWORD read = 0;
    guest::touch(ctx.r8.u32, ctx.r9.u32);  // the file system fills it: no faults
    if (!seek(h, ctx.r10.u32) || !ReadFile(h, guest::host(ctx.r8.u32), ctx.r9.u32, &read, nullptr)) {
        static int said;
        if (said++ < 20)
            log("io: read of %u bytes at %08X%s failed (%lu)\n", ctx.r9.u32, ctx.r8.u32,
                ctx.r10.u32 ? "" : " (no offset)", GetLastError());
        io_status(status_block, kUnsuccessful, 0);
        ctx.r3.u64 = kUnsuccessful;
        return;
    }
    if (read < ctx.r9.u32) {
        static int said;
        if (said++ < 20)
            log("io: read %u of %u bytes at offset %llu of %s\n", unsigned(read), ctx.r9.u32,
                ctx.r10.u32 ? (unsigned long long)guest::load64(ctx.r10.u32) : 0ull, path_of(h).c_str());
    }
    uint32_t status = read == 0 && ctx.r9.u32 ? kEndOfFile : kSuccess;
    // A sound's XMA data: the XMA decoder plays it from the converted sounds
    // (tools/convert_audio.py) by the file it came from, and needs only the
    // packets' headers of it, so its memory goes back. 64 MB does not hold
    // every sound effect too.
    std::string path = read ? path_of(h) : std::string();
    if (path.size() > 4 && _stricmp(path.c_str() + path.size() - 4, ".xma") == 0) {
        uint32_t offset = SetFilePointer(h, 0, nullptr, FILE_CURRENT) - read;
        xma::sound_read(path.c_str(), ctx.r8.u32, offset, read);
        if (read >= 2 * guest::kPage) {
            xma::save_packet_headers(ctx.r8.u32, read);
            guest::release(ctx.r8.u32, read);
        }
    }
    io_status(status_block, status, read);
    if (ctx.r4.u32)
        SetEvent(kernel::host_handle(ctx.r4.u32));
    ctx.r3.u64 = status;
}

IMPORT(NtWriteFile) {
    HANDLE h = kernel::host_handle(ctx.r3.u32);
    DWORD written = 0;
    guest::touch(ctx.r8.u32, ctx.r9.u32);
    if (!seek(h, ctx.r10.u32) || !WriteFile(h, guest::host(ctx.r8.u32), ctx.r9.u32, &written, nullptr)) {
        io_status(ctx.r7.u32, kUnsuccessful, 0);
        ctx.r3.u64 = kUnsuccessful;
        return;
    }
    io_status(ctx.r7.u32, kSuccess, written);
    if (ctx.r4.u32)
        SetEvent(kernel::host_handle(ctx.r4.u32));
    ctx.r3.u64 = kSuccess;
}

IMPORT(NtFlushBuffersFile) {
    FlushFileBuffers(kernel::host_handle(ctx.r3.u32));
    io_status(ctx.r4.u32, kSuccess, 0);
    ctx.r3.u64 = kSuccess;
}

IMPORT(NtQueryFullAttributesFile) {
    // (attributes*, FILE_NETWORK_OPEN_INFORMATION*: four times, allocation
    // size, end of file, attributes)
    std::string path = host_path(ctx.r3.u32);
    DWORD attributes = GetFileAttributesA(path.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) {
        ctx.r3.u64 = kObjectNameNotFound;
        return;
    }
    uint64_t size = 0;
    if (!(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (f != INVALID_HANDLE_VALUE) {
            DWORD high = 0, low = GetFileSize(f, &high);
            size = (uint64_t(high) << 32) | low;
            CloseHandle(f);
        }
    }
    uint32_t info = ctx.r4.u32;
    for (uint32_t i = 0; i < 4; i++)
        guest::store64(info + i * 8, 0);
    guest::store64(info + 32, (size + 4095) & ~uint64_t(4095));
    guest::store64(info + 40, size);
    guest::store32(info + 48, (attributes & FILE_ATTRIBUTE_DIRECTORY) ? 0x10 : 0x80);  // directory / normal
    ctx.r3.u64 = kSuccess;
}

IMPORT(NtQueryDirectoryFile) {
    // (handle, event, apc routine, apc context, io status*, information,
    // length, mask*, restart: the 360's has no information class, it is
    // always FileDirectoryInformation): through the kernel's own handle on the
    // directory (libxapi's FindFirstFileA faults). FILE_DIRECTORY_INFORMATION
    // is laid out as the 360's (ANSI names), in the host's byte order.
    HANDLE h = kernel::host_handle(ctx.r3.u32);
    EnterCriticalSection(&directories_lock);
    auto it = listings.find(h);
    HANDLE listing = it == listings.end() ? nullptr : it->second;
    LeaveCriticalSection(&directories_lock);
    uint32_t status = kNoMoreFiles, written = 0;
    if (listing) {
        uint32_t mask = ctx.r10.u32;
        std::string pattern;
        ANSI_STRING name;
        if (mask)
            pattern.assign(reinterpret_cast<const char*>(guest::host(guest::load32(mask + 4))), guest::load16(mask));
        RtlInitAnsiString(&name, pattern.c_str());
        alignas(8) uint8_t entries[4096];
        IO_STATUS_BLOCK io;
        status = uint32_t(NtQueryDirectoryFile(listing, nullptr, nullptr, nullptr, &io, entries, sizeof(entries),
                                               FileDirectoryInformation, mask ? &name : nullptr,
                                               BOOLEAN(kernel::arg(ctx, 8) & 0xFF)));
        // Each entry the guest's buffer holds: 64 bytes, then the name; the
        // next at an eight-byte boundary.
        uint32_t out = ctx.r8.u32, room = ctx.r9.u32, previous = 0;
        for (uint32_t at = 0; NT_SUCCESS(status);) {
            auto* e = reinterpret_cast<FILE_DIRECTORY_INFORMATION*>(entries + at);
            uint32_t size = 64 + e->FileNameLength;
            if (written + size > room)
                break;
            uint32_t to = out + written;
            if (written)
                guest::store32(out + previous, written - previous);  // the previous entry's NextEntryOffset
            guest::store32(to, 0);
            guest::store32(to + 4, e->FileIndex);
            const LARGE_INTEGER* times[6] = {&e->CreationTime, &e->LastAccessTime, &e->LastWriteTime,
                                             &e->ChangeTime, &e->EndOfFile, &e->AllocationSize};
            for (int i = 0; i < 6; i++)
                guest::store64(to + 8 + i * 8, uint64_t(times[i]->QuadPart));
            guest::store32(to + 56, e->FileAttributes);
            guest::store32(to + 60, e->FileNameLength);
            memcpy(guest::host(to + 64), e->FileName, e->FileNameLength);
            previous = written;
            written += (size + 7) & ~7u;
            if (!e->NextEntryOffset)
                break;
            at += e->NextEntryOffset;
        }
        if (NT_SUCCESS(status) && !written)
            status = 0x80000005;  // STATUS_BUFFER_OVERFLOW
    }
    io_status(ctx.r7.u32, status, written);
    if (ctx.r4.u32)
        SetEvent(kernel::host_handle(ctx.r4.u32));
    ctx.r3.u64 = status;
}

IMPORT(NtQueryInformationFile) {
    // (handle, io status*, information, length, class)
    HANDLE h = kernel::host_handle(ctx.r3.u32);
    uint32_t info = ctx.r5.u32;
    bool directory;
    EnterCriticalSection(&directories_lock);
    directory = directories.count(h) != 0;
    LeaveCriticalSection(&directories_lock);
    DWORD high = 0, low = directory ? 0 : GetFileSize(h, &high);
    uint64_t size = (uint64_t(high) << 32) | low;
    switch (ctx.r7.u32) {
    case 5:  // FileStandardInformation: allocation, end of file, links, delete pending, directory
        guest::store64(info, size);
        guest::store64(info + 8, size);
        guest::store32(info + 16, 1);
        guest::store8(info + 20, 0);
        guest::store8(info + 21, directory);
        io_status(ctx.r4.u32, kSuccess, 24);
        break;
    case 14: {  // FilePositionInformation
        LONG position_high = 0;
        DWORD position = SetFilePointer(h, 0, &position_high, FILE_CURRENT);
        guest::store64(info, (uint64_t(position_high) << 32) | position);
        io_status(ctx.r4.u32, kSuccess, 8);
        break;
    }
    case 34:  // FileNetworkOpenInformation: times (4 x 8), allocation, end of file, attributes
        memset(guest::host(info), 0, 56);
        guest::store64(info + 32, size);
        guest::store64(info + 40, size);
        guest::store32(info + 48, directory ? FILE_ATTRIBUTE_DIRECTORY : FILE_ATTRIBUTE_NORMAL);
        io_status(ctx.r4.u32, kSuccess, 56);
        break;
    default:
        log("io: NtQueryInformationFile class %u\n", ctx.r7.u32);
        ctx.r3.u64 = kNotImplemented;
        return;
    }
    ctx.r3.u64 = kSuccess;
}

IMPORT(NtSetInformationFile) {
    // (handle, io status*, information, length, class)
    HANDLE h = kernel::host_handle(ctx.r3.u32);
    switch (ctx.r7.u32) {
    case 14: {  // FilePositionInformation
        uint64_t position = guest::load64(ctx.r5.u32);
        LONG high = LONG(position >> 32);
        SetFilePointer(h, LONG(position), &high, FILE_BEGIN);
        if (kernel::trace_imports)
            log("io: seek %s to %llu\n", path_of(h).c_str(), (unsigned long long)position);
        break;
    }
    case 20: {  // FileEndOfFileInformation
        uint64_t end = guest::load64(ctx.r5.u32);
        LONG high = LONG(end >> 32);
        SetFilePointer(h, LONG(end), &high, FILE_BEGIN);
        SetEndOfFile(h);
        break;
    }
    default:
        log("io: NtSetInformationFile class %u\n", ctx.r7.u32);
        break;
    }
    io_status(ctx.r4.u32, kSuccess, 0);
    ctx.r3.u64 = kSuccess;
}

IMPORT(NtQueryVolumeInformationFile) {
    // (handle, io status*, information, length, class)
    uint32_t info = ctx.r5.u32;
    switch (ctx.r7.u32) {
    case 3:  // FileFsSizeInformation: total, available units, sectors per unit, bytes per sector
        guest::store64(info, 0x100000);
        guest::store64(info + 8, 0x80000);
        guest::store32(info + 16, 8);
        guest::store32(info + 20, 512);
        io_status(ctx.r4.u32, kSuccess, 24);
        break;
    default:
        memset(guest::host(info), 0, ctx.r6.u32);
        io_status(ctx.r4.u32, kSuccess, ctx.r6.u32);
        break;
    }
    ctx.r3.u64 = kSuccess;
}
