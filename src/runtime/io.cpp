// Files: the title's paths on the host's drives.
//
// game:\ and the DVD (\Device\Cdrom0) are D:\, the title's own folder on the
// Xbox, and update:\ (the title update) D:\update; anything else the title
// writes goes under T:\. libxapi's
// GetFileAttributesExA and FindFirstFileA fault, so only GetFileAttributesA,
// CreateFileA and handle calls are used. Directories open as handles the
// kernel only remembers (the title queries nothing in them).
#include "kernel.h"

#include <set>
#include <string>

namespace {

std::set<HANDLE> directories;
CRITICAL_SECTION directories_lock;
struct Init {
    Init() { InitializeCriticalSection(&directories_lock); }
} init_lock;

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
        EnterCriticalSection(&directories_lock);
        directories.insert(h);
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
    log("io: open %s\n", path.c_str());
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

IMPORT(NtReadFile) {
    // (handle, event, apc routine, apc context, io status*, buffer, length, offset*)
    HANDLE h = kernel::host_handle(ctx.r3.u32);
    uint32_t status_block = ctx.r7.u32;
    DWORD read = 0;
    if (!seek(h, ctx.r10.u32) || !ReadFile(h, guest::host(ctx.r8.u32), ctx.r9.u32, &read, nullptr)) {
        io_status(status_block, kUnsuccessful, 0);
        ctx.r3.u64 = kUnsuccessful;
        return;
    }
    uint32_t status = read == 0 && ctx.r9.u32 ? kEndOfFile : kSuccess;
    io_status(status_block, status, read);
    if (ctx.r4.u32)
        SetEvent(kernel::host_handle(ctx.r4.u32));
    ctx.r3.u64 = status;
}

IMPORT(NtWriteFile) {
    HANDLE h = kernel::host_handle(ctx.r3.u32);
    DWORD written = 0;
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
