// The kernel's printf family: DbgPrint, sprintf, _vsnprintf, vswprintf.
//
// Guest variadic arguments are 64-bit slots: r3..r10, then the caller's
// parameter save area (kernel::arg64), or for a va_list consecutive slots in
// guest memory. Doubles travel in those slots as their bits.
#include "kernel.h"

#include <cstdio>
#include <string>

namespace kernel {

int format(char* out, int size, uint32_t format, bool wide, uint64_t (*next)(void*, int), void* state) {
    std::string result;
    int index = 0;
    auto at = [&](uint32_t i) -> uint32_t { return wide ? guest::load16(format + i * 2) : guest::load8(format + i); };
    for (uint32_t i = 0; at(i); i++) {
        uint32_t c = at(i);
        if (c != '%') {
            result += char(c < 0x80 ? c : '?');
            continue;
        }
        // %[flags][width][.precision][length]conversion
        std::string spec = "%";
        c = at(++i);
        while (c && strchr("-+ #0", int(c))) {
            spec += char(c);
            c = at(++i);
        }
        if (c == '*') {
            spec += std::to_string(int32_t(next(state, index++)));
            c = at(++i);
        }
        while (c >= '0' && c <= '9') {
            spec += char(c);
            c = at(++i);
        }
        if (c == '.') {
            spec += '.';
            c = at(++i);
            if (c == '*') {
                spec += std::to_string(int32_t(next(state, index++)));
                c = at(++i);
            }
            while (c >= '0' && c <= '9') {
                spec += char(c);
                c = at(++i);
            }
        }
        bool is64 = false, narrow = false, long_ = false;
        for (;;) {
            if (c == 'l' && at(i + 1) == 'l') { is64 = true; i++; }
            else if (c == 'I' && at(i + 1) == '6' && at(i + 2) == '4') { is64 = true; i += 2; }
            else if (c == 'h') narrow = true;
            else if (c == 'l' || c == 'w') long_ = true;
            else break;
            c = at(++i);
        }
        if (!c)
            break;
        char piece[512];
        switch (c) {
        case 'd': case 'i': case 'u': case 'x': case 'X': case 'o': {
            uint64_t v = next(state, index++);
            spec += "ll";
            spec += char(c);
            if (!is64)
                v = (c == 'd' || c == 'i') ? uint64_t(int64_t(int32_t(uint32_t(v)))) : uint32_t(v);
            snprintf(piece, sizeof(piece), spec.c_str(), (long long)v);
            result += piece;
            break;
        }
        case 'p':
            snprintf(piece, sizeof(piece), "%08X", uint32_t(next(state, index++)));
            result += piece;
            break;
        case 'c': case 'C':
            result += char(next(state, index++));
            break;
        case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A': {
            uint64_t bits = next(state, index++);
            double d;
            memcpy(&d, &bits, 8);
            spec += char(c);
            snprintf(piece, sizeof(piece), spec.c_str(), d);
            result += piece;
            break;
        }
        case 's': case 'S': {
            uint32_t s = uint32_t(next(state, index++));
            // %s is the format's own width, %S the other one; h and l/w say.
            bool w = c == 's' ? (wide ? !narrow : long_) : (wide ? long_ : !narrow);
            std::string text;
            if (!s)
                text = "(null)";
            else if (w)
                for (uint32_t k = 0; guest::load16(s + k * 2); k++)
                    text += char(guest::load16(s + k * 2) < 0x80 ? guest::load16(s + k * 2) : '?');
            else
                text = reinterpret_cast<const char*>(guest::host(s));
            spec += 's';
            snprintf(piece, sizeof(piece), spec.c_str(), text.c_str());
            result += piece;
            break;
        }
        case 'n':
            next(state, index++);
            break;
        default:
            result += char(c);
            break;
        }
    }
    if (size > 0) {
        int n = int(result.size()) < size ? int(result.size()) : size - 1;
        memcpy(out, result.data(), n);
        out[n] = 0;
    }
    return int(result.size());
}

}  // namespace kernel

namespace {

struct Registers {
    PPCContext* ctx;
    int first;
};

uint64_t from_registers(void* state, int i) {
    Registers* r = static_cast<Registers*>(state);
    return kernel::arg64(*r->ctx, r->first + i);
}

uint64_t from_list(void* state, int i) { return guest::load64(*static_cast<uint32_t*>(state) + i * 8); }

// Writes a formatted string to guest memory, as narrow or UTF-16 characters;
// returns the length, or -1 when it did not fit (_vsnprintf's convention).
int64_t store(uint32_t buffer, uint32_t count, const char* text, int length, bool wide) {
    uint32_t n = uint32_t(length) < count ? uint32_t(length) : count;
    for (uint32_t i = 0; i < n; i++) {
        if (wide)
            guest::store16(buffer + i * 2, uint8_t(text[i]));
        else
            guest::store8(buffer + i, uint8_t(text[i]));
    }
    if (uint32_t(length) < count) {
        if (wide)
            guest::store16(buffer + length * 2, 0);
        else
            guest::store8(buffer + length, 0);
        return length;
    }
    return -1;
}

}  // namespace

IMPORT(DbgPrint) {
    char text[1024];
    Registers r{&ctx, 1};
    kernel::format(text, sizeof(text), ctx.r3.u32, false, from_registers, &r);
    log("%s", text);
    ctx.r3.u64 = 0;
}

IMPORT(sprintf) {
    static char text[4096];
    Registers r{&ctx, 2};
    int n = kernel::format(text, sizeof(text), ctx.r4.u32, false, from_registers, &r);
    ctx.r3.u64 = uint64_t(store(ctx.r3.u32, 0xFFFFFFFF, text, n, false));
}

IMPORT(_vsnprintf) {
    // (buffer, count, format, va_list)
    char text[4096];
    uint32_t list = ctx.r6.u32;
    int n = kernel::format(text, sizeof(text), ctx.r5.u32, false, from_list, &list);
    ctx.r3.u64 = uint64_t(store(ctx.r3.u32, ctx.r4.u32, text, n, false));
}

IMPORT(vswprintf) {
    // (buffer, format, va_list): wide in, wide out
    char text[4096];
    uint32_t list = ctx.r5.u32;
    int n = kernel::format(text, sizeof(text), ctx.r4.u32, true, from_list, &list);
    ctx.r3.u64 = uint64_t(store(ctx.r3.u32, 0xFFFFFFFF, text, n, true));
}
