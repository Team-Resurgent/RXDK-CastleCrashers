// The network (xam.xex's NetDll_* calls) as an Xbox with no one to talk to.
//
// The title sets up its sockets at start even when it never plays online,
// and fails hard if that does not work, so the calls succeed: sockets are
// handles that accept what they are sent and never receive anything
// (WSAEWOULDBLOCK), and the title's own address is loopback. Every NetDll_
// call's first argument is the caller type. Follows Xenia's xam_net.cc.
#include "kernel.h"

namespace {

constexpr uint32_t kWouldBlock = 10035;  // WSAEWOULDBLOCK
volatile LONG next_socket = 0x1000;

// The thread's last error (KTHREAD + 0x160): what WSAGetLastError returns.
void set_last_error(uint32_t error) { guest::store32(current_thread()->kthread + 0x160, error); }

}  // namespace

IMPORT(NetDll_WSAStartup) {
    // (caller, version, WSADATA*)
    if (ctx.r5.u32) {
        memset(guest::host(ctx.r5.u32), 0, 0x190);
        guest::store16(ctx.r5.u32, uint16_t(ctx.r4.u32));
        guest::store16(ctx.r5.u32 + 2, 0x0202);
    }
    ctx.r3.u64 = 0;
}

IMPORT(NetDll_WSACleanup) {
    // Diagnosing: the title cleans up its network when it is quitting.
    static bool dumped;
    if (!dumped)
        dump_calls(current_thread(), 400);
    dumped = true;
    ctx.r3.u64 = 0;
}
IMPORT(NetDll_WSAGetLastError) { ctx.r3.u64 = guest::load32(current_thread()->kthread + 0x160); }
IMPORT(NetDll_XNetStartup) { ctx.r3.u64 = 0; }
IMPORT(NetDll_XNetCleanup) { ctx.r3.u64 = 0; }

IMPORT(NetDll_XNetGetTitleXnAddr) {
    // (caller, XNADDR*): loopback, a fixed (non-zero) MAC address
    uint32_t a = ctx.r4.u32;
    memset(guest::host(a), 0, 36);
    guest::store32(a, 0x7F000001);
    static const uint8_t mac[6] = {0x00, 0x22, 0x48, 0x58, 0x44, 0x4B};
    memcpy(guest::host(a + 10), mac, 6);
    ctx.r3.u64 = 4;  // XNET_GET_XNADDR_STATIC
}

IMPORT(NetDll_XNetGetEthernetLinkStatus) { ctx.r3.u64 = 0; }  // no cable
IMPORT(NetDll_XNetGetConnectStatus) { ctx.r3.u64 = 0; }
IMPORT(NetDll_XNetQosListen) { ctx.r3.u64 = 0x65B; }  // ERROR_FUNCTION_FAILED
IMPORT(NetDll_XNetQosLookup) { ctx.r3.u64 = 0x65B; }
IMPORT(NetDll_XNetQosRelease) { ctx.r3.u64 = 0; }
IMPORT(NetDll_XNetUnregisterInAddr) { ctx.r3.u64 = 0; }

IMPORT(NetDll_XNetInAddrToXnAddr) {
    // (caller, address, XNADDR*, key id*)
    if (ctx.r5.u32) {
        memset(guest::host(ctx.r5.u32), 0, 36);
        guest::store32(ctx.r5.u32, ctx.r4.u32);
    }
    ctx.r3.u64 = 0;
}

IMPORT(NetDll_XNetXnAddrToInAddr) {
    // (caller, XNADDR*, key id*, address*)
    guest::store32(ctx.r6.u32, guest::load32(ctx.r4.u32));
    ctx.r3.u64 = 0;
}

IMPORT(NetDll_socket) { ctx.r3.u64 = uint32_t(InterlockedIncrement(&next_socket)); }
IMPORT(NetDll_closesocket) { ctx.r3.u64 = 0; }
IMPORT(NetDll_bind) { ctx.r3.u64 = 0; }
IMPORT(NetDll_setsockopt) { ctx.r3.u64 = 0; }
IMPORT(NetDll_ioctlsocket) { ctx.r3.u64 = 0; }

IMPORT(NetDll_recv) {
    set_last_error(kWouldBlock);
    ctx.r3.u64 = uint32_t(-1);
}

IMPORT(NetDll_recvfrom) {
    set_last_error(kWouldBlock);
    ctx.r3.u64 = uint32_t(-1);
}

IMPORT(NetDll_sendto) {
    // (caller, socket, buffer, length, flags, to*, to length): all sent
    ctx.r3.u64 = ctx.r6.u32;
}
