// The Xbox 360 GPU's command processor, and the kernel's video calls (Vd*).
//
// The title's Direct3D 9 writes PM4 packets into a ring buffer and moves the
// ring's write pointer (the CP_RB_WPTR register) to have the GPU run them.
// The GPU's registers are plain guest memory here (memory.cpp), so a host
// thread watches the write pointer, runs the packets and writes the read
// pointer back where the title asked: its Direct3D sees its commands
// consumed. The control packets (indirect buffers, fences, waits, memory
// writes, interrupts) do what the GPU would. Draws run their vertex shader
// on the CPU (gpu/xenos, Xenia's shader interpreter) and go to the NV2A
// (gpu/renderer.cpp); resolves end the frame and swaps present it. Packet
// formats and semantics follow Xenia's src/xenia/gpu
// (pm4_command_processor_implement.h, xenos.h).
#include "kernel.h"

#include "gpu/renderer.h"

#include <vector>

namespace {

// Registers (dword index); the register file is at kRegisters + index * 4.
constexpr uint32_t kRegisters = 0x7FC80000;
constexpr uint32_t kCpRbRptr = 0x01C4, kCpRbWptr = 0x01C5;
constexpr uint32_t kScratchUmsk = 0x01DC, kScratchAddr = 0x01DD, kScratchReg0 = 0x0578;
constexpr uint32_t kCoherStatusHost = 0x0A31;
constexpr uint32_t kWritebackStart = 0x0A04, kWritebackSize = 0x0A05;
constexpr uint32_t kVgtEventInitiator = 0x21F9;
constexpr uint32_t kRegisterCount = 0x8000;
constexpr uint32_t kD1ModeVCounter = 0x194C;

// Type-3 opcodes.
enum : uint32_t {
    kNop = 0x10, kRegRmw = 0x21, kDrawIndx = 0x22, kVizQuery = 0x23, kWaitForIdle = 0x26, kImLoad = 0x27,
    kImLoadImmediate = 0x2B, kSetConstant = 0x2D, kLoadAluConstant = 0x2F, kDrawIndx2 = 0x36,
    kIndirectBufferPfd = 0x37, kInvalidateState = 0x3B, kWaitRegMem = 0x3C, kMemWrite = 0x3D, kRegToMem = 0x3E,
    kIndirectBuffer = 0x3F, kCondWrite = 0x45, kEventWrite = 0x46, kMeInit = 0x48, kInterrupt = 0x54,
    kSetConstant2 = 0x55, kSetShaderConstants = 0x56, kEventWriteShd = 0x58, kEventWriteExt = 0x5A,
    kEventWriteZpd = 0x5B, kContextUpdate = 0x5E, kXeSwap = 0x64,
};
constexpr uint32_t kSwapSignature = 0x53574150;  // 'SWAP'

uint32_t ring, ring_dwords, rptr, rptr_writeback;
uint32_t interrupt_callback, interrupt_data;
GuestThread* interrupt_thread;
CRITICAL_SECTION interrupt_lock;
uint32_t frames;
// What EVENT_WRITE_SHD writes when asked for a counter: advanced by every
// vblank and swap (Xenia's command processor counter_).
volatile uint32_t counter;
volatile bool started;

// Per-second counts for the log.
uint32_t opcode_counts[128], draws, indices, shader_loads, constants_set;
// Cycles (rdtsc) since the last statistics line.
uint64_t shade_cycles, submit_cycles, busy_cycles, last_tsc, opcode_cycles[128];
// Statistics: draws, vertices and shading cycles by vertex shader.
struct ShaderUse {
    uint32_t hash, draws, vertices;
    uint64_t cycles;
} shader_uses[8];
// WAIT_REG_MEM polls that had to sleep (statistics).
uint32_t waits;

uint32_t reg(uint32_t index) { return guest::load32(kRegisters + index * 4); }
void set_reg(uint32_t index, uint32_t value) { guest::store32(kRegisters + index * 4, value); }

// GPU (physical) addresses: guest memory is mapped at 0x80000000 + physical.
uint32_t translate(uint32_t address) { return address < 0x20000000 ? 0x80000000 | address : address; }

// The GPU as the renderer and the shader interpreter see it.
struct State : gpu::GpuState {
    uint32_t reg(uint32_t index) const override { return ::reg(index); }
    const uint8_t* memory(uint32_t address) const override { return guest::host(translate(address)); }
} state;

// A 32-bit value to or from memory in the endianness a GPU address's low
// bits name (0 none, 1 8-in-16, 2 8-in-32, 3 16-in-32), as raw bytes.
uint32_t gpu_swap(uint32_t value, uint32_t endian) {
    switch (endian & 3) {
    case 1: return ((value & 0x00FF00FF) << 8) | ((value >> 8) & 0x00FF00FF);
    case 2: return __builtin_bswap32(value);
    case 3: return (value << 16) | (value >> 16);
    default: return value;
    }
}

void memory_write(uint32_t address, uint32_t value) {
    uint32_t v = gpu_swap(value, address & 3);
    __builtin_memcpy(guest::host(translate(address & ~3u)), &v, 4);
}

uint32_t memory_read(uint32_t address) {
    uint32_t v;
    __builtin_memcpy(&v, guest::host(translate(address & ~3u)), 4);
    return gpu_swap(v, address & 3);
}

void write_register(uint32_t index, uint32_t value) {
    if (index >= kRegisterCount)
        return;
    set_reg(index, value);
    if (index >= kScratchReg0 && index < kScratchReg0 + 8) {
        // Scratch registers the title enabled are written back to memory:
        // its fences.
        uint32_t n = index - kScratchReg0;
        if (reg(kScratchUmsk) & (1u << n))
            guest::store32(translate(reg(kScratchAddr) + n * 4), value);
    } else if (index == kCoherStatusHost) {
        set_reg(index, value | 0x80000000);
    }
}

void dispatch_interrupt(uint32_t source, uint32_t cpu) {
    if (!interrupt_callback)
        return;
    EnterCriticalSection(&interrupt_lock);
    GuestThread* previous = current_thread();
    set_current_thread(interrupt_thread);
    PPCContext& ctx = interrupt_thread->ctx;
    guest::store8(interrupt_thread->pcr + 0x10C, uint8_t(cpu));
    ctx.r1.u64 = interrupt_thread->stack + interrupt_thread->stack_size - 0x100;
    ctx.r3.u64 = source;
    ctx.r4.u64 = interrupt_data;
    guest::call(ctx, interrupt_callback);
    set_current_thread(previous);
    LeaveCriticalSection(&interrupt_lock);
}

bool match(uint32_t value, uint32_t ref, uint32_t function) {
    switch (function & 7) {
    case 0: return false;
    case 1: return value < ref;
    case 2: return value <= ref;
    case 3: return value == ref;
    case 4: return value != ref;
    case 5: return value >= ref;
    case 6: return value > ref;
    default: return true;
    }
}

// --- what the title draws, for working out how to draw it -------------------

// The frame whose draws are logged in full (about ten seconds in).
constexpr uint32_t kCaptureFrame = 600;
uint32_t active_vs, active_ps;  // hashes of the loaded shaders
std::vector<uint32_t> vs_code;   // the loaded vertex shader, for the interpreter

uint32_t fnv(const uint32_t* dwords, uint32_t count) {
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < count; i++)
        hash = (hash ^ dwords[i]) * 16777619u;
    return hash;
}

// Logs a shader's microcode the first time it is loaded (for an offline
// disassembler), and remembers which shaders are loaded.
void note_shader(bool pixel, const uint32_t* code, uint32_t size) {
    static uint32_t seen[64];
    static uint32_t seen_count;
    uint32_t hash = fnv(code, size);
    (pixel ? active_ps : active_vs) = hash;
    if (!pixel)
        vs_code.assign(code, code + size);
    for (uint32_t i = 0; i < seen_count; i++)
        if (seen[i] == hash)
            return;
    if (seen_count < 64)
        seen[seen_count++] = hash;
    log("shader: %s %08X %u dwords\n", pixel ? "ps" : "vs", hash, size);
    for (uint32_t i = 0; i < size; i += 8) {
        char line[128];
        int n = snprintf(line, sizeof(line), "shader: %08X %03X:", hash, i);
        for (uint32_t k = i; k < i + 8 && k < size; k++)
            n += snprintf(line + n, sizeof(line) - n, " %08X", code[k]);
        log("%s\n", line);
    }
}

void capture_draw(uint32_t opcode, uint32_t initiator, uint32_t index_base, uint32_t index_size) {
    // VGT_DRAW_INITIATOR: primitive type (0-5), source select (6-7: 0
    // indices in memory, 1 in the packet, 2 none), index size (11), count.
    log("draw: %s prim %u source %u index32 %u count %u indices %08X/%08X vs %08X ps %08X\n",
        opcode == kDrawIndx ? "DRAW_INDX" : "DRAW_INDX_2", initiator & 0x3F, (initiator >> 6) & 3,
        (initiator >> 11) & 1, initiator >> 16, index_base, index_size, active_vs, active_ps);
    static const struct {
        uint32_t index;
        const char* name;
    } regs[] = {{0x2000, "RB_SURFACE_INFO"}, {0x2001, "RB_COLOR_INFO"}, {0x2002, "RB_DEPTH_INFO"},
                {0x2081, "SCISSOR_TL"}, {0x2082, "SCISSOR_BR"}, {0x2104, "RB_COLOR_MASK"},
                {0x210F, "VPORT_XSCALE"}, {0x2110, "VPORT_XOFFSET"}, {0x2111, "VPORT_YSCALE"},
                {0x2112, "VPORT_YOFFSET"}, {0x2180, "SQ_PROGRAM_CNTL"}, {0x2181, "SQ_CONTEXT_MISC"},
                {0x2200, "RB_DEPTHCONTROL"}, {0x2201, "RB_BLENDCONTROL0"}, {0x2202, "RB_COLORCONTROL"},
                {0x2205, "PA_SU_SC_MODE_CNTL"}, {0x2206, "PA_CL_VTE_CNTL"}, {0x2208, "RB_MODECONTROL"},
                {0x2302, "PA_SU_VTX_CNTL"}, {0x2307, "SQ_VS_CONST"}, {0x2308, "SQ_PS_CONST"},
                {0x2318, "RB_COPY_CONTROL"}, {0x2319, "RB_COPY_DEST_BASE"}, {0x231A, "RB_COPY_DEST_PITCH"},
                {0x231B, "RB_COPY_DEST_INFO"}};
    char line[768];
    int n = 0;
    for (const auto& r : regs) {
        n += snprintf(line + n, sizeof(line) - n, " %s=%08X", r.name, reg(r.index));
        if (n > 400) {
            log("draw: %s\n", line);
            n = 0;
        }
    }
    if (n)
        log("draw: %s\n", line);
    // Fetch constants (texture: 6 dwords per slot; vertex: 2 per slot, in
    // the same registers) and the first ALU constants of each stage.
    for (uint32_t i = 0; i < 48; i += 8) {
        n = snprintf(line, sizeof(line), "draw: fetch %02X:", i);
        for (uint32_t k = i; k < i + 8; k++)
            n += snprintf(line + n, sizeof(line) - n, " %08X", reg(0x4800 + k));
        log("%s\n", line);
    }
    static const uint32_t stages[] = {0x4000u, 0x4400u};
    for (uint32_t base : stages) {
        for (uint32_t c = 0; c < 8; c++) {
            float v[4];
            for (int k = 0; k < 4; k++) {
                uint32_t bits = reg(base + c * 4 + k);
                memcpy(&v[k], &bits, 4);
            }
            log("draw: %s c%u = %g %g %g %g\n", base == 0x4000 ? "vs" : "ps", c, v[0], v[1], v[2], v[3]);
        }
    }
    // The first vertices of vertex fetch constant 0 (address bits 2-31), when
    // it is one (type 3; a texture's is 2) and its memory is there.
    uint32_t vf = reg(0x4800);
    uint32_t vertices = translate(vf & ~3u);
    uint32_t start, size;
    bool committed = false;
    if ((vf & 3) != 3 || !guest::query(vertices, start, size, committed) || !committed)
        return;
    n = snprintf(line, sizeof(line), "draw: vf0 data %08X:", vertices);
    for (uint32_t k = 0; k < 16; k++)
        n += snprintf(line + n, sizeof(line) - n, " %08X", guest::load32(vertices + k * 4));
    log("%s\n", line);
}

// A draw: its indices, its vertices through the vertex shader, then the NV2A
// (or, with the render backend in copy mode, a resolve).
void draw(uint32_t initiator, uint32_t index_base, uint32_t index_size, const uint32_t* inline_indices) {
    constexpr uint32_t kRbModeControl = 0x2208, kVgtIndxOffset = 0x2102;
    if ((reg(kRbModeControl) & 7) == 6) {  // kCopy
        gpu::renderer::resolve(state);
        return;
    }
    uint32_t primitive = initiator & 0x3F, source = (initiator >> 6) & 3, count = initiator >> 16;
    bool index32 = (initiator >> 11) & 1;
    if (!count || vs_code.empty())
        return;
    static std::vector<uint32_t> index_list;
    index_list.resize(count);
    uint32_t offset = reg(kVgtIndxOffset);
    if (source == 2) {  // auto: 0, 1, 2...
        for (uint32_t i = 0; i < count; i++)
            index_list[i] = i;
    } else {
        // In memory (DRAW_INDX) or in the packet (DRAW_INDX_2), 16 or 32 bits,
        // in the endianness of the index size dword's top bits.
        uint32_t endian = source == 0 ? index_size >> 30 : 1;
        const uint8_t* data = source == 0 ? guest::host(translate(index_base & ~3u))
                                          : reinterpret_cast<const uint8_t*>(inline_indices);
        for (uint32_t i = 0; i < count; i++) {
            uint32_t v;
            if (index32) {
                __builtin_memcpy(&v, data + i * 4, 4);
                v = gpu_swap(v, endian);
            } else {
                uint16_t h;
                __builtin_memcpy(&h, data + i * 2, 2);
                v = (endian == 1 || endian == 2) ? __builtin_bswap16(h) : h;
            }
            index_list[i] = v + offset;
        }
    }
    static std::vector<gpu::ShadedVertex> shaded;
    shaded.resize(count);
    uint64_t t0 = __builtin_ia32_rdtsc();
    if (!gpu::shade_native(state, active_vs, vs_code.data(), uint32_t(vs_code.size()), index_list.data(), count,
                           shaded.data()))
        for (uint32_t i = 0; i < count; i++)
            gpu::shade_vertex(state, vs_code.data(), uint32_t(vs_code.size()), index_list[i], shaded[i]);
    uint64_t t1 = __builtin_ia32_rdtsc();
    for (ShaderUse& u : shader_uses) {
        if (u.hash == active_vs || !u.hash) {
            u.hash = active_vs;
            u.draws++;
            u.vertices += count;
            u.cycles += t1 - t0;
            break;
        }
    }
    gpu::renderer::draw(state, primitive, shaded.data(), count, active_ps);
    shade_cycles += t1 - t0;
    submit_cycles += __builtin_ia32_rdtsc() - t1;
}

// Runs the packets in count dwords from address (a buffer that does not
// wrap) or from the ring at index (wrapping).
class Reader {
public:
    Reader(uint32_t address, uint32_t count, uint32_t wrap) : base_(address), count_(count), wrap_(wrap) {}
    uint32_t next() {
        uint32_t v = guest::load32(base_ + (pos_ % wrap_) * 4);
        pos_++;
        count_--;
        return v;
    }
    void skip(uint32_t n) {
        pos_ += n;
        count_ -= n;
    }
    uint32_t left() const { return count_; }
    void start_at(uint32_t index) { pos_ = index; }

private:
    uint32_t base_, count_, wrap_, pos_ = 0;
};

void execute(Reader& r);

void execute_type3(Reader& r, uint32_t packet) {
    uint32_t opcode = (packet >> 8) & 0x7F;
    uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
    if (count > r.left()) {
        log("gpu: packet %08X wants %u dwords, %u left\n", packet, count, r.left());
        r.skip(r.left());
        return;
    }
    opcode_counts[opcode]++;
    // Cycles per packet type (an indirect buffer's include its packets').
    struct Timer {
        uint32_t opcode;
        uint64_t start = __builtin_ia32_rdtsc();
        ~Timer() { opcode_cycles[opcode] += __builtin_ia32_rdtsc() - start; }
    } timer{opcode};
    switch (opcode) {
    case kIndirectBuffer:
    case kIndirectBufferPfd: {
        uint32_t address = translate(r.next());
        uint32_t length = r.next() & 0xFFFFF;
        Reader inner(address, length, 0xFFFFFFFF);
        execute(inner);
        break;
    }
    case kWaitRegMem: {
        uint32_t info = r.next(), poll = r.next(), ref = r.next(), mask = r.next();
        r.next();  // wait interval
        for (int tries = 0;; tries++) {
            // Memory is always coherent here: a coherency request is done
            // as soon as it is waited for.
            if (!(info & 0x10) && (poll & 0x7FFF) == kCoherStatusHost)
                set_reg(kCoherStatusHost, reg(kCoherStatusHost) & 0x7FFFFFFF);
            uint32_t value = (info & 0x10) ? memory_read(poll) : reg(poll & 0x7FFF);
            if (match(value & mask, ref, info))
                break;
            if (tries == 2000) {
                log("gpu: WAIT_REG_MEM on %s %08X never matched (%08X & %08X vs %08X)\n",
                    (info & 0x10) ? "memory" : "register", poll, value, mask, ref);
                break;
            }
            waits++;
            Sleep(1);
        }
        break;
    }
    case kRegRmw: {
        uint32_t info = r.next(), and_mask = r.next(), or_mask = r.next();
        uint32_t value = reg(info & 0x1FFF);
        value &= (info >> 31) ? reg(and_mask & 0x1FFF) : and_mask;
        value |= ((info >> 30) & 1) ? reg(or_mask & 0x1FFF) : or_mask;
        write_register(info & 0x1FFF, value);
        break;
    }
    case kRegToMem: {
        uint32_t index = r.next(), address = r.next();
        memory_write(address, reg(index & 0x7FFF));
        break;
    }
    case kMemWrite: {
        uint32_t address = r.next();
        for (uint32_t i = 1; i < count; i++, address += 4)
            memory_write(address, r.next());
        break;
    }
    case kCondWrite: {
        uint32_t info = r.next(), poll = r.next(), ref = r.next(), mask = r.next(), target = r.next(), data = r.next();
        uint32_t value = (info & 0x10) ? memory_read(poll) : reg(poll & 0x7FFF);
        if (match(value & mask, ref, info)) {
            if (info & 0x100)
                memory_write(target, data);
            else
                write_register(target, data);
        }
        break;
    }
    case kEventWrite:
        set_reg(kVgtEventInitiator, r.next() & 0x3F);
        r.skip(count - 1);
        break;
    case kEventWriteShd: {
        uint32_t initiator = r.next(), address = r.next(), value = r.next();
        set_reg(kVgtEventInitiator, initiator & 0x3F);
        if (initiator >> 31)
            value = counter;
        uint32_t target = address & ~3u;
        // Addresses past physical memory are in the writeback window.
        if (target > 0x1FFFFFFF && reg(kWritebackStart) && target - reg(kWritebackStart) < reg(kWritebackSize))
            target = 0x7F000000 + (target - reg(kWritebackStart));
        uint32_t v = gpu_swap(value, address & 3);
        __builtin_memcpy(guest::host(translate(target)), &v, 4);
        break;
    }
    case kEventWriteExt: {
        // A draw's screen extents: report the whole screen.
        uint32_t initiator = r.next(), address = r.next();
        set_reg(kVgtEventInitiator, initiator & 0x3F);
        static const uint16_t extents[6] = {0, 8192 >> 3, 0, 8192 >> 3, 0, 1};
        for (int i = 0; i < 6; i++)
            guest::store16(translate(address & ~3u) + i * 2, extents[i]);
        break;
    }
    case kEventWriteZpd:
        set_reg(kVgtEventInitiator, r.next() & 0x3F);
        break;
    case kInterrupt: {
        uint32_t cpus = r.next();
        for (uint32_t n = 0; n < 6; n++)
            if (cpus & (1u << n))
                dispatch_interrupt(1, n);
        break;
    }
    case kXeSwap: {
        uint32_t magic = r.next();
        uint32_t front = r.next(), width = r.next(), height = r.next();
        r.skip(count - 4);
        if (magic != kSwapSignature)
            log("gpu: swap packet without its signature\n");
        if (frames < 3)
            log("gpu: swap %u: front buffer %08X %ux%u\n", frames, front, width, height);
        gpu::renderer::present();
        frames++;
        counter = counter + 1;
        break;
    }
    case kSetConstant: {
        // offset_type: the register block (0 ALU, 1 fetch, 2 bool, 3 loop, 4 registers)
        uint32_t offset_type = r.next();
        uint32_t index = offset_type & 0x7FF;
        static const uint32_t bases[] = {0x4000, 0x4800, 0x4900, 0x4908, 0x2000};
        uint32_t type = (offset_type >> 16) & 0xFF;
        uint32_t first = (type < 5 ? bases[type] : 0x2000) + index;
        for (uint32_t i = 1; i < count; i++)
            write_register(first + i - 1, r.next());
        constants_set++;
        break;
    }
    case kSetConstant2: {
        uint32_t index = r.next() & 0xFFFF;
        for (uint32_t i = 1; i < count; i++)
            write_register(index + i - 1, r.next());
        constants_set++;
        break;
    }
    case kLoadAluConstant: {
        uint32_t address = translate(r.next() & 0x3FFFFFFF);
        uint32_t offset_type = r.next();
        uint32_t size = r.next() & 0xFFF;
        static const uint32_t bases[] = {0x4000, 0x4800, 0x4900, 0x4908, 0x2000};
        uint32_t type = (offset_type >> 16) & 0xFF;
        uint32_t first = (type < 5 ? bases[type] : 0x2000) + (offset_type & 0x7FF);
        for (uint32_t i = 0; i < size; i++)
            write_register(first + i, guest::load32(address + i * 4));
        constants_set++;
        break;
    }
    case kSetShaderConstants: {
        uint32_t offset_type = r.next();
        uint32_t first = offset_type & 0xFFFF;
        for (uint32_t i = 1; i < count; i++)
            write_register(first + i - 1, r.next());
        constants_set++;
        break;
    }
    case kImLoad: {
        // (type, address and size of the microcode)
        uint32_t address_type = r.next(), start_size = r.next();
        uint32_t size = start_size & 0xFFFF;
        uint32_t address = translate(address_type & ~3u);
        static uint32_t code[4096];
        for (uint32_t i = 0; i < size && i < 4096; i++)
            code[i] = guest::load32(address + i * 4);
        note_shader((address_type & 3) != 0, code, size < 4096 ? size : 4096);
        shader_loads++;
        break;
    }
    case kImLoadImmediate: {
        uint32_t type = r.next(), start_size = r.next();
        uint32_t size = start_size & 0xFFFF;
        static uint32_t code[4096];
        for (uint32_t i = 0; i < size; i++) {
            uint32_t v = r.next();
            if (i < 4096)
                code[i] = v;
        }
        r.skip(count - 2 - size);
        note_shader((type & 3) != 0, code, size < 4096 ? size : 4096);
        shader_loads++;
        break;
    }
    case kDrawIndx:
    case kDrawIndx2: {
        uint32_t viz = opcode == kDrawIndx ? r.next() : 0;
        (void)viz;
        uint32_t initiator = r.next();
        uint32_t index_count = initiator >> 16;
        uint32_t index_base = 0, index_size = 0;
        static uint32_t inline_indices[1024];
        if (opcode == kDrawIndx && count >= 4) {
            index_base = r.next();
            index_size = r.next();
            r.skip(count - 4);
        } else {
            // DRAW_INDX_2: the indices (if any) follow in the packet.
            uint32_t rest = count - (opcode == kDrawIndx ? 2 : 1);
            for (uint32_t i = 0; i < rest; i++) {
                uint32_t v = r.next();
                if (i < 1024)
                    inline_indices[i] = __builtin_bswap32(v);  // back to memory order
            }
        }
        if (frames == kCaptureFrame)
            capture_draw(opcode, initiator, index_base, index_size);
        draw(initiator, index_base, index_size, inline_indices);
        draws++;
        indices += index_count;
        break;
    }
    case kMeInit:
    case kNop:
    case kInvalidateState:
    case kVizQuery:
    case kContextUpdate:
    case kWaitForIdle:
    default:
        r.skip(count);
        break;
    }
}

void execute(Reader& r) {
    while (r.left()) {
        uint32_t packet = r.next();
        if (packet == 0 || packet == 0x0BADF00D || packet == 0xCDCDCDCD)
            continue;
        switch (packet >> 30) {
        case 0: {
            uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
            uint32_t index = packet & 0x7FFF;
            bool single = (packet >> 15) & 1;
            if (count > r.left()) {
                r.skip(r.left());
                return;
            }
            for (uint32_t i = 0; i < count; i++)
                write_register(single ? index : index + i, r.next());
            break;
        }
        case 1: {
            uint32_t a = r.next(), b = r.next();
            write_register(packet & 0x7FF, a);
            write_register((packet >> 11) & 0x7FF, b);
            break;
        }
        case 2:
            break;
        case 3:
            execute_type3(r, packet);
            break;
        }
    }
}

DWORD WINAPI gpu_thread(void*) {
    // The NV2A device belongs to this thread: everything that draws runs here.
    if (!gpu::renderer::init())
        log("gpu: no Direct3D device, nothing will be drawn\n");
    DWORD last_log = GetTickCount();
    uint32_t last_frames = 0;
    for (;;) {
        uint32_t wptr = reg(kCpRbWptr) % (ring_dwords ? ring_dwords : 1);
        if (ring && wptr != rptr) {
            uint32_t count = (wptr + ring_dwords - rptr) % ring_dwords;
            Reader r(ring, count, ring_dwords);
            r.start_at(rptr);
            uint64_t t0 = __builtin_ia32_rdtsc();
            execute(r);
            busy_cycles += __builtin_ia32_rdtsc() - t0;
            rptr = wptr;
            set_reg(kCpRbRptr, rptr);
            if (rptr_writeback)
                guest::store32(rptr_writeback, rptr);
        } else {
            Sleep(1);  // this thread runs above the title's: never just yield
        }
        DWORD now = GetTickCount();
        // Every few seconds, one short line: the serial port is slow, and
        // DbgPrint waits for it on this (the highest-priority) thread.
        constexpr DWORD kStatsMs = 5000;
        if (now - last_log >= kStatsMs) {
            uint64_t tsc = __builtin_ia32_rdtsc(), elapsed = tsc - last_tsc;
            static uint64_t last_idle;
            uint64_t idle = kernel::idle_cycles();
            if (elapsed && frames != last_frames) {
                // The CPU's time: idle, then this thread's: vertex shading,
                // NV2A submission, the rest of the packets (mostly waits).
                unsigned seconds = kStatsMs / 1000;
                static uint32_t last_calls;
                uint32_t calls = gpu::renderer::draw_calls();
                log("gpu: %u fps, %u draws/s (%u NV2A), %u indices/s; idle %u%%, GPU thread %u%% (shading %u, submitting %u, "
                    "packets %u), %u wait sleeps/s\n",
                    (frames - last_frames) / seconds, draws / seconds, (calls - last_calls) / seconds,
                    indices / seconds,
                    unsigned((idle - last_idle) * 100 / elapsed), unsigned(busy_cycles * 100 / elapsed),
                    unsigned(shade_cycles * 100 / elapsed), unsigned(submit_cycles * 100 / elapsed),
                    unsigned((busy_cycles - shade_cycles - submit_cycles) * 100 / elapsed), waits / seconds);
                last_calls = calls;
            }
            waits = 0;
            last_idle = idle;
            last_tsc = tsc;
            memset(opcode_cycles, 0, sizeof(opcode_cycles));
            memset(opcode_counts, 0, sizeof(opcode_counts));
            shade_cycles = submit_cycles = busy_cycles = 0;
            char line[384];
            int n = snprintf(line, sizeof(line), "gpu: by vertex shader (draws, vertices, ms):");
            for (ShaderUse& u : shader_uses) {
                if (u.hash)
                    n += snprintf(line + n, sizeof(line) - n, " %08X %u %u %u", u.hash, u.draws, u.vertices,
                                  unsigned(u.cycles / 733000));
                u.draws = u.vertices = 0;
                u.cycles = 0;
            }
            log("%s\n", line);
            draws = indices = constants_set = 0;
            last_frames = frames;
            last_log = now;
        }
    }
}

DWORD WINAPI vblank_thread(void*) {
    for (;;) {
        Sleep(16);
        set_reg(kD1ModeVCounter, reg(kD1ModeVCounter) + 480);
        counter = counter + 1;
        dispatch_interrupt(0, 2);
    }
}

void start() {
    if (started)
        return;
    started = true;
    InitializeCriticalSection(&interrupt_lock);
    // Registers the title reads that the hardware keeps: the interrupt
    // handler checks the status for a vblank (Xenia's ReadRegister).
    set_reg(0x0F00, 0x08100748);         // RB_EDRAM_TIMING
    set_reg(0x0F01, 0x0000200E);         // RB_BC_CONTROL
    set_reg(0x1951, 1);                  // D1MODE_VBLANK_VLINE_STATUS: vblank
    set_reg(0x1961, (640u << 16) | 480);  // D1MODE_VIEWPORT_SIZE
    interrupt_thread = guest_thread_state(0x10000, 0);
    kernel::start_runtime_thread(gpu_thread, nullptr, 0x40000);
}

}  // namespace

IMPORT(VdInitializeRingBuffer) {
    // (ring, log2 of its size in quadwords)
    start();
    ring = translate(ctx.r3.u32);
    uint32_t bytes = 1u << (ctx.r4.u32 + 3);
    ring_dwords = bytes / 4;
    rptr = 0;
    memset(guest::host(ring), 0, bytes);
    log("gpu: ring buffer at %08X, %u KB\n", ring, bytes >> 10);
}

IMPORT(VdEnableRingBufferRPtrWriteBack) {
    rptr_writeback = translate(ctx.r3.u32);
}

IMPORT(VdSetGraphicsInterruptCallback) {
    start();
    interrupt_callback = ctx.r3.u32;
    interrupt_data = ctx.r4.u32;
    static bool vblank;
    if (!vblank) {
        vblank = true;
        kernel::start_runtime_thread(vblank_thread, nullptr, 0x8000);
    }
}

IMPORT(VdSwap) {
    // (ring space, front buffer's fetch constant, ..., front buffer*, format*,
    // color space*, width*, height*): a swap packet for the command processor
    // in the 64 dwords the title reserved.
    uint32_t buffer = ctx.r3.u32;
    uint32_t fetch = ctx.r4.u32;
    uint32_t front = guest::load32(ctx.r8.u32);
    uint32_t width = guest::load32(uint32_t(kernel::arg(ctx, 8)));
    uint32_t height = guest::load32(uint32_t(kernel::arg(ctx, 9)));
    uint32_t d = 0;
    auto put = [&](uint32_t v) { guest::store32(buffer + 4 * d++, v); };
    put((5u << 16) | 0x4800);  // type 0: six fetch constant dwords
    for (int i = 0; i < 6; i++) {
        uint32_t v = guest::load32(fetch + i * 4);
        if (i == 1)  // the base address: physical
            v = (v & 0xFFF) | (((v >> 12) << 12) & 0x1FFFFFFF);
        put(v);
    }
    put((3u << 30) | ((4u - 1) << 16) | (kXeSwap << 8));
    put(kSwapSignature);
    put(front & 0x1FFFFFFF);
    put(width);
    put(height);
    while (d < 64)
        put(2u << 30);  // type 2: nothing
}

IMPORT(VdGetSystemCommandBuffer) {
    memset(guest::host(ctx.r3.u32), 0, 0x94);
    guest::store32(ctx.r3.u32, 0xBEEF0000);
    guest::store32(ctx.r4.u32, 0xBEEF0001);
}

IMPORT(VdQueryVideoMode) {
    // X_VIDEO_MODE: 640x480 progressive 4:3, 60 Hz, NTSC-M.
    uint32_t m = ctx.r3.u32;
    memset(guest::host(m), 0, 0x30);
    guest::store32(m + 0x00, 640);
    guest::store32(m + 0x04, 480);
    float refresh = 60.0f;
    uint32_t bits;
    memcpy(&bits, &refresh, 4);
    guest::store32(m + 0x14, bits);
    guest::store32(m + 0x18, 1);
    guest::store32(m + 0x1C, 0x8A);  // pixel rate
    guest::store32(m + 0x20, 3);     // widescreen flag: 4:3
}

IMPORT(VdQueryVideoFlags) { ctx.r3.u64 = 0; }
IMPORT(VdSetDisplayMode) { ctx.r3.u64 = 0; }
IMPORT(VdInitializeEngines) { ctx.r3.u64 = 1; }
IMPORT(VdShutdownEngines) {}
IMPORT(VdEnableDisableClockGating) { ctx.r3.u64 = 0; }
IMPORT(VdSetSystemCommandBufferGpuIdentifierAddress) {}
IMPORT(VdIsHSIOTrainingSucceeded) { ctx.r3.u64 = 1; }
IMPORT(VdRetrainEDRAM) { ctx.r3.u64 = 0; }
IMPORT(VdRetrainEDRAMWorker) { ctx.r3.u64 = 0; }
IMPORT(VdCallGraphicsNotificationRoutines) { ctx.r3.u64 = 0; }

IMPORT(VdPersistDisplay) {
    if (ctx.r4.u32)
        guest::store32(ctx.r4.u32, guest::alloc(64));
    ctx.r3.u64 = 1;
}

IMPORT(VdInitializeScalerCommandBuffer) {
    // Its 12th argument: dwords to fill with nothing, returned as the count.
    uint32_t dest = uint32_t(kernel::arg(ctx, 10)), count = uint32_t(kernel::arg(ctx, 11));
    for (uint32_t i = 0; i < count; i++)
        guest::store32(dest + i * 4, 0x80000000);
    ctx.r3.u64 = count;
}

IMPORT(VdGetCurrentDisplayGamma) {
    guest::store32(ctx.r3.u32, 2);
    float power = 2.22222233f;
    uint32_t bits;
    memcpy(&bits, &power, 4);
    guest::store32(ctx.r4.u32, bits);
}

IMPORT(VdGetCurrentDisplayInformation) {
    // X_DISPLAY_INFO: a 640x480 front buffer shown at 640x480, 60 Hz.
    uint32_t info = ctx.r3.u32;
    memset(guest::host(info), 0, 0x58);
    guest::store16(info + 0x00, 640);
    guest::store16(info + 0x02, 480);
    guest::store32(info + 0x08 + 0x08, 640);  // scaler source rect x2, y2
    guest::store32(info + 0x08 + 0x0C, 480);
    guest::store32(info + 0x08 + 0x10, 640);
    guest::store32(info + 0x08 + 0x14, 480);
    guest::store16(info + 0x48, 640);
    guest::store16(info + 0x4A, 480);
    float refresh = 60.0f;
    uint32_t bits;
    memcpy(&bits, &refresh, 4);
    guest::store32(info + 0x4C, bits);
    guest::store16(info + 0x56, 640);
}
