// The NV2A side of the GPU (renderer.h).
//
// Register layouts and the 360's texture tiling follow Xenia's
// src/xenia/gpu (registers.h, xenos.h, texture_address.h).
#include "gpu/renderer.h"

#include <xtl.h>
#include <xgraphics.h>

#include <cstring>
#include <map>
#include <vector>

void log(const char* format, ...);

namespace gpu {
namespace {

constexpr UINT kWidth = 640, kHeight = 480;

IDirect3D8* d3d;
IDirect3DDevice8* device;

// After releasing a texture: a new one may get its address (set_texture).
void forget_textures();

// Frames presented: cached textures compare their guest data once a frame.
uint32_t frame_number;

// Registers (dword index).
enum : uint32_t {
    kPaScWindowOffset = 0x2080,
    kRbColorMask = 0x2104,
    kRbAlphaRef = 0x210E,
    kPaClVportXScale = 0x210F,
    kPaClVportXOffset = 0x2110,
    kPaClVportYScale = 0x2111,
    kPaClVportYOffset = 0x2112,
    kPaClVportZScale = 0x2113,
    kPaClVportZOffset = 0x2114,
    kRbBlendControl0 = 0x2201,
    kRbColorControl = 0x2202,
    kPaSuScModeCntl = 0x2205,
    kPaClVteCntl = 0x2206,
    kRbCopyControl = 0x2318,
    kRbColorClear = 0x231E,
    kFetchConstants = 0x4800,
};

float as_float(uint32_t bits) {
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

// --- textures ------------------------------------------------------------------

// The 360's tiled layout: a 32x32 macro tile of blocks, banks and pipes
// interleaved (Xenia texture_address.h Tiled2D).
int32_t tiled_2d(int32_t x, int32_t y, uint32_t pitch_aligned, unsigned bytes_per_block_log2) {
    int32_t outer = ((y >> 5) * int32_t(pitch_aligned >> 5) + (x >> 5)) << 6;
    int32_t inner = (((y >> 1) & 7) << 3) | (x & 7);
    int32_t bytes = (outer | inner) << bytes_per_block_log2;
    uint32_t bank = (y >> 4) & 1;
    uint32_t pipe = ((x >> 3) & 3) ^ (((y >> 3) & 1) << 1);
    return int32_t(((y & 1) << 4) | (pipe << 6) | (bank << 11)) | (bytes & 0xF) | (((bytes >> 4) & 1) << 5) |
           (((bytes >> 5) & 7) << 8) | (bytes >> 8 << 12);
}

// A 3D texture's: 32x16x4 macro tiles (Xenia texture_address.h Tiled3D).
int32_t tiled_3d(int32_t x, int32_t y, int32_t z, uint32_t pitch_aligned, uint32_t height_aligned,
                 unsigned bytes_per_block_log2) {
    int32_t outer = ((((z >> 2) * int32_t(height_aligned >> 4) + (y >> 4)) * int32_t(pitch_aligned >> 5)) + (x >> 5))
                    << 7;
    int32_t inner = ((z & 3) << 5) | (((y >> 1) & 3) << 3) | (x & 7);
    int32_t bytes = (outer | inner) << bytes_per_block_log2;
    uint32_t bank = ((y >> 3) ^ (z >> 2)) & 1;
    uint32_t pipe = ((x >> 3) & 3) ^ (bank << 1);
    return int32_t(((y & 1) << 4) | (pipe << 6) | (bank << 11)) | (bytes & 0xF) | (((bytes >> 4) & 1) << 5) |
           (((bytes >> 5) & 7) << 8) | (bytes >> 8 << 12);
}

void swap_block(uint8_t* p, uint32_t size, uint32_t endian) {
    for (uint32_t i = 0; i < size; i += 4) {
        uint32_t v;
        memcpy(&v, p + i, 4);
        switch (endian) {
        case 1: v = ((v & 0x00FF00FF) << 8) | ((v >> 8) & 0x00FF00FF); break;
        case 2: v = __builtin_bswap32(v); break;
        case 3: v = (v << 16) | (v >> 16); break;
        default: break;
        }
        memcpy(p + i, &v, 4);
    }
}

uint32_t next_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v)
        p <<= 1;
    return p;
}

struct Texture {
    IDirect3DTexture8* texture = nullptr;
    float u_scale = 1, v_scale = 1;  // the image within its power-of-two texture
    uint32_t hash = 0;               // of the guest data it was made from (sample_hash)
    uint32_t checked = ~0u;          // the frame the hash was last compared in
    uint32_t used = 0;               // the frame it was last drawn with (eviction)
};

struct TextureKey {
    uint32_t address, format_size, slice;
    bool red_alpha;
    bool operator<(const TextureKey& o) const {
        if (address != o.address)
            return address < o.address;
        if (format_size != o.format_size)
            return format_size < o.format_size;
        if (slice != o.slice)
            return slice < o.slice;
        return red_alpha < o.red_alpha;
    }
};

std::map<TextureKey, Texture> textures;

// --- decoding, for the textures fixed function cannot use as they are ----------

uint32_t rgb565(uint16_t c) {
    uint32_t r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;
    return ((r * 255 / 31) << 16) | ((g * 255 / 63) << 8) | (b * 255 / 31);
}

// The colours of a DXT colour block (8 bytes, little-endian after the byte
// order fix), alpha 255 (0 for DXT1's transparent entry).
void dxt_colors(const uint8_t* p, bool dxt1, uint32_t out[16]) {
    uint16_t c0 = uint16_t(p[0] | (p[1] << 8)), c1 = uint16_t(p[2] | (p[3] << 8));
    uint32_t a = rgb565(c0), b = rgb565(c1), palette[4];
    palette[0] = a | 0xFF000000;
    palette[1] = b | 0xFF000000;
    auto mix = [](uint32_t x, uint32_t y, int wx, int wy, int d) {
        uint32_t r = 0;
        for (int s = 0; s < 24; s += 8)
            r |= ((((x >> s) & 0xFF) * wx + ((y >> s) & 0xFF) * wy) / d) << s;
        return r;
    };
    if (!dxt1 || c0 > c1) {
        palette[2] = mix(a, b, 2, 1, 3) | 0xFF000000;
        palette[3] = mix(a, b, 1, 2, 3) | 0xFF000000;
    } else {
        palette[2] = mix(a, b, 1, 1, 2) | 0xFF000000;
        palette[3] = 0;
    }
    uint32_t indices = uint32_t(p[4] | (p[5] << 8) | (p[6] << 16) | (uint32_t(p[7]) << 24));
    for (int i = 0; i < 16; i++)
        out[i] = palette[(indices >> (i * 2)) & 3];
}

// A block (4x4 texels for DXT, 1 otherwise) as 0xAARRGGBB texels.
void decode_block(uint32_t format, const uint8_t* p, uint32_t out[16]) {
    switch (format) {
    case 18:
        dxt_colors(p, true, out);
        break;
    case 19:  // DXT2/3: explicit 4-bit alpha, then the colours
        dxt_colors(p + 8, false, out);
        for (int i = 0; i < 16; i++) {
            uint32_t a = (p[i / 2] >> ((i & 1) * 4)) & 0xF;
            out[i] = (out[i] & 0xFFFFFF) | ((a * 17) << 24);
        }
        break;
    case 20: {  // DXT4/5: interpolated alpha, then the colours
        dxt_colors(p + 8, false, out);
        uint32_t a0 = p[0], a1 = p[1], alphas[8] = {a0, a1};
        for (int i = 2; i < 8; i++)
            alphas[i] = a0 > a1 ? (a0 * (8 - i) + a1 * (i - 1)) / 7
                                : i < 6 ? (a0 * (6 - i) + a1 * (i - 1)) / 5 : (i == 6 ? 0 : 255);
        uint64_t bits = 0;
        for (int i = 0; i < 6; i++)
            bits |= uint64_t(p[2 + i]) << (8 * i);
        for (int i = 0; i < 16; i++)
            out[i] = (out[i] & 0xFFFFFF) | (alphas[(bits >> (3 * i)) & 7] << 24);
        break;
    }
    default:
        out[0] = 0xFFFFFFFF;
        break;
    }
}

// A texture fetch constant: the image it describes, in plain block order
// with the byte order fixed (blocks are 4x4 texels for DXT, 1 otherwise),
// padded to power-of-two dimensions.
struct Image {
    uint32_t format = 0, address = 0, width = 0, height = 0;
    uint32_t block = 1, bytes_per_block = 0;
    uint32_t pot_w = 0, pot_h = 0, pot_blocks_w = 0;
    D3DFORMAT d3d_format = D3DFMT_UNKNOWN;
    bool tiled = false;
    // Several images: a 3D texture's depth slices (volume) or a stacked 2D
    // texture's layers (array). One of them is read (slice).
    uint32_t depth = 1, slice = 0;
    bool volume = false;
    const uint8_t* data = nullptr;  // the guest data, all slices
    uint32_t data_bytes = 0;
    std::vector<uint8_t> linear;
};

// A cheap hash of an image's guest data (256 dwords spread over it): the
// title rewrites textures in place (fonts, reused memory), so a cached
// texture is remade when this changes.
uint32_t sample_hash(const Image& image) {
    if (!image.data || image.data_bytes < 4)
        return 0;
    uint32_t hash = 2166136261u, words = image.data_bytes / 4;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t v;
        memcpy(&v, image.data + size_t(uint64_t(i) * words / 256) * 4, 4);
        hash = (hash ^ v) * 16777619u;
    }
    return hash;
}

// The slice a fetch's third coordinate selects: normalized for volumes and
// arrays alike (the title's z are layer centres: 40.5 / 64).
uint32_t slice_for(const Image& image, float z) {
    int i = int(__builtin_floorf(z * float(image.depth)));
    return i < 0 ? 0 : i >= int(image.depth) ? image.depth - 1 : uint32_t(i);
}

bool fetch_image(const GpuState& gpu, uint32_t slot, Image& image, bool read = true) {
    uint32_t f[6];
    for (int i = 0; i < 6; i++)
        f[i] = gpu.reg(kFetchConstants + slot * 6 + i);
    if ((f[0] & 3) != 2)  // not a texture fetch constant
        return false;
    image.format = f[1] & 0x3F;
    uint32_t endian = (f[1] >> 6) & 3;
    image.address = (f[1] >> 12) << 12;
    uint32_t dimension = (f[5] >> 9) & 3;  // 1D, 2D or stacked, 3D, cube
    bool packed_mips = (f[5] >> 11) & 1;
    if (dimension == 2) {
        image.width = (f[2] & 0x7FF) + 1;
        image.height = ((f[2] >> 11) & 0x7FF) + 1;
        image.depth = ((f[2] >> 22) & 0x3FF) + 1;
        image.volume = true;
    } else {
        image.width = (f[2] & 0x1FFF) + 1;
        image.height = ((f[2] >> 13) & 0x1FFF) + 1;
        image.depth = ((f[1] >> 10) & 1) ? ((f[2] >> 26) & 0x3F) + 1 : 1;  // stacked
    }
    if (image.slice >= image.depth)
        image.slice = image.depth - 1;
    image.tiled = f[0] >> 31;
    uint32_t pitch = ((f[0] >> 22) & 0x1FF) << 5;  // in texels
    switch (image.format) {
    case 18: image.d3d_format = D3DFMT_DXT1; image.block = 4; image.bytes_per_block = 8; break;
    case 19: image.d3d_format = D3DFMT_DXT3; image.block = 4; image.bytes_per_block = 16; break;
    case 20: image.d3d_format = D3DFMT_DXT5; image.block = 4; image.bytes_per_block = 16; break;
    case 6: image.d3d_format = D3DFMT_A8R8G8B8; image.bytes_per_block = 4; break;
    case 15: image.d3d_format = D3DFMT_A4R4G4B4; image.bytes_per_block = 2; break;
    case 4: image.d3d_format = D3DFMT_R5G6B5; image.bytes_per_block = 2; break;
    case 3: image.d3d_format = D3DFMT_A1R5G5B5; image.bytes_per_block = 2; break;
    case 2: image.d3d_format = D3DFMT_A8; image.bytes_per_block = 1; break;
    default: return true;  // d3d_format stays unknown
    }
    uint32_t block = image.block, bytes_per_block = image.bytes_per_block;
    image.pot_w = next_pow2(image.width);
    image.pot_h = next_pow2(image.height);
    if (block == 4) {
        image.pot_w = image.pot_w < 4 ? 4 : image.pot_w;
        image.pot_h = image.pot_h < 4 ? 4 : image.pot_h;
    }
    image.pot_blocks_w = (image.pot_w + block - 1) / block;
    unsigned bpb_log2 = __builtin_ctz(bytes_per_block);
    uint32_t blocks_w = (image.width + block - 1) / block, blocks_h = (image.height + block - 1) / block;
    uint32_t pitch_blocks = ((pitch ? pitch : image.width) + block - 1) / block;
    uint32_t pitch_aligned = (pitch_blocks + 31) & ~31u;
    // Packed mips: a base level 16 or less on its short side sits in the
    // mip tail tile, 16 texels down (wider than tall) or across (Xenia
    // texture_util.cc GetPackedMipOffset).
    uint32_t packed_x = 0, packed_y = 0;
    if (packed_mips) {
        auto log2_ceil = [](uint32_t v) { return v <= 1 ? 0u : 32u - uint32_t(__builtin_clz(v - 1)); };
        uint32_t log2_w = log2_ceil(image.width), log2_h = log2_ceil(image.height);
        // 16 texels: 4 blocks of a compressed format (the laser, DXT3 64x16).
        if ((log2_w < log2_h ? log2_w : log2_h) <= 4) {
            if (log2_w > log2_h)
                packed_y = 16 / block;
            else
                packed_x = 16 / block;
        }
    }
    image.data = gpu.memory(image.address);
    image.data_bytes = (image.tiled ? pitch_aligned : pitch_blocks) * bytes_per_block *
                       ((blocks_h + packed_y + 31) & ~31u) * (image.volume ? (image.depth + 3) & ~3u : image.depth);
    if (!read)
        return true;
    image.linear.assign(size_t(image.pot_blocks_w) * ((image.pot_h + block - 1) / block) * bytes_per_block, 0);
    // Array layers follow each other, each a 2D image of 32-block-aligned
    // rows and height, 4 KB aligned (Xenia texture_util.cc).
    uint32_t height_aligned = (blocks_h + 31) & ~31u;
    uint32_t row_bytes = (image.tiled ? pitch_aligned : pitch_blocks) * bytes_per_block;
    uint32_t layer_bytes = (row_bytes * height_aligned + 4095) & ~4095u;
    const uint8_t* source = gpu.memory(image.address);
    if (!image.volume)
        source += size_t(image.slice) * layer_bytes;
    for (uint32_t y = 0; y < blocks_h; y++) {
        for (uint32_t x = 0; x < blocks_w; x++) {
            uint32_t offset;
            uint32_t sx = x + packed_x, sy = y + packed_y;
            if (image.volume)
                offset = image.tiled ? uint32_t(tiled_3d(int32_t(sx), int32_t(sy), int32_t(image.slice), pitch_aligned,
                                                         height_aligned, bpb_log2))
                                     : (image.slice * blocks_h + sy) * row_bytes + sx * bytes_per_block;
            else
                offset = image.tiled ? uint32_t(tiled_2d(int32_t(sx), int32_t(sy), pitch_aligned, bpb_log2))
                                     : sy * row_bytes + sx * bytes_per_block;
            uint8_t* to = &image.linear[(size_t(y) * image.pot_blocks_w + x) * bytes_per_block];
            memcpy(to, source + offset, bytes_per_block);
            if (bytes_per_block >= 4 || endian == 1)
                swap_block(to, bytes_per_block < 4 ? 4 : bytes_per_block, endian);
        }
    }
    return true;
}

// The image as 0xAARRGGBB texels, pot_w x pot_h (what the shader reads as
// x, y, z, w is red, green, blue, alpha).
std::vector<uint32_t> decode_argb(const Image& image) {
    std::vector<uint32_t> texels(size_t(image.pot_w) * image.pot_h, 0);
    uint32_t block = image.block, bytes_per_block = image.bytes_per_block;
    uint32_t blocks_w = (image.width + block - 1) / block, blocks_h = (image.height + block - 1) / block;
    for (uint32_t by = 0; by < blocks_h; by++) {
        for (uint32_t bx = 0; bx < blocks_w; bx++) {
            const uint8_t* p = &image.linear[(size_t(by) * image.pot_blocks_w + bx) * bytes_per_block];
            uint32_t block_texels[16];
            if (block == 4) {
                decode_block(image.format, p, block_texels);
            } else {
                uint32_t v = 0;
                memcpy(&v, p, bytes_per_block < 4 ? bytes_per_block : 4);
                // k_8: the one channel, X (low byte) and red alike
                block_texels[0] = image.format == 2 ? ((v & 0xFF) << 16) | (v & 0xFF) : v;
            }
            for (uint32_t y = 0; y < block; y++)
                for (uint32_t x = 0; x < block; x++) {
                    uint32_t tx = bx * block + x, ty = by * block + y;
                    if (tx < image.pot_w && ty < image.pot_h)
                        texels[size_t(ty) * image.pot_w + tx] = block_texels[y * block + x];
                }
        }
    }
    return texels;
}

// The channels a texture fetch gives a shader (x, y, z, w) from a decoded
// 0xAARRGGBB texel, through the fetch constant's swizzle.
struct Swizzle {
    int shift[4];  // bit position of each channel, -1: 0, -2: 1
};

Swizzle fetch_swizzle(const GpuState& gpu, uint32_t slot, const Image& image) {
    uint32_t swizzle = (gpu.reg(kFetchConstants + slot * 6 + 3) >> 1) & 0xFFF;
    // The components X Y Z W of the format, as decode_argb leaves them:
    // 8888 in memory order (X is the low byte), DXT as red green blue alpha.
    static const int kMemory[4] = {0, 8, 16, 24}, kRgba[4] = {16, 8, 0, 24};
    const int* component = image.block == 4 ? kRgba : kMemory;
    Swizzle s;
    for (int i = 0; i < 4; i++) {
        uint32_t c = (swizzle >> (3 * i)) & 7;
        s.shift[i] = c < 4 ? component[c] : c == 4 ? -1 : -2;
    }
    return s;
}

uint32_t channel_byte(uint32_t texel, int shift) { return shift >= 0 ? (texel >> shift) & 0xFF : shift == -1 ? 0 : 255; }

// 0xAARRGGBB texels as a new texture (nullptr if there is no memory).
IDirect3DTexture8* create_argb(const std::vector<uint32_t>& texels, uint32_t w, uint32_t h) {
    IDirect3DTexture8* texture = nullptr;
    if (FAILED(device->CreateTexture(w, h, 1, 0, D3DFMT_A8R8G8B8, 0, &texture))) {
        log("renderer: no memory for a %ux%u texture\n", w, h);
        return nullptr;
    }
    D3DLOCKED_RECT locked;
    if (SUCCEEDED(texture->LockRect(0, &locked, nullptr, 0))) {
        XGSwizzleRect(texels.data(), w * 4, nullptr, locked.pBits, w, h, nullptr, 4);
        texture->UnlockRect(0);
    }
    return texture;
}

// An image's blocks as a new texture of the given format (its own, or one
// with the same layout): compressed rows as they are, others swizzled.
IDirect3DTexture8* create_raw(const Image& image, D3DFORMAT format) {
    IDirect3DTexture8* texture = nullptr;
    if (FAILED(device->CreateTexture(image.pot_w, image.pot_h, 1, 0, format, 0, &texture))) {
        log("renderer: no memory for a %ux%u texture\n", image.pot_w, image.pot_h);
        return nullptr;
    }
    D3DLOCKED_RECT locked;
    if (SUCCEEDED(texture->LockRect(0, &locked, nullptr, 0))) {
        if (image.block == 4) {
            uint32_t row = image.pot_blocks_w * image.bytes_per_block;
            for (uint32_t y = 0; y < (image.pot_h + 3) / 4; y++)
                memcpy(static_cast<uint8_t*>(locked.pBits) + size_t(y) * locked.Pitch, &image.linear[size_t(y) * row],
                       row);
        } else {
            XGSwizzleRect(image.linear.data(), image.pot_w * image.bytes_per_block, nullptr, locked.pBits, image.pot_w,
                          image.pot_h, nullptr, image.bytes_per_block);
        }
        texture->UnlockRect(0);
    }
    return texture;
}

// Red and blue exchanged in every DXT block's two 565 colours. A DXT1
// block's mode is the colours' order (c0 > c1: four colours, else three and
// transparent): where the exchange reverses it, the colours trade places and
// the indices follow.
void swap_dxt_red_blue(Image& image) {
    auto swap565 = [](uint16_t c) { return uint16_t((c & 0x07E0) | (c >> 11) | ((c & 0x1F) << 11)); };
    uint32_t colour_offset = image.bytes_per_block == 16 ? 8 : 0;
    bool dxt1 = image.format == 18;
    for (size_t at = 0; at + image.bytes_per_block <= image.linear.size(); at += image.bytes_per_block) {
        uint8_t* p = &image.linear[at + colour_offset];
        uint16_t c0 = uint16_t(p[0] | (p[1] << 8)), c1 = uint16_t(p[2] | (p[3] << 8));
        uint16_t s0 = swap565(c0), s1 = swap565(c1);
        if (dxt1 && c0 != c1 && (c0 > c1) != (s0 > s1)) {
            uint16_t t = s0;
            s0 = s1;
            s1 = t;
            bool four = c0 > c1;
            uint32_t indices = uint32_t(p[4] | (p[5] << 8) | (p[6] << 16)) | (uint32_t(p[7]) << 24), out = 0;
            for (int i = 0; i < 16; i++) {
                uint32_t index = (indices >> (2 * i)) & 3;
                index = four ? index ^ 1 : index < 2 ? index ^ 1 : index;  // 0<->1, 2<->3 / 0<->1
                out |= index << (2 * i);
            }
            for (int i = 0; i < 4; i++)
                p[4 + i] = uint8_t(out >> (8 * i));
        }
        p[0] = uint8_t(s0), p[1] = uint8_t(s0 >> 8), p[2] = uint8_t(s1), p[3] = uint8_t(s1 >> 8);
    }
}

// The texture a texture fetch constant describes, as an NV2A texture.
// red_alpha: white texels with the image's red channel as alpha (a pixel
// shader that uses the texture as a mask).
const Texture* texture_for(const GpuState& gpu, uint32_t slot, bool red_alpha, float z) {
    Image image;
    if (!fetch_image(gpu, slot, image, false))
        return nullptr;
    image.slice = slice_for(image, z);
    TextureKey key{image.address, image.format | (image.width << 6) | (image.height << 19), image.slice, red_alpha};
    auto it = textures.find(key);
    if (it != textures.end() && it->second.checked == frame_number)
        return &it->second;
    uint32_t hash = sample_hash(image);
    if (it != textures.end()) {
        if (it->second.hash == hash) {
            it->second.checked = frame_number;
            return &it->second;
        }
        if (it->second.texture) {
            it->second.texture->Release();
            forget_textures();
        }
        textures.erase(it);
    }
    if (image.d3d_format == D3DFMT_UNKNOWN) {
        log("renderer: texture format %u (%ux%u at %08X) is not converted\n", image.format, image.width,
            image.height, image.address);
        textures[key].hash = hash;
        return nullptr;
    }
    fetch_image(gpu, slot, image);
    uint32_t pot_w = image.pot_w, pot_h = image.pot_h;

    Texture t;
    t.hash = hash;
    t.checked = frame_number;
    t.u_scale = float(image.width) / float(pot_w);
    t.v_scale = float(image.height) / float(pot_h);
    if (red_alpha) {
        // Decoded to 32-bit texels: white, alpha from red.
        std::vector<uint32_t> texels = decode_argb(image);
        Swizzle swizzle = fetch_swizzle(gpu, slot, image);
        uint32_t lit = 0;
        for (uint32_t& texel : texels) {
            texel = 0x00FFFFFF | (channel_byte(texel, swizzle.shift[0]) << 24);
            lit += (texel >> 24) != 0;
        }
        t.texture = create_argb(texels, pot_w, pot_h);
        log("renderer: texture %ux%u format %u%s at %08X as a red mask (%u texels lit, x from %d, swizzle %03X)\n",
            image.width, image.height, image.format, image.tiled ? " tiled" : "", image.address, lit, swizzle.shift[0],
            (gpu.reg(kFetchConstants + slot * 6 + 3) >> 1) & 0xFFF);
        return &(textures[key] = t);
    }

    t.texture = create_raw(image, image.d3d_format);
    if (!t.texture) {
        textures[key].hash = hash;
        return nullptr;
    }
    log("renderer: texture %ux%ux%u%s slice %u format %u%s at %08X\n", image.width, image.height, image.depth,
        image.volume ? " volume" : "", image.slice, image.format, image.tiled ? " tiled" : "", image.address);
    return &(textures[key] = t);
}

// --- the palette shader (AF2C3D4E) ------------------------------------------------
//
// t = tf0(r0), p = tf1(t.y + r1.x, t.x * r1.y)
// colour = overlay(overlay(p, t.w), r1.z), alpha = t.z * r1.w
// where overlay(c, s) = s < 0.5 ? c * 2s : c + (1 - c)(2s - 1).
//
// On the NV2A: texture stage 0 is the shape repacked (blue t.x, green t.y,
// red t.z, alpha t.w), stage 1 reads the palette at (green, blue)
// (DPNDNT_GB) from a palette baked with r1.x and r1.y applied, and the
// combiners (palette_program) do the overlays and alpha.

std::map<uint64_t, Texture> shape_textures;  // by address and slice

const Texture* shape_texture_for(const GpuState& gpu, float z) {
    Image shape;
    if (!fetch_image(gpu, 0, shape, false) || shape.d3d_format == D3DFMT_UNKNOWN)
        return nullptr;
    shape.slice = slice_for(shape, z);
    uint64_t key = (uint64_t(shape.slice) << 32) | shape.address;
    auto it = shape_textures.find(key);
    if (it != shape_textures.end() && it->second.checked == frame_number)
        return &it->second;
    uint32_t hash = sample_hash(shape);
    if (it != shape_textures.end()) {
        if (it->second.hash == hash) {
            it->second.checked = frame_number;
            return &it->second;
        }
        if (it->second.texture) {
            it->second.texture->Release();
            forget_textures();
        }
        shape_textures.erase(it);
    }
    fetch_image(gpu, 0, shape);
    Swizzle s = fetch_swizzle(gpu, 0, shape);
    Texture& t = shape_textures[key];
    t.hash = hash;
    t.checked = frame_number;
    // Stage 0 wants (x, y, z, w) as (blue, green, red, alpha): with the
    // usual swizzle, red and blue swapped. As the title's own format that is
    // A8B8G8R8, or DXT with each block's colours swapped (an exact change);
    // anything else is decoded to 32 bits.
    bool usual = s.shift[0] == 16 && s.shift[1] == 8 && s.shift[2] == 0 && s.shift[3] == 24;
    if (usual && shape.format == 6) {
        t.texture = create_raw(shape, D3DFMT_A8B8G8R8);
    } else if (usual && shape.block == 4) {
        swap_dxt_red_blue(shape);
        t.texture = create_raw(shape, shape.d3d_format);
    } else {
        std::vector<uint32_t> texels = decode_argb(shape);
        for (uint32_t& texel : texels) {
            uint32_t x = channel_byte(texel, s.shift[0]), y = channel_byte(texel, s.shift[1]);
            uint32_t z = channel_byte(texel, s.shift[2]), w = channel_byte(texel, s.shift[3]);
            texel = (w << 24) | (z << 16) | (y << 8) | x;
        }
        t.texture = create_argb(texels, shape.pot_w, shape.pot_h);
    }
    t.u_scale = float(shape.width) / float(shape.pot_w);
    t.v_scale = float(shape.height) / float(shape.pot_h);
    log("renderer: shape texture %ux%ux%u%s slice %u (z %g) format %u at %08X\n", shape.width, shape.height,
        shape.depth, shape.volume ? " volume" : "", shape.slice, z, shape.format, shape.address);
    return &t;
}

// A palette texture as the shader reads it (red, green, blue bytes),
// decoded once and compared with its guest data once a frame.
struct PaletteSource {
    uint32_t width = 0, height = 0, clamp_x = 0, clamp_y = 0;
    uint32_t hash = 0, checked = ~0u;
    std::vector<uint8_t> rgb;  // width x height x 3
};
std::map<uint32_t, PaletteSource> palette_sources;  // by address

const PaletteSource* palette_source_for(const GpuState& gpu, const Image& palette) {
    PaletteSource& source = palette_sources[palette.address];
    if (source.checked == frame_number && source.width)
        return &source;
    uint32_t hash = sample_hash(palette);
    if (source.width && source.hash == hash) {
        source.checked = frame_number;
        return &source;
    }
    Image image;
    fetch_image(gpu, 1, image);
    Swizzle s = fetch_swizzle(gpu, 1, image);
    std::vector<uint32_t> colors = decode_argb(image);
    source.width = image.width;
    source.height = image.height;
    uint32_t f0 = gpu.reg(kFetchConstants + 6 + 0);
    source.clamp_x = (f0 >> 10) & 7;
    source.clamp_y = (f0 >> 13) & 7;
    source.rgb.resize(size_t(image.width) * image.height * 3);
    for (uint32_t y = 0; y < image.height; y++)
        for (uint32_t x = 0; x < image.width; x++) {
            uint32_t texel = colors[size_t(y) * image.pot_w + x];
            uint8_t* to = &source.rgb[(size_t(y) * image.width + x) * 3];
            for (int c = 0; c < 3; c++)
                to[c] = uint8_t(channel_byte(texel, s.shift[c]));
        }
    source.hash = hash;
    source.checked = frame_number;
    return &source;
}

struct PaletteKey {
    uint32_t palette;  // texture address
    int32_t u, v;      // r1.x and r1.y in 1/256ths
    bool operator<(const PaletteKey& o) const {
        if (palette != o.palette)
            return palette < o.palette;
        if (u != o.u)
            return u < o.u;
        return v < o.v;
    }
};

std::map<PaletteKey, Texture> palette_textures;
uint32_t palette_bakes;  // statistics
// Textures of evicted palettes, reused for new ones of the same size two
// frames later (the NV2A may still draw from them this frame).
struct Pooled {
    uint32_t size_key, frame;
    IDirect3DTexture8* texture;
};
std::vector<Pooled> palette_pool;

void pool_palette(IDirect3DTexture8* texture) {
    if (!texture)
        return;
    D3DSURFACE_DESC desc;
    texture->GetLevelDesc(0, &desc);
    palette_pool.push_back({(desc.Width << 16) | desc.Height, frame_number, texture});
}

// The palette with r1.x added to u and v scaled by r1.y, in its own
// addressing mode and filtered as the 360 samples it (a scrolling palette
// glides), so that stage 1 reads it at the shape's own (y, x).
const Texture* palette_texture_for(const GpuState& gpu, float r1x, float r1y) {
    Image palette;
    if (!fetch_image(gpu, 1, palette, false) || palette.d3d_format == D3DFMT_UNKNOWN)
        return nullptr;
    PaletteKey key{palette.address, int32_t(__builtin_floorf(r1x * 256.0f + 0.5f)),
                   int32_t(__builtin_floorf(r1y * 256.0f + 0.5f))};
    const PaletteSource* source = palette_source_for(gpu, palette);
    auto it = palette_textures.find(key);
    if (it != palette_textures.end() && it->second.hash == source->hash)
        return &it->second;
    uint32_t pot_w = palette.pot_w, pot_h = palette.pot_h, size_key = (pot_w << 16) | pot_h;
    if (it != palette_textures.end()) {  // the palette changed: remake in place
        pool_palette(it->second.texture);
        palette_textures.erase(it);
    }
    if (palette_textures.size() >= 64) {  // scrolling palettes: keep the cache (16 KB each) small
        for (auto& entry : palette_textures)
            pool_palette(entry.second.texture);
        palette_textures.clear();
        forget_textures();
    }

    // Bilinear in 16.16 fixed point over the decoded palette.
    static std::vector<uint32_t> out;
    out.assign(size_t(pot_w) * pot_h, 0);
    int w = int(source->width), h = int(source->height);
    auto address = [](int i, int size, uint32_t mode) {
        if (mode == 0)  // wrap
            return ((i % size) + size) % size;
        if (mode == 1) {  // mirror
            int period = 2 * size, m = ((i % period) + period) % period;
            return m < size ? m : period - 1 - m;
        }
        return i < 0 ? 0 : i >= size ? size - 1 : i;  // clamp
    };
    // Source x for each output column (u offset), source y for each row.
    static std::vector<int> x0s, x1s, y0s, y1s;
    static std::vector<uint32_t> axs, ays;
    x0s.resize(pot_w), x1s.resize(pot_w), axs.resize(pot_w);
    y0s.resize(pot_h), y1s.resize(pot_h), ays.resize(pot_h);
    for (uint32_t px = 0; px < pot_w; px++) {
        // ((px + 0.5) / pot_w + u) * w - 0.5, in 1/65536ths
        int64_t fx = (((int64_t(2 * px + 1) << 15) / pot_w) + (int64_t(key.u) << 8)) * w - 32768;
        int64_t x0 = fx >> 16;
        x0s[px] = address(int(x0), w, source->clamp_x);
        x1s[px] = address(int(x0) + 1, w, source->clamp_x);
        axs[px] = uint32_t(fx & 0xFFFF) >> 8;  // 0..255
    }
    for (uint32_t py = 0; py < pot_h; py++) {
        int64_t fy = ((((int64_t(2 * py + 1) << 15) / pot_h) * key.v) >> 8) * h - 32768;
        int64_t y0 = fy >> 16;
        y0s[py] = address(int(y0), h, source->clamp_y);
        y1s[py] = address(int(y0) + 1, h, source->clamp_y);
        ays[py] = uint32_t(fy & 0xFFFF) >> 8;
    }
    const uint8_t* rgb = source->rgb.data();
    for (uint32_t py = 0; py < pot_h; py++) {
        const uint8_t* row0 = rgb + size_t(y0s[py]) * w * 3;
        const uint8_t* row1 = rgb + size_t(y1s[py]) * w * 3;
        uint32_t ay = ays[py];
        uint32_t* to = &out[size_t(py) * pot_w];
        for (uint32_t px = 0; px < pot_w; px++) {
            const uint8_t *a = row0 + x0s[px] * 3, *b = row0 + x1s[px] * 3;
            const uint8_t *c = row1 + x0s[px] * 3, *d = row1 + x1s[px] * 3;
            uint32_t ax = axs[px], texel = 0xFF000000;
            for (int k = 0; k < 3; k++) {
                uint32_t top = a[k] * (256 - ax) + b[k] * ax, bottom = c[k] * (256 - ax) + d[k] * ax;
                uint32_t value = (top * (256 - ay) + bottom * ay + 32768) >> 16;
                texel |= value << (16 - 8 * k);
            }
            to[px] = texel;
        }
    }
    palette_bakes++;

    Texture& t = palette_textures[key];
    t.hash = source->hash;
    for (size_t i = 0; i < palette_pool.size(); i++)
        if (palette_pool[i].size_key == size_key && frame_number - palette_pool[i].frame >= 2) {
            t.texture = palette_pool[i].texture;
            palette_pool[i] = palette_pool.back();
            palette_pool.pop_back();
            break;
        }
    if (t.texture) {
        D3DLOCKED_RECT locked;
        if (SUCCEEDED(t.texture->LockRect(0, &locked, nullptr, 0))) {
            XGSwizzleRect(out.data(), pot_w * 4, nullptr, locked.pBits, pot_w, pot_h, nullptr, 4);
            t.texture->UnlockRect(0);
        }
        forget_textures();  // the same texture, new contents
    } else {
        t.texture = create_argb(out, pot_w, pot_h);
    }
    return &t;
}

// The combiners (stage 0 starts with r0.a = t0.a, the shading s):
//   0: r1 = 2 s p, t2 = 2 (1 - p)(s - 1/2)
//   1: t2 = p + t2                        (the lighter half)
//   2: r0 = r0.a >= 0.5 ? t2 : r1          (overlay(p, s))
//   3: t3 = t0 . (1, 0, 0) = t.z, blue to alpha
//   4: r0 = r0 * v0 + c0, r0.a = t3.a * v0.a (the r1.z tint, r1.w alpha)
D3DPIXELSHADERDEF palette_program(float lighten) {
    D3DPIXELSHADERDEF d = {};
    d.PSTextureModes = PS_TEXTUREMODES(PS_TEXTUREMODES_PROJECT2D, PS_TEXTUREMODES_DPNDNT_GB, PS_TEXTUREMODES_NONE,
                                       PS_TEXTUREMODES_NONE);
    d.PSCombinerCount = PS_COMBINERCOUNT(5, PS_COMBINERCOUNT_MUX_MSB | PS_COMBINERCOUNT_SAME_C0 |
                                                PS_COMBINERCOUNT_SAME_C1);
    d.PSRGBInputs[0] = PS_COMBINERINPUTS(PS_REGISTER_T1 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_T0 | PS_CHANNEL_ALPHA | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_T1 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_INVERT,
                                         PS_REGISTER_T0 | PS_CHANNEL_ALPHA | PS_INPUTMAPPING_HALFBIAS_NORMAL);
    d.PSRGBOutputs[0] = PS_COMBINEROUTPUTS(PS_REGISTER_R1, PS_REGISTER_T2, PS_REGISTER_DISCARD,
                                           PS_COMBINEROUTPUT_SHIFTLEFT_1 | PS_COMBINEROUTPUT_AB_CD_SUM);
    d.PSRGBInputs[1] = PS_COMBINERINPUTS(PS_REGISTER_T1 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_ONE | PS_CHANNEL_RGB,
                                         PS_REGISTER_T2 | PS_CHANNEL_RGB | PS_INPUTMAPPING_SIGNED_IDENTITY,
                                         PS_REGISTER_ONE | PS_CHANNEL_RGB);
    d.PSRGBOutputs[1] = PS_COMBINEROUTPUTS(PS_REGISTER_DISCARD, PS_REGISTER_DISCARD, PS_REGISTER_T2,
                                           PS_COMBINEROUTPUT_IDENTITY | PS_COMBINEROUTPUT_AB_CD_SUM);
    d.PSRGBInputs[2] = PS_COMBINERINPUTS(PS_REGISTER_R1 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_ONE | PS_CHANNEL_RGB,
                                         PS_REGISTER_T2 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_ONE | PS_CHANNEL_RGB);
    d.PSRGBOutputs[2] = PS_COMBINEROUTPUTS(PS_REGISTER_DISCARD, PS_REGISTER_DISCARD, PS_REGISTER_R0,
                                           PS_COMBINEROUTPUT_IDENTITY | PS_COMBINEROUTPUT_AB_CD_MUX);
    d.PSRGBInputs[3] = PS_COMBINERINPUTS(PS_REGISTER_T0 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_C1 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_ZERO, PS_REGISTER_ZERO);
    d.PSRGBOutputs[3] = PS_COMBINEROUTPUTS(PS_REGISTER_T3, PS_REGISTER_DISCARD, PS_REGISTER_DISCARD,
                                           PS_COMBINEROUTPUT_IDENTITY | PS_COMBINEROUTPUT_AB_DOT_PRODUCT |
                                               PS_COMBINEROUTPUT_AB_BLUE_TO_ALPHA);
    d.PSRGBInputs[4] = PS_COMBINERINPUTS(PS_REGISTER_R0 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_V0 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_C0 | PS_CHANNEL_RGB | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                         PS_REGISTER_ONE | PS_CHANNEL_RGB);
    d.PSRGBOutputs[4] = PS_COMBINEROUTPUTS(PS_REGISTER_DISCARD, PS_REGISTER_DISCARD, PS_REGISTER_R0,
                                           PS_COMBINEROUTPUT_IDENTITY | PS_COMBINEROUTPUT_AB_CD_SUM);
    d.PSAlphaInputs[4] = PS_COMBINERINPUTS(PS_REGISTER_T3 | PS_CHANNEL_ALPHA | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                           PS_REGISTER_V0 | PS_CHANNEL_ALPHA | PS_INPUTMAPPING_UNSIGNED_IDENTITY,
                                           PS_REGISTER_ZERO, PS_REGISTER_ZERO);
    d.PSAlphaOutputs[4] = PS_COMBINEROUTPUTS(PS_REGISTER_R0, PS_REGISTER_DISCARD, PS_REGISTER_DISCARD,
                                             PS_COMBINEROUTPUT_IDENTITY);
    // Final: r0 as it is.
    d.PSFinalCombinerInputsABCD = PS_COMBINERINPUTS(PS_REGISTER_ZERO, PS_REGISTER_ZERO, PS_REGISTER_ZERO,
                                                    PS_REGISTER_R0 | PS_CHANNEL_RGB);
    d.PSFinalCombinerInputsEFG = PS_COMBINERINPUTS(PS_REGISTER_ZERO, PS_REGISTER_ZERO,
                                                   PS_REGISTER_R0 | PS_CHANNEL_ALPHA, 0);
    uint8_t k = uint8_t(lighten * 255.0f + 0.5f);
    for (int i = 0; i < 8; i++) {
        d.PSConstant0[i] = D3DCOLOR_ARGB(0, k, k, k);
        d.PSConstant1[i] = D3DCOLOR_ARGB(0, 255, 0, 0);
    }
    return d;
}

// --- state -----------------------------------------------------------------------

D3DBLEND blend_factor(uint32_t f) {
    switch (f) {
    case 0: return D3DBLEND_ZERO;
    case 1: return D3DBLEND_ONE;
    case 4: return D3DBLEND_SRCCOLOR;
    case 5: return D3DBLEND_INVSRCCOLOR;
    case 6: return D3DBLEND_SRCALPHA;
    case 7: return D3DBLEND_INVSRCALPHA;
    case 8: return D3DBLEND_DESTCOLOR;
    case 9: return D3DBLEND_INVDESTCOLOR;
    case 10: return D3DBLEND_DESTALPHA;
    case 11: return D3DBLEND_INVDESTALPHA;
    case 16: return D3DBLEND_SRCALPHASAT;
    default: return D3DBLEND_ONE;
    }
}

D3DCMPFUNC compare(uint32_t f) {
    static const D3DCMPFUNC map[8] = {D3DCMP_NEVER,   D3DCMP_LESS,     D3DCMP_EQUAL,        D3DCMP_LESSEQUAL,
                                      D3DCMP_GREATER, D3DCMP_NOTEQUAL, D3DCMP_GREATEREQUAL, D3DCMP_ALWAYS};
    return map[f & 7];
}

// Render and texture stage states as last set: Direct3D puts every Set
// call in the push buffer, so an unchanged value is not set again.
struct StateCache {
    DWORD render[D3DRS_MAX];
    bool render_known[D3DRS_MAX] = {};
    DWORD stage[D3DTSS_MAXSTAGES][D3DTSS_MAX];
    bool stage_known[D3DTSS_MAXSTAGES][D3DTSS_MAX] = {};
    IDirect3DBaseTexture8* texture[D3DTSS_MAXSTAGES] = {};
    bool texture_known[D3DTSS_MAXSTAGES] = {};
} cache;

void set_render_state(D3DRENDERSTATETYPE state, DWORD value) {
    if (unsigned(state) < D3DRS_MAX) {
        if (cache.render_known[state] && cache.render[state] == value)
            return;
        cache.render_known[state] = true;
        cache.render[state] = value;
    }
    device->SetRenderState(state, value);
}

void set_stage_state(DWORD stage, D3DTEXTURESTAGESTATETYPE type, DWORD value) {
    if (stage < D3DTSS_MAXSTAGES && unsigned(type) < D3DTSS_MAX) {
        if (cache.stage_known[stage][type] && cache.stage[stage][type] == value)
            return;
        cache.stage_known[stage][type] = true;
        cache.stage[stage][type] = value;
    }
    device->SetTextureStageState(stage, type, value);
}

void forget_textures() {
    for (bool& known : cache.texture_known)
        known = false;
}

void set_texture(DWORD stage, IDirect3DBaseTexture8* texture) {
    if (stage < D3DTSS_MAXSTAGES) {
        if (cache.texture_known[stage] && cache.texture[stage] == texture)
            return;
        cache.texture_known[stage] = true;
        cache.texture[stage] = texture;
    }
    device->SetTexture(stage, texture);
}

void apply_state(const GpuState& gpu) {
    uint32_t blend = gpu.reg(kRbBlendControl0);
    uint32_t src = blend & 0x1F, dst = (blend >> 8) & 0x1F;
    bool blending = !(src == 1 && dst == 0);
    set_render_state(D3DRS_ALPHABLENDENABLE, blending);
    if (blending) {
        set_render_state(D3DRS_SRCBLEND, blend_factor(src));
        set_render_state(D3DRS_DESTBLEND, blend_factor(dst));
    }
    uint32_t color = gpu.reg(kRbColorControl);
    bool alpha_test = (color >> 3) & 1;
    set_render_state(D3DRS_ALPHATESTENABLE, alpha_test);
    if (alpha_test) {
        set_render_state(D3DRS_ALPHAFUNC, compare(color & 7));
        float ref = as_float(gpu.reg(kRbAlphaRef));
        set_render_state(D3DRS_ALPHAREF, DWORD(ref * 255.0f + 0.5f));
    }
    // Culling: the 360 marks front faces by the face bit; a 2D title rarely
    // culls, and getting it backwards hides everything.
    set_render_state(D3DRS_CULLMODE, D3DCULL_NONE);
    set_render_state(D3DRS_ZENABLE, D3DZB_FALSE);
    uint32_t mask = gpu.reg(kRbColorMask) & 0xF;
    set_render_state(D3DRS_COLORWRITEENABLE,
                           ((mask & 1) ? D3DCOLORWRITEENABLE_RED : 0) | ((mask & 2) ? D3DCOLORWRITEENABLE_GREEN : 0) |
                               ((mask & 4) ? D3DCOLORWRITEENABLE_BLUE : 0) |
                               ((mask & 8) ? D3DCOLORWRITEENABLE_ALPHA : 0));
}

// --- pixel shaders -------------------------------------------------------------------

// What a pixel shader amounts to in fixed function: the interpolator its
// colour comes from, and whether it multiplies by a texture (texture fetch
// constant 0, coordinates from an interpolator).
enum class Mode {
    kModulate,  // texture * colour
    kMask,      // colour.alpha == 0: the texture; else colour with alpha texture.red * colour.alpha
    kTexture,   // the texture alone
    kPalette,   // the palette shader: an NV2A program (palette_program)
};

struct PixelShader {
    uint32_t hash;
    int color;       // interpolator, -1: white
    int texcoord;    // interpolator, -1: no texture
    Mode mode;
};

const PixelShader kPixelShaders[] = {
    {0x8A78BDFF, 0, -1, Mode::kModulate},  // Direct3D's clear: oC0 = r0
    {0x33E1A369, 1, 0, Mode::kModulate},   // sprites: oC0 = tfetch3D(tf0, r0.xyz) * r1
    {0x81DBCACE, 1, 0, Mode::kMask},       // text and images: oC0 = r1.w == 0 ? tfetch2D(tf0, r0.xy)
                                           //   : (r1.xyz, tfetch2D(tf0, r0.xy).x * r1.w)
    // Backgrounds, the menu frame and most sprites: a shape texture coloured
    // through a palette texture (tf1) at offsets in r1, overlay-blended.
    {0xAF2C3D4E, 1, 0, Mode::kPalette},
};

const PixelShader* pixel_shader(uint32_t hash) {
    for (const PixelShader& p : kPixelShaders)
        if (p.hash == hash)
            return &p;
    return nullptr;
}

struct Vertex {
    float x, y, z, rhw;
    D3DCOLOR color;
    float u, v;
};
constexpr DWORD kFvf = D3DFVF_XYZRHW | D3DFVF_DIFFUSE | D3DFVF_TEX1;

// Cycles spent in DrawPrimitiveUP (stats: the rest of draw is state).
uint64_t primitive_cycles;
// Statistics: cycles in draw's phases (setup, vertices, triangles, state,
// texture lookup, run set-up).
uint64_t phase_cycles[6], phase_mark;
// The pixel shader set: -1 fixed function, else the palette program's c0.
int current_program = -2;

// Diagnostic: the draws of one frame in a few hundred are logged, and of
// the next frame after request_debug_frame.
bool debug_frame;
volatile bool debug_requested;

uint8_t to_byte(float v) {
    if (!(v > 0))
        return 0;
    if (v >= 1)
        return 255;
    return uint8_t(v * 255.0f + 0.5f);
}

}  // namespace

namespace renderer {

bool init() {
    d3d = Direct3DCreate8(D3D_SDK_VERSION);
    if (!d3d)
        return false;
    D3DPRESENT_PARAMETERS pp = {};
    pp.BackBufferWidth = kWidth;
    pp.BackBufferHeight = kHeight;
    pp.BackBufferFormat = D3DFMT_X8R8G8B8;
    pp.BackBufferCount = 1;
    pp.EnableAutoDepthStencil = FALSE;
    pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    pp.FullScreen_PresentationInterval = D3DPRESENT_INTERVAL_ONE_OR_IMMEDIATE;
    if (FAILED(d3d->CreateDevice(0, D3DDEVTYPE_HAL, nullptr, D3DCREATE_HARDWARE_VERTEXPROCESSING, &pp, &device)))
        return false;
    set_render_state(D3DRS_LIGHTING, FALSE);
    device->SetVertexShader(kFvf);
    device->Clear(0, nullptr, D3DCLEAR_TARGET, 0, 1.f, 0);
    log("renderer: device ready\n");
    return true;
}

void draw(const GpuState& gpu, uint32_t primitive, const ShadedVertex* in, uint32_t count, uint32_t ps_hash) {
    phase_mark = __builtin_ia32_rdtsc();
    if (!device || !count)
        return;
    const PixelShader* ps = pixel_shader(ps_hash);
    if (!ps) {
        static uint32_t said[16];
        static int said_count;
        bool known = false;
        for (int i = 0; i < said_count; i++)
            known |= said[i] == ps_hash;
        if (!known && said_count < 16) {
            said[said_count++] = ps_hash;
            log("renderer: pixel shader %08X drawn as its first interpolator\n", ps_hash);
        }
        static const PixelShader fallback{0, 0, -1, Mode::kModulate};
        ps = &fallback;
    }
    // The viewport transform (PA_CL_VTE_CNTL): divide by w unless the
    // position is already in screen space, then scale and offset.
    uint32_t vte = gpu.reg(kPaClVteCntl);
    float xs = (vte & 1) ? as_float(gpu.reg(kPaClVportXScale)) : 1;
    float xo = (vte & 2) ? as_float(gpu.reg(kPaClVportXOffset)) : 0;
    float ys = (vte & 4) ? as_float(gpu.reg(kPaClVportYScale)) : 1;
    float yo = (vte & 8) ? as_float(gpu.reg(kPaClVportYOffset)) : 0;
    // vtx_xy_fmt: the position is already divided by w (screen space). The
    // w0 format bit only says the rasteriser works with 1/W: the shader's w
    // is still W.
    bool xy_divided = (vte >> 8) & 1;
    // A surface with MSAA (RB_SURFACE_INFO) has its pixels' samples side by
    // side: Direct3D clears a 640x480 1x target as a 320x240 4x one. Its
    // coordinates are in those larger pixels.
    uint32_t msaa = (gpu.reg(0x2000) >> 16) & 3;
    float msaa_x = msaa == 2 ? 2.0f : 1.0f, msaa_y = msaa ? 2.0f : 1.0f;

    // Point lists: each point a screen-aligned square (a quad list here), as
    // big as its oPts.x diameter (clamped by PA_SU_POINT_MINMAX) or
    // PA_SU_POINT_SIZE, with its 0..1 sprite coordinates in the interpolator
    // SQ_CONTEXT_MISC.param_gen_pos names when SQ_PROGRAM_CNTL.param_gen.
    static std::vector<ShadedVertex> points;
    if (primitive == 1) {
        uint32_t size = gpu.reg(0x2280), minmax = gpu.reg(0x2281);
        float fixed_half_w = float(size >> 16) / 16, fixed_half_h = float(size & 0xFFFF) / 16;
        float min_half = float(minmax & 0xFFFF) / 16, max_half = float(minmax >> 16) / 16;
        bool param_gen = (gpu.reg(0x2180) >> 18) & 1;
        uint32_t gen = (gpu.reg(0x2181) >> 8) & 0xFF;
        points.resize(size_t(count) * 4);
        static const float kCorner[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        for (uint32_t i = 0; i < count; i++) {
            float half_w = fixed_half_w, half_h = fixed_half_h;
            if (in[i].point_size >= 0) {
                float half = in[i].point_size / 2;
                half = half < min_half ? min_half : half > max_half ? max_half : half;
                half_w = half_h = half;
            }
            // Screen pixels to the position's units (the viewport scale, and
            // w unless it is already divided).
            float w = xy_divided ? 1.0f : in[i].position[3];
            float to_x = xs != 0 ? w / xs : 0, to_y = ys != 0 ? w / ys : 0;
            for (int c = 0; c < 4; c++) {
                ShadedVertex& q = points[size_t(i) * 4 + c];
                q = in[i];
                q.position[0] += kCorner[c][0] * half_w * to_x;
                q.position[1] += kCorner[c][1] * half_h * to_y;
                if (param_gen && gen < uint32_t(kInterpolators)) {
                    q.interpolators[gen][0] = (kCorner[c][0] + 1) / 2;
                    q.interpolators[gen][1] = (kCorner[c][1] + 1) / 2;
                }
            }
        }
        static bool said;
        if (!said)
            log("renderer: point sprites: size %08X minmax %08X param_gen %d at %u, ps %08X, first size %g\n", size,
                minmax, int(param_gen), gen, ps_hash, in[0].point_size);
        said = true;
        in = points.data();
        count *= 4;
        primitive = 13;
    }

    phase_cycles[0] += __builtin_ia32_rdtsc() - phase_mark;
    phase_mark = __builtin_ia32_rdtsc();
    // Vertices with the texture coordinates as the shader has them (scaled
    // to the power-of-two texture when drawn).
    static std::vector<Vertex> v;
    v.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        const float* p = in[i].position;
        float w = p[3];
        float rhw = xy_divided || w == 0 ? 1.0f : 1.0f / w;
        float x = xy_divided ? p[0] : p[0] * rhw, y = xy_divided ? p[1] : p[1] * rhw;
        // The 360 samples at pixel centres (.5), Direct3D 8 at integers.
        v[i].x = (x * xs + xo) * msaa_x - 0.5f;
        v[i].y = (y * ys + yo) * msaa_y - 0.5f;
        v[i].z = 0;
        v[i].rhw = rhw > 0 ? rhw : 1.0f;
        if (ps->mode == Mode::kPalette) {
            // r1.z tints: below 0.5 towards black (the vertex colour), above
            // towards white (the vertex colour, then c0 added); r1.w: alpha.
            const float* c = in[i].interpolators[ps->color];
            float scale = c[2] < 0.5f ? 2 * c[2] : 2 - 2 * c[2];
            uint8_t b = to_byte(scale);
            v[i].color = D3DCOLOR_ARGB(to_byte(c[3]), b, b, b);
        } else if (ps->color >= 0) {
            const float* c = in[i].interpolators[ps->color];
            v[i].color = D3DCOLOR_ARGB(to_byte(c[3]), to_byte(c[0]), to_byte(c[1]), to_byte(c[2]));
        } else {
            v[i].color = 0xFFFFFFFF;
        }
        if (ps->texcoord >= 0) {
            v[i].u = in[i].interpolators[ps->texcoord][0];
            v[i].v = in[i].interpolators[ps->texcoord][1];
        } else {
            v[i].u = v[i].v = 0;
        }
    }

    phase_cycles[1] += __builtin_ia32_rdtsc() - phase_mark;
    phase_mark = __builtin_ia32_rdtsc();
    // Primitives as triangles, each with the vertex its texture and tint
    // come from (sprites of one draw choose their own array layer).
    static std::vector<Vertex> tri;
    static std::vector<uint32_t> source;
    tri.clear();
    source.clear();
    auto add = [&](uint32_t a, uint32_t b, uint32_t c) {
        tri.push_back(v[a]);
        tri.push_back(v[b]);
        tri.push_back(v[c]);
        source.push_back(a);
    };
    switch (primitive) {
    case 4:  // triangle list
        for (uint32_t i = 0; i + 2 < count; i += 3)
            add(i, i + 1, i + 2);
        break;
    case 5:  // triangle fan
        for (uint32_t i = 1; i + 1 < count; i++)
            add(0, i, i + 1);
        break;
    case 6:  // triangle strip
        for (uint32_t i = 0; i + 2 < count; i++)
            if (i & 1)
                add(i + 1, i, i + 2);
            else
                add(i, i + 1, i + 2);
        break;
    case 8:  // rectangle list: three corners, the fourth implied
        for (uint32_t i = 0; i + 2 < count; i += 3) {
            // The corner opposite the missing one is at the right angle: the
            // one not on the longest edge (the diagonal).
            auto d2 = [&](uint32_t a, uint32_t b) {
                float dx = v[a].x - v[b].x, dy = v[a].y - v[b].y;
                return dx * dx + dy * dy;
            };
            uint32_t a = i, b = i + 1, c = i + 2;  // c: the right angle
            float ab = d2(i, i + 1), bc = d2(i + 1, i + 2), ca = d2(i + 2, i);
            if (bc >= ab && bc >= ca) {
                a = i + 1; b = i + 2; c = i;
            } else if (ca >= ab && ca >= bc) {
                a = i + 2; b = i; c = i + 1;
            }
            Vertex d = v[c];
            d.x = v[a].x + v[b].x - v[c].x;
            d.y = v[a].y + v[b].y - v[c].y;
            d.u = v[a].u + v[b].u - v[c].u;
            d.v = v[a].v + v[b].v - v[c].v;
            add(a, b, c);
            tri.push_back(v[a]);
            tri.push_back(v[b]);
            tri.push_back(d);
            source.push_back(a);
        }
        break;
    case 13:  // quad list
        for (uint32_t i = 0; i + 3 < count; i += 4) {
            add(i, i + 1, i + 2);
            add(i, i + 2, i + 3);
        }
        break;
    default: {
        static bool said;
        if (!said)
            log("renderer: primitive type %u is not drawn\n", primitive);
        said = true;
        return;
    }
    }
    if (tri.empty())
        return;
    if (debug_frame) {
        log("renderer: draw prim %u ps %08X vte %03X blend %08X color %08X, %u vertices:\n", primitive, ps_hash, vte,
            gpu.reg(kRbBlendControl0), gpu.reg(kRbColorControl), count);
        log("renderer:   window offset %08X mode %08X vtx %08X clip %08X surface %08X scissor %08X-%08X\n",
            gpu.reg(kPaScWindowOffset), gpu.reg(kPaSuScModeCntl), gpu.reg(0x2302), gpu.reg(0x2204),
            gpu.reg(0x2000), gpu.reg(0x2081), gpu.reg(0x2082));
        for (uint32_t i = 0; i < count && i < 4; i++)
            log("renderer:   in (%g %g %g %g) -> (%g %g rhw %g) color %08X uv (%g %g)\n", in[i].position[0],
                in[i].position[1], in[i].position[2], in[i].position[3], v[i].x, v[i].y, v[i].rhw,
                unsigned(v[i].color), v[i].u, v[i].v);
    }

    phase_cycles[2] += __builtin_ia32_rdtsc() - phase_mark;
    phase_mark = __builtin_ia32_rdtsc();
    apply_state(gpu);
    // The window scissor (PA_SC_WINDOW_SCISSOR_TL/BR): the 360 never draws
    // outside its render target; here the back buffer is larger than it.
    uint32_t tl = gpu.reg(0x2081), br = gpu.reg(0x2082);
    auto clamp = [](float v, float hi) { return LONG(v < 0 ? 0 : v > hi ? hi : v); };
    D3DRECT scissor = {clamp(float(tl & 0x7FFF) * msaa_x, kWidth), clamp(float((tl >> 16) & 0x7FFF) * msaa_y, kHeight),
                       clamp(float(br & 0x7FFF) * msaa_x, kWidth), clamp(float((br >> 16) & 0x7FFF) * msaa_y, kHeight)};
    device->SetScissors(1, FALSE, &scissor);
    device->SetVertexShader(kFvf);

    // What a source vertex selects: the array layer (its z), and for the
    // palette shader the palette offset and tint, for the mask shader the
    // branch. Runs of triangles that select the same are drawn together.
    struct Selection {
        float z = 0, r1x = 0, r1y = 0, r1z = 0;
        bool mask = false;
        bool operator==(const Selection& o) const {
            return z == o.z && r1x == o.r1x && r1y == o.r1y && r1z == o.r1z && mask == o.mask;
        }
    };
    auto select = [&](uint32_t i) {
        Selection s;
        if (ps->texcoord >= 0)
            s.z = in[i].interpolators[ps->texcoord][2];
        if (ps->mode == Mode::kPalette) {
            const float* r1 = in[i].interpolators[ps->color];
            s.r1x = r1[0];
            s.r1y = r1[1];
            s.r1z = r1[2] > 0.5f ? 1 : 0;  // only which side of 0.5 changes the program
        } else if (ps->mode == Mode::kMask) {
            s.mask = in[i].interpolators[ps->color][3] != 0;
        }
        return s;
    };
    static std::vector<Vertex> run;
    size_t triangles = source.size();
    for (size_t first = 0; first < triangles;) {
        Selection s = select(source[first]);
        size_t last = first + 1;
        while (last < triangles && select(source[last]) == s)
            last++;

        phase_cycles[3] += __builtin_ia32_rdtsc() - phase_mark;
        phase_mark = __builtin_ia32_rdtsc();
        // kMask: the colour's alpha chooses the shader's branch.
        Mode mode = ps->mode;
        if (mode == Mode::kMask)
            mode = s.mask ? Mode::kModulate : Mode::kTexture;
        const Texture* texture = nullptr;
        const Texture* palette = nullptr;
        float lighten = 0;
        if (mode == Mode::kPalette) {
            texture = shape_texture_for(gpu, s.z);
            palette = palette_texture_for(gpu, s.r1x, s.r1y);
            float r1z = in[source[first]].interpolators[ps->color][2];
            lighten = r1z > 0.5f ? 2 * r1z - 1 : 0;
            if (!palette || !palette->texture)
                texture = nullptr;
        } else if (ps->texcoord >= 0) {
            texture = texture_for(gpu, 0, s.mask, s.z);
        }

        if (debug_frame && ps->texcoord >= 0) {
            Image image;
            if (fetch_image(gpu, 0, image, false))
                log("renderer:   run of %u triangles: mode %d mask %d z %g texture %ux%ux%u format %u at %08X%s\n",
                    unsigned(last - first), int(mode), int(s.mask), s.z, image.width, image.height, image.depth,
                    image.format, image.address, texture && texture->texture ? "" : " (none)");
        }
        phase_cycles[4] += __builtin_ia32_rdtsc() - phase_mark;
        phase_mark = __builtin_ia32_rdtsc();
        if (texture)
            const_cast<Texture*>(texture)->used = frame_number;
        run.assign(tri.begin() + first * 3, tri.begin() + last * 3);
        if (texture)
            for (Vertex& vertex : run) {
                vertex.u *= texture->u_scale;
                vertex.v *= texture->v_scale;
            }
        if (texture && texture->texture) {
            set_texture(0, texture->texture);
            // kTexture: the texture alone; kModulate: times the colour (a mask
            // texture is white, so its colour is the vertex colour).
            D3DTEXTUREOP op = mode == Mode::kTexture ? D3DTOP_SELECTARG1 : D3DTOP_MODULATE;
            set_stage_state(0, D3DTSS_COLOROP, op);
            set_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
            set_stage_state(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
            set_stage_state(0, D3DTSS_ALPHAOP, op);
            set_stage_state(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
            set_stage_state(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);
            set_stage_state(0, D3DTSS_MINFILTER, D3DTEXF_LINEAR);
            set_stage_state(0, D3DTSS_MAGFILTER, D3DTEXF_LINEAR);
            set_stage_state(0, D3DTSS_MIPFILTER, D3DTEXF_NONE);
            set_stage_state(0, D3DTSS_ADDRESSU, D3DTADDRESS_CLAMP);
            set_stage_state(0, D3DTSS_ADDRESSV, D3DTADDRESS_CLAMP);
        } else {
            set_texture(0, nullptr);
            set_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG2);
            set_stage_state(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
            set_stage_state(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG2);
            set_stage_state(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);
        }
        if (mode == Mode::kPalette && texture && texture->texture) {
            set_texture(1, palette->texture);
            set_stage_state(1, D3DTSS_MINFILTER, D3DTEXF_LINEAR);
            set_stage_state(1, D3DTSS_MAGFILTER, D3DTEXF_LINEAR);
            set_stage_state(1, D3DTSS_MIPFILTER, D3DTEXF_NONE);
            set_stage_state(1, D3DTSS_ADDRESSU, D3DTADDRESS_WRAP);
            set_stage_state(1, D3DTSS_ADDRESSV, D3DTADDRESS_WRAP);
            // The program only changes with its c0 (lighten): set again
            // only then (-1: fixed function).
            int k = int(lighten * 255.0f + 0.5f);
            if (current_program != k) {
                D3DPIXELSHADERDEF program = palette_program(lighten);
                device->SetPixelShaderProgram(&program);
                current_program = k;
            }
        } else {
            if (current_program != -1) {
                device->SetPixelShader(0);
                current_program = -1;
            }
            set_texture(1, nullptr);
            set_stage_state(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
        }
        phase_cycles[5] += __builtin_ia32_rdtsc() - phase_mark;
        phase_mark = __builtin_ia32_rdtsc();
        uint64_t t0 = __builtin_ia32_rdtsc();
        device->DrawPrimitiveUP(D3DPT_TRIANGLELIST, UINT(run.size() / 3), run.data(), sizeof(Vertex));
        primitive_cycles += __builtin_ia32_rdtsc() - t0;
        phase_mark = __builtin_ia32_rdtsc();
        first = last;
    }
}

// The 360 clears its render target after copying it out; here the back
// buffer is the frame itself, so the clear waits until it has been shown.
bool clear_pending;
D3DCOLOR clear_color;

void resolve(const GpuState& gpu) {
    uint32_t copy = gpu.reg(kRbCopyControl);
    {
        // Diagnostic: each resolve destination once.
        static std::map<uint64_t, bool> seen;
        uint64_t what = (uint64_t(gpu.reg(0x2319)) << 32) | gpu.reg(0x231B);
        if (!seen[what]) {
            seen[what] = true;
            log("renderer: resolve surface %08X color %08X copy %08X dest %08X pitch %08X info %08X scissor "
                "%08X-%08X offset %08X\n",
                gpu.reg(0x2000), gpu.reg(0x2001), copy, gpu.reg(0x2319), gpu.reg(0x231A), gpu.reg(0x231B),
                gpu.reg(0x2081), gpu.reg(0x2082), gpu.reg(0x2080));
        }
    }
    if ((copy >> 8) & 1) {  // color_clear_enable
        uint32_t c = gpu.reg(kRbColorClear);
        clear_pending = true;
        clear_color = D3DCOLOR_ARGB(c >> 24, (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF);
    }
}

void request_debug_frame() { debug_requested = true; }

uint64_t draw_call_cycles() { return primitive_cycles; }

uint32_t palettes_baked() { return palette_bakes; }

const uint64_t* draw_phase_cycles() { return phase_cycles; }

void present() {
    if (!device)
        return;
    uint32_t presented = ++frame_number;
    if (presented % 300 == 0) {
        // Textures not drawn with for ten seconds (an earlier level's) go.
        auto evict = [&](auto& cache) {
            for (auto it = cache.begin(); it != cache.end();) {
                if (presented - it->second.used > 600) {
                    if (it->second.texture)
                        it->second.texture->Release();
                    it = cache.erase(it);
                    forget_textures();
                } else {
                    ++it;
                }
            }
        };
        evict(textures);
        evict(shape_textures);
    }
    if (presented % 1800 == 1) {
        // The caches' textures and their memory (level 0, power-of-two).
        auto bytes = [](IDirect3DTexture8* t) -> uint32_t {
            D3DSURFACE_DESC desc;
            return t && SUCCEEDED(t->GetLevelDesc(0, &desc)) ? desc.Size : 0;
        };
        uint32_t plain = 0, shapes = 0, palettes = 0, pooled = 0;
        for (auto& e : textures)
            plain += bytes(e.second.texture);
        for (auto& e : shape_textures)
            shapes += bytes(e.second.texture);
        for (auto& e : palette_textures)
            palettes += bytes(e.second.texture);
        for (auto& e : palette_pool)
            pooled += bytes(e.texture);
        log("renderer: %u frames presented; textures %u (%u KB), shapes %u (%u KB), palettes %u (%u KB), pool %u "
            "(%u KB), palette sources %u\n",
            presented, unsigned(textures.size()), plain >> 10, unsigned(shape_textures.size()), shapes >> 10,
            unsigned(palette_textures.size()), palettes >> 10, unsigned(palette_pool.size()), pooled >> 10,
            unsigned(palette_sources.size()));
    }
    debug_frame = presented % 600 == 0 || debug_requested;
    debug_requested = false;
    device->Present(nullptr, nullptr, nullptr, nullptr);
    device->SetScissors(0, FALSE, nullptr);  // the whole back buffer
    if (clear_pending)
        device->Clear(0, nullptr, D3DCLEAR_TARGET, clear_color, 1.f, 0);
    clear_pending = false;
}

}  // namespace renderer
}  // namespace gpu
