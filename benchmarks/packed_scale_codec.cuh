// A panel is the 32 consecutive Q4 tiles consumed by one MMA iteration.
// Quant nibbles are unchanged. Scales store the low byte, sign and a
// four-bit delta from the panel's minimum high byte. Wide panels stay raw.
static constexpr int kPanelTiles = 32;
static constexpr int kTileQuants = 256;
static constexpr int kRawTile = 288;
static constexpr int kRawPanel = kPanelTiles * kRawTile;
static constexpr int kPanelQuants = kPanelTiles * kTileQuants;
static constexpr int kScaleTile = 26;
static constexpr int kPanelBase = kPanelQuants + kPanelTiles * kScaleTile;
static constexpr int kEncodedPanel = (kPanelBase + 32) & ~31;

static __host__ __device__ uint16_t unpack_scale(const uint8_t* panel, int tile, int row) {
    const uint8_t* scale = panel + kPanelQuants + tile * kScaleTile;
    const unsigned delta = (scale[16 + row / 2] >> (4 * (row & 1))) & 15;
    const unsigned sign = (scale[24 + row / 8] >> (row & 7)) & 1;
    return (uint16_t)(scale[row] | ((panel[kPanelBase] + delta) << 8) | (sign << 15));
}

static size_t encode_scale_panels(std::vector<uint8_t>& destination, const uint8_t* source, size_t bytes) {
    const size_t panels = bytes / kRawPanel;
    if (!panels || panels * kRawPanel != bytes) return bytes;
    size_t offset = align_up(panels * sizeof(uint32_t), 32);
    destination.resize(bytes + offset);
    auto* index = reinterpret_cast<uint32_t*>(destination.data());
    for (size_t p = 0; p < panels; ++p) {
        const uint8_t* panel = source + p * kRawPanel;
        uint8_t* output = destination.data() + offset;
        unsigned lo = 127, hi = 0;
        for (int tile = 0; tile < kPanelTiles; ++tile) {
            const uint8_t* scale = panel + tile * kRawTile + kTileQuants;
            for (int row = 0; row < 16; ++row) {
                const unsigned high = scale[row * 2 + 1] & 127;
                lo = std::min(lo, high);
                hi = std::max(hi, high);
            }
        }
        const bool compact = hi - lo < 16;
        index[p] = (uint32_t)offset | compact;
        if (!compact) {
            memcpy(output, panel, kRawPanel);
            offset += kRawPanel;
            continue;
        }
        memset(output + kPanelQuants, 0, kEncodedPanel - kPanelQuants);
        output[kPanelBase] = lo;
        for (int tile = 0; tile < kPanelTiles; ++tile) {
            const uint8_t* raw = panel + tile * kRawTile;
            memcpy(output + tile * kTileQuants, raw, kTileQuants);
            uint8_t* scale = output + kPanelQuants + tile * kScaleTile;
            for (int row = 0; row < 16; ++row) {
                const unsigned high = raw[kTileQuants + row * 2 + 1];
                scale[row] = raw[kTileQuants + row * 2];
                scale[16 + row / 2] |= ((high & 127) - lo) << (4 * (row & 1));
                scale[24 + row / 8] |= (high >> 7) << (row & 7);
            }
        }
        offset += kEncodedPanel;
    }
    return std::min(offset, bytes);
}

static bool verify_scale_panels(const uint8_t* encoded, const uint8_t* original, size_t bytes) {
    const auto* index = reinterpret_cast<const uint32_t*>(encoded);
    for (size_t p = 0; p < bytes / kRawPanel; ++p) {
        const uint8_t* raw = original + p * kRawPanel;
        const uint8_t* panel = encoded + (index[p] & ~1u);
        if (!(index[p] & 1)) {
            if (memcmp(raw, panel, kRawPanel)) return false;
            continue;
        }
        for (int tile = 0; tile < kPanelTiles; ++tile) {
            const uint8_t* raw_tile = raw + tile * kRawTile;
            if (memcmp(raw_tile, panel + tile * kTileQuants, kTileQuants)) return false;
            for (int row = 0; row < 16; ++row) {
                const uint8_t* scale = raw_tile + kTileQuants + row * 2;
                if (unpack_scale(panel, tile, row) != (scale[0] | (scale[1] << 8))) return false;
            }
        }
    }
    return true;
}
