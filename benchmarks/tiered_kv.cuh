// GUM's CUDA adaptation of Vanta's TurboQuant-MSE keys and TTKV age tiers.
// Quantized rows stay addressable for prefix replay and for groups with protected tokens.
static constexpr int kTierHead = 256;
static constexpr int kTierRecent = 512;
static constexpr int kTierOld = 4096;
static constexpr int kTierSplit = 1024;
static constexpr int kTierMidGroup = 4;
static constexpr int kTierOldGroup = 16;

struct TierRow {
    // Each warp lane fetches the eight codes for its dot product in one load.
    uint32_t keys[32];
    uint8_t values[kTierHead / 2];
    float key_scale;
    float2 value_range[kTierHead / 32];
};

struct TierCache {
    float2* recent = nullptr;
    TierRow *rows = nullptr, *middle = nullptr, *old = nullptr;
    // E195: overlay maps a logical position to its compacted individual-row slot
    // (-1 = no individual row; the position is served by the pooled tiers or the
    // recent ring). Null in non-pooling modes, where rows stay positional.
    const int* overlay = nullptr;
    // E195: rolling buffer serving the transient width==1 reads. Every position
    // is read as its individual 4-bit row while it straddles the recent-window
    // boundary (ends [t+recent, t+recent+4)), then forever after by its pooled
    // group row. Rows are packed at commit into slot t % w1_slots; a slot is
    // only overwritten when the commit cursor passes t + w1_slots, far beyond
    // the last straddler read (t + recent + 4). Null in non-pooling modes.
    TierRow* w1 = nullptr;
    int w1_slots = 0;
    int ring = 0, capacity = 0, recent_tokens = kTierRecent, middle_groups = 0;
    // E195: span = logical context the positional old tier must cover. capacity
    // shrinks to the protected-row working set; span keeps the 16:1 pool sized.
    int span = 0;
    bool pooling = false;
    size_t bytes(int heads) const {
        return (size_t)heads * (ring * sizeof(float2) * kTierHead +
            ((pooling ? w1_slots : 0) + capacity +
             (pooling ? middle_groups + span / kTierOldGroup : 0)) * sizeof(TierRow));
    }

    void allocate(int heads) {
        check(cudaMalloc(&recent, (size_t)heads * ring * kTierHead * sizeof(float2)), "recent KV");
        check(cudaMalloc(&rows, (size_t)heads * capacity * sizeof(TierRow)), "TurboQuant KV");
        if (pooling) {
            check(cudaMalloc(&middle, (size_t)heads * middle_groups * sizeof(TierRow)), "TTKV middle");
            check(cudaMalloc(&old, (size_t)heads * (span / kTierOldGroup) * sizeof(TierRow)), "TTKV old");
            check(cudaMalloc(&w1, (size_t)heads * w1_slots * sizeof(TierRow)), "TTKV straddler rows");
        }
    }

    TierCache layer(int heads, int layer_index) const {
        auto result = *this;
        const size_t base = (size_t)heads * layer_index;
        result.recent += base * ring * kTierHead;
        result.rows += base * capacity;
        if (pooling) {
            result.w1 += base * w1_slots;
            result.middle += base * middle_groups;
            result.old += base * (span / kTierOldGroup);
        }
        return result;
    }
};

// Lloyd-Max centroids of the standard normal approximation to a 256-D
// rotated, normalized coordinate scaled by sqrt(256).
__device__ __constant__ float kTierCentroids[16] = {
    -2.73258957f, -2.06901723f, -1.61804639f, -1.25623120f,
    -0.94234046f, -0.65675912f, -0.38804830f, -0.12839503f,
     0.12839503f,  0.38804830f,  0.65675912f,  0.94234046f,
     1.25623120f,  1.61804639f,  2.06901723f,  2.73258957f
};

__device__ float tier_rotate(float x, float* scratch, int i) {
    // Signed Hadamard, an orthogonal structured rotation. Packing carries the
    // normalization in key_scale; queries apply it once here.
    const unsigned sign = ((unsigned)(i + 1) * 0x9e3779b9u) ^ ((unsigned)i * 0x85ebca6bu);
    x = __popc(sign) & 1 ? -x : x;
    for (int bit = 1; bit < 32; bit *= 2) {
        const float other = __shfl_xor_sync(0xffffffffu, x, bit);
        x = (i & bit) ? other - x : x + other;
    }
    for (int bit = 32; bit < kTierHead; bit *= 2) {
        scratch[i] = x;
        __syncthreads();
        const float other = scratch[i ^ bit];
        __syncthreads();
        x = (i & bit) ? other - x : x + other;
    }
    return x;
}

__global__ void k_tier_append(TierCache cache, const float* k, const float* v, int pos, int heads) {
    const int head = blockIdx.x, row = blockIdx.y, i = threadIdx.x;
    const size_t from = ((size_t)row * heads + head) * kTierHead + i;
    cache.recent[((size_t)head * cache.ring + (pos + row) % cache.ring) * kTierHead + i] =
        make_float2(k[from], v[from]);
}

__global__ void k_tier_compact(TierCache cache, int pos, int count, const int* path) {
    const int head = blockIdx.x, i = threadIdx.x;
    float2* recent = cache.recent + (size_t)head * cache.ring * kTierHead;
    for (int row = 0; row < count; ++row) {
        if (path[row] != row)
            recent[((pos + row) % cache.ring) * kTierHead + i] =
                recent[((pos + path[row]) % cache.ring) * kTierHead + i];
    }
}

__global__ void k_tier_pack(TierCache cache, int start_group, int width, int nrot, float rope_base,
                            int fresh_from) {
    __shared__ float scratch[kTierHead], warp_norm[kTierHead / 32], norm;
    __shared__ uint8_t key_codes[kTierHead];
    const int head = blockIdx.x, group = start_group + blockIdx.y, i = threadIdx.x;
    const int start = group * width, lane = i & 31;
    const float2* recent = cache.recent + (size_t)head * cache.ring * kTierHead;
    float key = 0, value = 0;
    const int half = nrot / 2;
    const int pair = i < half ? i + half : i - half;
    const float frequency = width > 1 && i < nrot ? powf(rope_base, -2.f * (i % half) / nrot) : 0;
    for (int offset = 0; offset < width; ++offset) {
        const float2* row = recent + ((start + offset) % cache.ring) * kTierHead;
        float rotated = row[i].x;
        if (frequency != 0) {
            float s, c;
            sincosf((0.5f * (width - 1) - offset) * frequency, &s, &c);
            rotated = rotated * c + (i < half ? -s : s) * row[pair].x;
        }
        key += rotated;
        value += row[i].y;
    }
    key /= width;
    value /= width;
    float square = key * key, lo = value, hi = value;
    for (int bit = 16; bit; bit /= 2) {
        square += __shfl_xor_sync(0xffffffffu, square, bit);
        lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, bit));
        hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, bit));
    }
    if (!lane) warp_norm[i / 32] = square;
    __syncthreads();
    if (i == 0) {
        float sum = 0;
        for (float part : warp_norm) sum += part;
        norm = sqrtf(sum);
    }
    __syncthreads();
    const float rotated_key = tier_rotate(key, scratch, i);
    const float coordinate = norm > 0 ? rotated_key / norm : 0;
    int left = 0, right = 15;
    while (left < right) {
        const int mid = (left + right) / 2;
        if (coordinate > (kTierCentroids[mid] + kTierCentroids[mid + 1]) * 0.5f) left = mid + 1;
        else right = mid;
    }
    const float scale = (hi - lo) / 15.f;
    const int quant_value = scale > 0 ? max(0, min(15, __float2int_rn((value - lo) / scale))) : 0;
    key_codes[i] = left;
    __syncthreads();
    uint32_t key_word = 0;
    if (i < 32) {
        #pragma unroll
        for (int part = 0; part < kTierHead / 32; ++part)
            key_word |= (uint32_t)key_codes[32 * part + i] << (4 * part);
    }
    const int next_value = __shfl_xor_sync(0xffffffffu, quant_value, 1);
    TierRow* pool = width == 1 ? (cache.w1 ? cache.w1 : cache.rows) :
        width == kTierMidGroup ? cache.middle : cache.old;
    const int groups = width == kTierMidGroup ? cache.middle_groups :
        width == 1 ? (cache.w1 ? cache.w1_slots : cache.capacity) : cache.span / kTierOldGroup;
    const int slot = group % groups;
    // E195: the rolling straddler buffer recycles slots every w1_slots
    // positions. Re-packing a position older than one full slot cycle would
    // clobber the slot's current occupant, so the rolling write only covers
    // fresh positions; the compacted overlay write has no recycling and is
    // always safe.
    const bool fresh = width != 1 || group >= fresh_from;
    if (fresh) {
        TierRow& out = pool[(size_t)head * groups + slot];
        if (i == 0) out.key_scale = norm / 16.f;
        if (i < 32) out.keys[i] = key_word;
        if (!lane) out.value_range[i / 32] = make_float2(scale, lo);
        if (!(i & 1)) out.values[i / 2] = quant_value | (next_value << 4);
    }
    if (width == 1 && cache.overlay) {
        // E195: old protected groups read compacted rows through the overlay.
        const int cslot = cache.overlay[group];
        if (cslot >= 0) {
            TierRow& comp = cache.rows[(size_t)head * cache.capacity + cslot];
            if (i == 0) comp.key_scale = norm / 16.f;
            if (i < 32) comp.keys[i] = key_word;
            if (!lane) comp.value_range[i / 32] = make_float2(scale, lo);
            if (!(i & 1)) comp.values[i / 2] = quant_value | (next_value << 4);
        }
    }
}

__global__ void k_tier_query(float* rotated, const float* query, int heads, int stride) {
    __shared__ float scratch[kTierHead];
    const int i = threadIdx.x;
    const size_t head = (size_t)blockIdx.y * heads + blockIdx.x;
    rotated[head * kTierHead + i] = tier_rotate(query[head * stride + i], scratch, i) / 16.f;
}

__device__ int tier_index(const uint8_t* packed, int i) {
    return (packed[i / 2] >> (4 * (i & 1))) & 15;
}

// Once a tile is outside the rolling middle tier for every query, a group
// contributes either its pooled row or its sixteen consecutive overlay rows.
// Keep the original logical order and split boundaries in both reductions.
template<int HeadTile>
__global__ void k_tier_old_attention(float* partial, TierCache cache, const float* rotated,
                                    const uint8_t* protected_tokens, int query_heads, int heads,
                                    int splits, int first_row, float scale) {
    constexpr int groups = kTierSplit / kTierOldGroup;
    constexpr int warps = kTierHead / 32;
    static_assert(groups == 64);
    __shared__ int rows[kTierSplit], protected_before[groups / 32], count;
    __shared__ float scores[HeadTile][kTierSplit], centroids[16];
    __shared__ float warp_max[HeadTile][warps], maximum[HeadTile], sum[HeadTile];
    const int i = threadIdx.x, lane = i & 31, warp = i / 32;
    const int h = blockIdx.x * HeadTile, kh = h / (query_heads / heads);
    const int group_base = blockIdx.z * groups;
    const size_t partial_head = (size_t)blockIdx.y * query_heads + h;
    const TierRow* history = cache.rows + (size_t)kh * cache.capacity;
    const TierRow* old = cache.old + (size_t)kh * (cache.span / kTierOldGroup);
    rotated += ((size_t)first_row * query_heads + partial_head) * kTierHead;
    if (i < 16) centroids[i] = kTierCentroids[i];

    const int group = group_base + i;
    const bool keep = i < groups && protected_tokens[group * kTierOldGroup];
    const unsigned mask = __ballot_sync(0xffffffffu, keep);
    if (i < groups && lane == 0) protected_before[warp] = __popc(mask);
    __syncthreads();
    if (i < groups) {
        const int prior = (warp ? protected_before[0] : 0) + __popc(mask & ((1u << lane) - 1));
        const int offset = i + prior * (kTierOldGroup - 1);
        if (keep) {
            const int first = cache.overlay[group * kTierOldGroup];
            #pragma unroll
            for (int j = 0; j < kTierOldGroup; ++j) rows[offset + j] = first + j;
        } else {
            rows[offset] = -group - 1;
        }
    }
    if (i == 0) count = groups + (protected_before[0] + protected_before[1]) * (kTierOldGroup - 1);
    __syncthreads();

    float q[HeadTile][warps];
    float local_max[HeadTile];
    #pragma unroll
    for (int head = 0; head < HeadTile; ++head) {
        local_max[head] = -INFINITY;
        #pragma unroll
        for (int part = 0; part < warps; ++part)
            q[head][part] = rotated[head * kTierHead + part * 32 + lane];
    }
    for (int slot = warp; slot < count; slot += warps) {
        const int row = rows[slot];
        const TierRow& packed = row >= 0 ? history[row] : old[-row - 1];
        const uint32_t codes = packed.keys[lane];
        float dot[HeadTile] = {};
        #pragma unroll
        for (int part = 0; part < warps; ++part) {
            const float key = centroids[(codes >> (4 * part)) & 15];
            #pragma unroll
            for (int head = 0; head < HeadTile; ++head) dot[head] += q[head][part] * key;
        }
        #pragma unroll
        for (int head = 0; head < HeadTile; ++head) {
            for (int bit = 16; bit; bit /= 2) dot[head] += __shfl_down_sync(0xffffffffu, dot[head], bit);
        }
        if (lane == 0) {
            const float key_scale = packed.key_scale;
            #pragma unroll
            for (int head = 0; head < HeadTile; ++head) {
                dot[head] *= key_scale;
                const float score = dot[head] * scale + (row >= 0 ? 0.f : 2.77258872f);
                scores[head][slot] = score;
                local_max[head] = fmaxf(local_max[head], score);
            }
        }
    }
    #pragma unroll
    for (int head = 0; head < HeadTile; ++head)
        if (lane == 0) warp_max[head][warp] = local_max[head];
    __syncthreads();
    if (i < HeadTile) {
        float mx = -INFINITY;
        for (float part : warp_max[i]) mx = fmaxf(mx, part);
        maximum[i] = mx;
    }
    __syncthreads();
    #pragma unroll
    for (int head = 0; head < HeadTile; ++head)
        for (int slot = i; slot < count; slot += kTierHead)
            scores[head][slot] = expf(scores[head][slot] - maximum[head]);
    __syncthreads();
    if (i < HeadTile) {
        float s = 0;
        for (int slot = 0; slot < count; ++slot) s += scores[i][slot];
        sum[i] = s;
    }

    float value[HeadTile] = {};
    for (int slot = 0; slot < count; ++slot) {
        const int row = rows[slot];
        const TierRow& packed = row >= 0 ? history[row] : old[-row - 1];
        const float2 range = packed.value_range[warp];
        const float v = tier_index(packed.values, i) * range.x + range.y;
        #pragma unroll
        for (int head = 0; head < HeadTile; ++head) value[head] += scores[head][slot] * v;
    }
    __syncthreads();
    #pragma unroll
    for (int head = 0; head < HeadTile; ++head) {
        float* dst = partial + ((partial_head + head) * splits + blockIdx.z) * (kTierHead + 2);
        dst[i] = value[head];
        if (i == 0) dst[kTierHead] = maximum[head], dst[kTierHead + 1] = sum[head];
    }
}

__global__ void k_tier_attention(float* output, float* partial, TierCache cache, const float* query,
                                const float* rotated, const uint8_t* protected_tokens, int pos,
                                int query_heads, int heads, int stride, const int* depth,
                                const int* path, int path_stride, int splits, int first_row, float scale,
                                int first_split = 0) {
    __shared__ float scores[kTierSplit], centroids[16], maximum, sum;
    // 0 means skip this slot; 1 means a single token; 4 and 16 select pooled rows.
    __shared__ uint8_t widths[kTierSplit];
    const int h = blockIdx.x, row = first_row + blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kh = h / (query_heads / heads);
    const size_t head = (size_t)row * query_heads + h;
    query += head * stride;
    rotated += head * kTierHead;
    const int end = pos + (depth ? depth[row] : row) + 1;
    const int recent_start = min(pos, max(0, end - cache.recent_tokens));
    const int old_start = max(0, end - kTierOld);
    const int split = first_split + blockIdx.z;
    const int start = split * kTierSplit;
    const int count = max(0, min(kTierSplit, end - start));
    const int* branch = path ? path + row * path_stride : nullptr;
    const float2* recent = cache.recent + (size_t)kh * cache.ring * kTierHead;
    const TierRow* history = cache.rows + (size_t)kh * cache.capacity;
    const TierRow* middle = cache.pooling ? cache.middle + (size_t)kh * cache.middle_groups : nullptr;
    const TierRow* old = cache.pooling ? cache.old + (size_t)kh * (cache.span / kTierOldGroup) : nullptr;
    if (threadIdx.x < 16) centroids[threadIdx.x] = kTierCentroids[threadIdx.x];
    __syncthreads();
    for (int slot = warp; slot < count; slot += kTierHead / 32) {
        const int t = start + slot;
        int width = 1;
        if (cache.pooling && t < recent_start) {
            const int tier2 = t & ~(kTierOldGroup - 1);
            const int tier1 = t & ~(kTierMidGroup - 1);
            int anchor = t;
            // The mask stores each group's structural floor at its aligned start.
            // A protected group keeps its original individual rows at both ages.
            if (tier2 + kTierOldGroup <= old_start && !protected_tokens[tier2])
                anchor = tier2, width = kTierOldGroup;
            else if ((cache.middle_groups == cache.span / kTierMidGroup || tier2 + kTierOldGroup > old_start)
                     && tier1 + kTierMidGroup <= recent_start && !protected_tokens[tier1 + 1])
                anchor = tier1, width = kTierMidGroup;
            if (anchor != t) width = 0;
        }
        if (lane == 0) widths[slot] = width;
        float score = -INFINITY;
        if (width) {
            const int slot_pos = branch && t >= pos ? pos + branch[t - pos] : t;
            float dot = 0;
            if (t >= recent_start) {
                const float2* key = recent + (slot_pos % cache.ring) * kTierHead;
                for (int group = 0; group < kTierHead; group += 32) {
                    float term = query[group + lane] * key[group + lane].x;
                    for (int bit = 16; bit; bit /= 2) term += __shfl_down_sync(0xffffffffu, term, bit);
                    dot += term;
                }
            } else {
                // E195: straddling the recent-window boundary reads the rolling
                // individual row (bit-identical to the full-context layout);
                // deeper width==1 reads are protected groups via the overlay.
                const bool straddle = cache.w1 &&
                    (t & ~(kTierMidGroup - 1)) + kTierMidGroup > recent_start;
                const TierRow& packed = width == 1 ?
                    (straddle ? cache.w1[(size_t)kh * cache.w1_slots + t % cache.w1_slots] :
                                history[cache.overlay ? cache.overlay[t] : t]) :
                    width == kTierMidGroup ?
                    middle[(t / kTierMidGroup) % cache.middle_groups] : old[t / kTierOldGroup];
                const uint32_t codes = packed.keys[lane];
                #pragma unroll
                for (int part = 0; part < kTierHead / 32; ++part)
                    dot += rotated[part * 32 + lane] * centroids[(codes >> (4 * part)) & 15];
                for (int bit = 16; bit; bit /= 2) dot += __shfl_down_sync(0xffffffffu, dot, bit);
                dot *= packed.key_scale;
            }
            score = dot * scale + (width == 1 ? 0.f : width == kTierMidGroup ? 1.38629436f : 2.77258872f);
        }
        if (lane == 0) scores[slot] = score;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        maximum = -INFINITY;
        for (int slot = 0; slot < count; ++slot) maximum = fmaxf(maximum, scores[slot]);
    }
    __syncthreads();
    for (int slot = threadIdx.x; slot < count; slot += kTierHead)
        scores[slot] = widths[slot] ? expf(scores[slot] - maximum) : 0.f;
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0;
        for (int slot = 0; slot < count; ++slot) sum += scores[slot];
    }
    __syncthreads();
    const int i = threadIdx.x;
    float value = 0;
    for (int slot = 0; slot < count; ++slot) {
        const int width = widths[slot], t = start + slot;
        if (!width) continue;
        float v;
        if (t >= recent_start) {
            const int slot_pos = branch && t >= pos ? pos + branch[t - pos] : t;
            v = recent[(slot_pos % cache.ring) * kTierHead + i].y;
        } else {
            const bool straddle = cache.w1 &&
                (t & ~(kTierMidGroup - 1)) + kTierMidGroup > recent_start;
            const TierRow& packed = width == 1 ?
                (straddle ? cache.w1[(size_t)kh * cache.w1_slots + t % cache.w1_slots] :
                            history[cache.overlay ? cache.overlay[t] : t]) :
                width == kTierMidGroup ?
                middle[(t / kTierMidGroup) % cache.middle_groups] : old[t / kTierOldGroup];
            const float2 range = packed.value_range[i / 32];
            v = tier_index(packed.values, i) * range.x + range.y;
        }
        value += scores[slot] * v;
    }
    if (splits == 1) {
        output[head * kTierHead + i] = value / sum;
    } else {
        const size_t partial_head = (size_t)blockIdx.y * query_heads + h;
        float* dst = partial + (partial_head * splits + split) * (kTierHead + 2);
        dst[i] = value;
        if (i == 0) dst[kTierHead] = maximum, dst[kTierHead + 1] = sum;
    }
}

__global__ void k_tier_merge(float* output, const float* partial, int splits) {
    const size_t head = blockIdx.x;
    partial += head * splits * (kTierHead + 2);
    __shared__ float maximum;
    if (threadIdx.x == 0) {
        maximum = -INFINITY;
        for (int split = 0; split < splits; ++split)
            maximum = fmaxf(maximum, partial[split * (kTierHead + 2) + kTierHead]);
    }
    __syncthreads();
    float value = 0, sum = 0;
    for (int split = 0; split < splits; ++split) {
        const float* src = partial + split * (kTierHead + 2);
        const float scale = expf(src[kTierHead] - maximum);
        value += src[threadIdx.x] * scale;
        sum += src[kTierHead + 1] * scale;
    }
    output[head * kTierHead + threadIdx.x] = value / sum;
}
