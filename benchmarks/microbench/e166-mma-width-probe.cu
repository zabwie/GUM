// E166 probe: validate the packed-verify MMA path at verify widths 48 and 64
// (q4_mma GROUPS=6/8) BEFORE wiring them into real_decode.cu.
//
// Checks per (K, rows) case:
//   1. BIT-IDENTITY: one fused q4_mma<NB,8,G> launch vs G separate
//      q4_mma<NB,8,1> launches on per-group slices. GROUPS only offsets the
//      activation/y pointers, so per-row reduction order is unchanged and the
//      outputs must match exactly. This also covers partial-row masking
//      (rows 33/48/63) because both sides use the same `row < rows` guard.
//   2. DP4A CROSS-CHECK: q4_mma vs the dp4a k_mat_q4<NB> fallback within fp32
//      tolerance (accumulation order differs; the int block-dot is exact).
//   3. OCCUPANCY: cudaFuncGetAttributes (registers/smem/max threads) for every
//      instantiation the decoder would launch, plus the dp4a NB=64 fallback.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
#include "../packed_scale_codec.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t s, const char* w) {
    if (s != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(s)); exit(1); }
}
static uint64_t align_up64(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// ---- kernels copied verbatim from benchmarks/real_decode.cu ----
__global__ void k_quant_x(const float* x, long nb, int8_t* xq, int2* xmeta) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nb) return;
    float amax = 0.f; for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
    float d = amax / 127.f; float id = d > 0.f ? 1.f / d : 0.f; int s = 0;
    for (int i = 0; i < 32; ++i) { int q = __float2int_rn(x[b * 32 + i] * id); xq[b * 32 + i] = (int8_t)q; s += q; }
    xmeta[b] = make_int2(__float_as_int(d), s);
}

template <int NB>
__global__ void k_mat_q4(const uint8_t* w, long K, long rows, const int8_t* xq, const int2* xmeta, float* y, long ystride, const int* row_ids = nullptr) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32; const size_t rowbytes = (size_t)nb * 18;
    const uint8_t* r = w + (size_t)(row_ids ? row_ids[row] : row) * rowbytes;
    float acc[NB];
#pragma unroll
    for (int v = 0; v < NB; ++v) acc[v] = 0.f;
    for (long b = lane; b < nb; b += 32) {
        const uint16_t* blk = (const uint16_t*)(r + (size_t)b * 18);
        const float d = __half2float(__ushort_as_half(blk[0]));
        uint32_t q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
        const uint32_t lo0 = q[0] & 0x0F0F0F0Fu, lo1 = q[1] & 0x0F0F0F0Fu, lo2 = q[2] & 0x0F0F0F0Fu, lo3 = q[3] & 0x0F0F0F0Fu;
        const uint32_t hi0 = (q[0] >> 4) & 0x0F0F0F0Fu, hi1 = (q[1] >> 4) & 0x0F0F0F0Fu, hi2 = (q[2] >> 4) & 0x0F0F0F0Fu, hi3 = (q[3] >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int v = 0; v < NB; ++v) {
            const int4* xp = (const int4*)(xq + v * (K / 32 * 32) + b * 32);
            const int4 xa = xp[0], xb = xp[1];
            const int2 meta = xmeta[v * (K / 32) + b];
            int s = 0;
            s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
            s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
            acc[v] += d * __int_as_float(meta.x) * (float)(s - 8 * meta.y);
        }
    }
#pragma unroll
    for (int v = 0; v < NB; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[v * ystride + row] = a;
    }
}

template <int NB, int WARPS, int GROUPS = 1, bool ENCODED = false>
__global__ void q4_mma(const uint8_t* w, long K, long rows, const int8_t* xq, const int2* xmeta, float* y, long ystride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ float partial[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int group = lane >> 2, thread = lane & 3;
    const long tile_index = blockIdx.x / GROUPS;
    const long group_index = blockIdx.x % GROUPS;
    const long first_row = tile_index * 16;
    const long nb = K / 32;
    xq += group_index * NB * K;
    xmeta += group_index * NB * nb;
    y += group_index * NB * ystride;
    float sum[32 / WARPS][NB / 8][4] = {};
    for (long start = 0; start < nb; start += 32) {
        const uint8_t* panel = w + ((size_t)tile_index * nb + start) * kRawTile;
        bool compact = false;
        if constexpr (ENCODED) {
            const uint32_t offset = ((const uint32_t*)w)[(tile_index * nb + start) / kPanelTiles];
            panel = w + (offset & ~1u);
            compact = offset & 1;
        }
#pragma unroll
        for (int strip = 0; strip < 32 / WARPS; ++strip) {
            const int k = warp + strip * WARPS;
            const uint8_t* tile = panel + k * (compact ? kTileQuants : kRawTile);
            const uint32_t* quants = (const uint32_t*)tile;
            const uint16_t* scales = (const uint16_t*)(tile + 256);
            const uint32_t u = quants[group * 4 + thread];
            const uint32_t v = quants[(group + 8) * 4 + thread];
            const float da = __half2float(__ushort_as_half(compact ? unpack_scale(panel, k, group) : scales[group]));
            const float db = __half2float(__ushort_as_half(compact ? unpack_scale(panel, k, group + 8) : scales[group + 8]));
            constexpr uint32_t mask = 0x0f0f0f0fu;
#pragma unroll
            for (int n = 0; n < NB / 8; ++n) {
                const uint32_t* xp = (const uint32_t*)(xq + (n * 8 + group) * K + (start + k) * 32);
                const uint32_t xb0 = xp[thread], xb1 = xp[thread + 4];
                int c0, c1, c2, c3;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {0,0,0,0};"
                    : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                    : "r"(u & mask),"r"(v & mask),"r"((u>>4)&mask),"r"((v>>4)&mask),"r"(xb0),"r"(xb1));
                const int col = n * 8 + 2 * thread;
                const int2 m0 = xmeta[col * nb + start + k], m1 = xmeta[(col + 1) * nb + start + k];
                sum[strip][n][0] += da * __int_as_float(m0.x) * (float)(c0 - 8 * m0.y);
                sum[strip][n][1] += da * __int_as_float(m1.x) * (float)(c1 - 8 * m1.y);
                sum[strip][n][2] += db * __int_as_float(m0.x) * (float)(c2 - 8 * m0.y);
                sum[strip][n][3] += db * __int_as_float(m1.x) * (float)(c3 - 8 * m1.y);
            }
        }
    }
#pragma unroll
    for (int strip = 0; strip < 32 / WARPS; ++strip) {
        const int k = warp + strip * WARPS;
#pragma unroll
        for (int n = 0; n < NB / 8; ++n) {
            const int col = n * 8 + 2 * thread;
            const int slot = (k + 8 * thread + group) & 31;
            partial[(col * 16 + group) * 32 + slot] = sum[strip][n][0];
            partial[((col + 1) * 16 + group) * 32 + slot] = sum[strip][n][1];
            partial[(col * 16 + group + 8) * 32 + slot] = sum[strip][n][2];
            partial[((col + 1) * 16 + group + 8) * 32 + slot] = sum[strip][n][3];
        }
    }
    __syncthreads();
    for (int o = warp; o < NB * 16; o += WARPS) {
        const int rotation = 8 * (o / 32) + (o % 8);
        float total = partial[o * 32 + ((lane + rotation) & 31)];
        for (int delta = 16; delta; delta >>= 1) total += __shfl_down_sync(0xffffffffu, total, delta);
        const long row = first_row + o % 16;
        if (lane == 0 && row < rows) y[(o / 16) * ystride + row] = total;
    }
#endif
}

static void pack_q4_tiles(uint8_t* destination, const uint8_t* source, long K, long rows) {
    const long blocks = K / 32;
    for (long first = 0; first < rows; first += 16) {
        for (long b = 0; b < blocks; ++b) {
            uint8_t* tile = destination + ((size_t)(first / 16) * blocks + b) * 288;
            for (long r = 0; r < 16; ++r) {
                const uint8_t* from = source + ((size_t)(first + r) * blocks + b) * 18;
                memcpy(tile + r * 16, from + 2, 16);
                memcpy(tile + 256 + r * 2, from, 2);
            }
        }
    }
}

// ---- probe harness ----
struct Case { long K; long rows; };

static bool run_case(long K, long rows, int nb_mma, int groups, bool deep_k) {
    // deep_k selects the NB=16/GROUPS=4 wide-K branch the decoder uses for K>=17408
    const long nb = K / 32;
    const size_t rowbytes = (size_t)nb * 18;
    const size_t wbytes = (size_t)rows * rowbytes;
    const size_t wtiles = (size_t)((rows + 15) / 16) * nb * kRawTile;   // tiled layout rounds rows up to 16
    std::vector<uint8_t> raw(wbytes), packed(wtiles);
    srand(1234 + (int)K + (int)rows);
    for (size_t i = 0; i < wbytes; i += 18) {
        raw[i] = (uint8_t)(rand() & 0xff); raw[i + 1] = (uint8_t)(rand() & 0xff);   // fp16 scale bits
        for (int j = 2; j < 18; ++j) raw[i + j] = (uint8_t)(rand() & 0xff);
    }
    pack_q4_tiles(packed.data(), raw.data(), K, rows);

    // activations sized for the verify WIDTH (the kernel quantizes/reads nb_mma
    // vectors), not for rows — rows can be fewer (partial-tile masking)
    const long nvec = nb_mma;
    std::vector<float> x((size_t)nvec * K);
    for (auto& v : x) v = (rand() % 2000 - 1000) / 997.f;

    uint8_t *d_w, *d_w_raw;
    int8_t* d_xq;
    int2* d_meta;
    float *d_x, *d_test, *d_ref, *d_dp4a;
    check(cudaMalloc(&d_w, wtiles), "w tiled");
    check(cudaMalloc(&d_w_raw, wbytes), "w raw");
    check(cudaMalloc(&d_x, x.size() * 4), "x");
    const size_t qbytes = (size_t)nvec * K, mbytes = (size_t)nvec * nb * sizeof(int2);
    check(cudaMalloc((void**)&d_xq, qbytes), "xq");
    check(cudaMalloc(&d_meta, mbytes), "meta");
    check(cudaMalloc(&d_test, (size_t)rows * nb_mma * 4), "y_test");
    check(cudaMalloc(&d_ref, (size_t)rows * nb_mma * 4), "y_ref");
    check(cudaMalloc(&d_dp4a, (size_t)rows * nb_mma * 4), "y_dp4a");
    check(cudaMemcpy(d_w, packed.data(), wtiles, cudaMemcpyHostToDevice), "w h2d");
    check(cudaMemcpy(d_w_raw, raw.data(), wbytes, cudaMemcpyHostToDevice), "w raw h2d");
    check(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x h2d");
    k_quant_x<<<(unsigned)((nvec * nb + 127) / 128), 128>>>(d_x, nvec * nb, d_xq, d_meta);
    check(cudaDeviceSynchronize(), "quant");

    // fused launch (the decoder's new dispatch)
    const unsigned tiles = (unsigned)((rows + 15) / 16);
    if (deep_k) q4_mma<16, 8, 4><<<4 * tiles, 256, 16 * 16 * 32 * 4>>>(d_w, K, rows, d_xq, d_meta, d_test, rows);
    else if (nb_mma == 48) q4_mma<8, 8, 6><<<6 * tiles, 256, 16 * 8 * 32 * 4>>>(d_w, K, rows, d_xq, d_meta, d_test, rows);
    else q4_mma<8, 8, 8><<<8 * tiles, 256, 16 * 8 * 32 * 4>>>(d_w, K, rows, d_xq, d_meta, d_test, rows);

    // reference: per-group decomposition (what GROUPS must reproduce bit-exactly).
    // Each call keeps the FULL rows (the guard is on the global row tile*16+o%16)
    // and offsets only the activation/y pointers — exactly what GROUPS does.
    check(cudaMemset(d_ref, 0, (size_t)rows * nb_mma * 4), "ref zero");
    const int per_group = deep_k ? 16 : 8;
    const int gcount = nb_mma / per_group;
    for (int g = 0; g < gcount; ++g) {
        if (deep_k) q4_mma<16, 8, 1><<<tiles, 256, 16 * 16 * 32 * 4>>>(d_w, K, rows,
            d_xq + (size_t)g * 16 * K, d_meta + (size_t)g * 16 * nb, d_ref + (size_t)g * 16 * rows, rows);
        else q4_mma<8, 8, 1><<<tiles, 256, 16 * 8 * 32 * 4>>>(d_w, K, rows,
            d_xq + (size_t)g * 8 * K, d_meta + (size_t)g * 8 * nb, d_ref + (size_t)g * 8 * rows, rows);
    }
    // dp4a fallback cross-check (reads the RAW row-major layout, as in the decoder)
    if (nb_mma == 64) k_mat_q4<64><<<(unsigned)((rows + 3) / 4), 128>>>(d_w_raw, K, rows, d_xq, d_meta, d_dp4a, rows);
    else k_mat_q4<48><<<(unsigned)((rows + 3) / 4), 128>>>(d_w_raw, K, rows, d_xq, d_meta, d_dp4a, rows);
    check(cudaDeviceSynchronize(), "run");

    std::vector<float> test((size_t)rows * nb_mma), ref((size_t)rows * nb_mma), dp4a((size_t)rows * nb_mma);
    check(cudaMemcpy(test.data(), d_test, test.size() * 4, cudaMemcpyDeviceToHost), "test d2h");
    check(cudaMemcpy(ref.data(), d_ref, ref.size() * 4, cudaMemcpyDeviceToHost), "ref d2h");
    check(cudaMemcpy(dp4a.data(), d_dp4a, dp4a.size() * 4, cudaMemcpyDeviceToHost), "dp4a d2h");

    size_t bit_diff = 0; double worst_rel = 0, worst_dp4a = 0;
    int shown = 0;
    for (size_t i = 0; i < test.size(); ++i) {
        if (memcmp(&test[i], &ref[i], 4) != 0) {
            ++bit_diff;
            if (shown++ < 6)
                printf("  diff at v=%zu row=%zu: fused=%.9g ref=%.9g (delta=%.3g)\n",
                       i / (size_t)rows, i % (size_t)rows, test[i], ref[i], (double)test[i] - (double)ref[i]);
        }
        const double scale = std::max(1e-9, (double)std::abs(ref[i]));
        worst_rel = std::max(worst_rel, std::abs((double)test[i] - (double)ref[i]) / scale);
        worst_dp4a = std::max(worst_dp4a, std::abs((double)dp4a[i] - (double)test[i]) / scale);
    }
    cudaFree(d_w); cudaFree(d_w_raw); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_meta);
    cudaFree(d_test); cudaFree(d_ref); cudaFree(d_dp4a);

    const char* label = deep_k ? "q4_mma<16,8,4>" : nb_mma == 48 ? "q4_mma<8,8,6>" : "q4_mma<8,8,8>";
    printf("case K=%ld rows=%2ld %s vs %dx per-group: bit_diff=%zu worst_rel=%.3g | dp4a worst_rel=%.3g %s\n",
           K, rows, label, gcount, bit_diff, worst_rel, worst_dp4a,
           (bit_diff == 0 && worst_dp4a < 1e-3) ? "PASS" : "FAIL");
    return bit_diff == 0 && worst_dp4a < 1e-3;
}

int main() {
    check(cudaSetDevice(0), "device");
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0), "props");
    printf("device=%s sm=%d.%d sm_per_multiproc=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    struct Attr { const char* name; const void* fn; } attrs[] = {
        {"q4_mma<8,8,8>", (const void*)q4_mma<8, 8, 8>},
        {"q4_mma<8,8,6>", (const void*)q4_mma<8, 8, 6>},
        {"q4_mma<16,8,4>", (const void*)q4_mma<16, 8, 4>},
        {"k_mat_q4<64>", (const void*)k_mat_q4<64>},
    };
    for (auto& a : attrs) {
        cudaFuncAttributes at{};
        if (cudaFuncGetAttributes(&at, a.fn) != cudaSuccess) { printf("attr,%s,FAILED\n", a.name); continue; }
        printf("attr,%s,regs=%d,smem_static=%zu,max_threads=%d,binary=v%d\n", a.name,
               at.numRegs, at.sharedSizeBytes, at.maxThreadsPerBlock, at.binaryVersion);
    }

    bool ok = true;
    // width 64: every K class the packed path selects (K%1024==0, K>=5120) plus K=4096 masking cases
    ok &= run_case(4096, 33, 64, 8, false);    // partial tile + partial group masking
    ok &= run_case(4096, 48, 64, 8, false);
    ok &= run_case(4096, 63, 64, 8, false);
    ok &= run_case(5120, 64, 64, 8, false);
    ok &= run_case(12288, 64, 64, 8, false);
    ok &= run_case(17408, 64, 64, 8, true);    // wide-K branch
    // width 48 (held in reserve for a future non-power-of-two width)
    ok &= run_case(4096, 48, 48, 6, false);
    ok &= run_case(12288, 47, 48, 6, false);   // odd rows: masking in both dims
    printf("%s\n", ok ? "E166_MMA_PROBE_ALL_PASS" : "E166_MMA_PROBE_FAILURES");
    return ok ? 0 : 1;
}
