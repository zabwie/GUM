// E051 microbench: (A) tiled smem xq+xs/xsum GEMV vs production k_mat_q4 (v0), bit-exact
// comparison; (B) transposed-state delta-net vs production k_delta_net_multi, bit-exact.
//
// Tiled GEMV design (E050 fix): block = WRPS warps, warp = 1 row; K is walked in chunks of
// CB quanta-blocks; per chunk the block cooperatively stages xq quanta into smem at 36-B
// padded stride (9 uint32 per 32-elem block; word-stride 9 is coprime with 32 banks ->
// conflict-free 4-B reads — dodges the E050 "int4 smem half-rate/8-way" negative) and
// xs/xsum fused as one dense int2. Accumulation keeps the EXACT production order:
// lane L visits global blocks b = chunk_b0 + L, +32, ... ascending, chunks ascending —
// identical to v0's `for (b = lane; b < nb; b += 32)` — and the same warp tree reduction,
// so outputs are bit-identical to v0 (memcmp, not tolerance).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

static uint16_t host_half_bits(float x) {
    const __half h = __float2half_rn(x);
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// ---------------- production kernel (real_decode.cu k_mat_q4), NB=16 specialization ----
__global__ void k_v0(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32; const size_t rowbytes = (size_t)nb * 18;
    const uint8_t* r = w + (size_t)row * rowbytes;
    float acc[16];
#pragma unroll
    for (int v = 0; v < 16; ++v) acc[v] = 0.f;
    for (long b = lane; b < nb; b += 32) {
        const uint16_t* blk = (const uint16_t*)(r + (size_t)b * 18);
        const float d = __half2float(__ushort_as_half(blk[0]));
        uint32_t q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
        const uint32_t lo0 = q[0] & 0x0F0F0F0Fu, lo1 = q[1] & 0x0F0F0F0Fu, lo2 = q[2] & 0x0F0F0F0Fu, lo3 = q[3] & 0x0F0F0F0Fu;
        const uint32_t hi0 = (q[0] >> 4) & 0x0F0F0F0Fu, hi1 = (q[1] >> 4) & 0x0F0F0F0Fu, hi2 = (q[2] >> 4) & 0x0F0F0F0Fu, hi3 = (q[3] >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int v = 0; v < 16; ++v) {
            const int4* xp = (const int4*)(xq + v * (K / 32 * 32) + b * 32);
            const int4 xa = xp[0], xb = xp[1];
            int s = 0;
            s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
            s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
            acc[v] += d * xs[v * (K / 32) + b] * (float)(s - 8 * xsum[v * (K / 32) + b]);
        }
    }
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[v * ystride + row] = a;
    }
}

// ---------------- E051 tiled kernel ---------------------------------------------------
// WRPS warps per block (each owns one row), chunks of CB 32-elem quanta blocks.
// smem: quanta uint32 tile [NB][CB][9] (8 live + 1 pad word) then int2 tile [NB][CB].
template <int NB, int WRPS>
__global__ void k_tiled(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride, int CB) {
    extern __shared__ uint32_t smem[];
    uint32_t* tq = smem;                       // [NB][CB][9]
    int2* ts = (int2*)(tq + (size_t)NB * CB * 9);   // [NB][CB] = {float_bits(xs), xsum}
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const long nb = K / 32;
    const size_t rowbytes = (size_t)nb * 18;
    const long row = (long)blockIdx.x * WRPS + (tid >> 5);
    float acc[NB];
#pragma unroll
    for (int v = 0; v < NB; ++v) acc[v] = 0.f;
    for (long c0 = 0; c0 < nb; c0 += CB) {
        const int cb = (int)min((long)CB, nb - c0);
        // cooperative tile stage (ALL threads participate — no early return before syncs)
        for (int i = tid; i < NB * cb * 8; i += WRPS * 32) {
            const int v = i / (cb * 8), r = i % (cb * 8), b = r >> 3, u = r & 7;
            tq[((size_t)v * CB + b) * 9 + u] = *(const uint32_t*)(xq + (size_t)v * (K / 32 * 32) + (size_t)(c0 + b) * 32 + u * 4);
        }
        for (int i = tid; i < NB * cb; i += WRPS * 32) {
            const int v = i / cb, b = i % cb;
            ts[(size_t)v * CB + b] = make_int2(__float_as_int(xs[(size_t)v * nb + c0 + b]), xsum[(size_t)v * nb + c0 + b]);
        }
        __syncthreads();
        if (row < rows) {
            const uint8_t* r = w + (size_t)row * rowbytes;
            for (long b = lane; b < cb; b += 32) {
                const uint16_t* blk = (const uint16_t*)(r + (size_t)(c0 + b) * 18);
                const float d = __half2float(__ushort_as_half(blk[0]));
                uint32_t q[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
                const uint32_t lo0 = q[0] & 0x0F0F0F0Fu, lo1 = q[1] & 0x0F0F0F0Fu, lo2 = q[2] & 0x0F0F0F0Fu, lo3 = q[3] & 0x0F0F0F0Fu;
                const uint32_t hi0 = (q[0] >> 4) & 0x0F0F0F0Fu, hi1 = (q[1] >> 4) & 0x0F0F0F0Fu, hi2 = (q[2] >> 4) & 0x0F0F0F0Fu, hi3 = (q[3] >> 4) & 0x0F0F0F0Fu;
#pragma unroll
                for (int v = 0; v < NB; ++v) {
                    const uint32_t* tqvb = tq + ((size_t)v * CB + b) * 9;
                    const uint32_t xa_x = tqvb[0], xa_y = tqvb[1], xa_z = tqvb[2], xa_w = tqvb[3];
                    const uint32_t xb_x = tqvb[4], xb_y = tqvb[5], xb_z = tqvb[6], xb_w = tqvb[7];
                    int s = 0;
                    s = __dp4a((int)lo0, (int)xa_x, s); s = __dp4a((int)lo1, (int)xa_y, s); s = __dp4a((int)lo2, (int)xa_z, s); s = __dp4a((int)lo3, (int)xa_w, s);
                    s = __dp4a((int)hi0, (int)xb_x, s); s = __dp4a((int)hi1, (int)xb_y, s); s = __dp4a((int)hi2, (int)xb_z, s); s = __dp4a((int)hi3, (int)xb_w, s);
                    const int2 p = ts[(size_t)v * CB + b];
                    acc[v] += d * __int_as_float(p.x) * (float)(s - 8 * p.y);
                }
            }
        }
        __syncthreads();
    }
    if (row < rows) {
#pragma unroll
        for (int v = 0; v < NB; ++v) {
            float a = acc[v];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
            if (lane == 0) y[v * ystride + row] = a;
        }
    }
}

// ---------------- E051 multi-row kernel ------------------------------------------------
// R rows per warp share the xq + xs/xsum loads (LSU amortization — the NB=16 collapse is
// LSU/L1-bound, not DRAM-bound: v0 issues ~70 loads per 128 dp4a; R=4 issues ~66 per 512).
// xs/xsum staged in smem as dense int2, chunked over K to fit. xq/weights stay global.
// Per-row b sequence = lane, lane+32, ... ascending -> bit-exact vs v0.
template <int NB, int R>
__global__ void k_mr(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride, int CB) {
    extern __shared__ int2 smem2[];            // [NB][CB] chunk tile
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const long nb = K / 32;
    const size_t rowbytes = (size_t)nb * 18;
    const long row0 = (long)blockIdx.x * (4 * R) + (tid >> 5) * R;
    float acc[R][NB];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int v = 0; v < NB; ++v) acc[r][v] = 0.f;
    for (long c0 = 0; c0 < nb; c0 += CB) {
        const int cb = (int)min((long)CB, nb - c0);
        for (int i = tid; i < NB * cb; i += blockDim.x) {
            const int v = i / cb, b = i % cb;
            smem2[(size_t)v * CB + b] = make_int2(__float_as_int(xs[(size_t)v * nb + c0 + b]), xsum[(size_t)v * nb + c0 + b]);
        }
        __syncthreads();
        for (long b = lane; b < cb; b += 32) {
            const long gb = c0 + b;
            // per-row weight quanta
            uint32_t lo[R][4], hi[R][4]; float d[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r >= rows) continue;
                const uint16_t* blk = (const uint16_t*)(w + (size_t)(row0 + r) * rowbytes + (size_t)gb * 18);
                d[r] = __half2float(__ushort_as_half(blk[0]));
                uint32_t q[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
#pragma unroll
                for (int j = 0; j < 4; ++j) { lo[r][j] = q[j] & 0x0F0F0F0Fu; hi[r][j] = (q[j] >> 4) & 0x0F0F0F0Fu; }
            }
#pragma unroll
            for (int v = 0; v < NB; ++v) {
                const int4* xp = (const int4*)(xq + v * (K / 32 * 32) + gb * 32);
                const int4 xa = xp[0], xb = xp[1];
                const int2 p = smem2[(size_t)v * CB + b];
                const float xsc = __int_as_float(p.x);
                const int corr = 8 * p.y;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (row0 + r >= rows) continue;
                    int s = 0;
                    s = __dp4a((int)lo[r][0], xa.x, s); s = __dp4a((int)lo[r][1], xa.y, s); s = __dp4a((int)lo[r][2], xa.z, s); s = __dp4a((int)lo[r][3], xa.w, s);
                    s = __dp4a((int)hi[r][0], xb.x, s); s = __dp4a((int)hi[r][1], xb.y, s); s = __dp4a((int)hi[r][2], xb.z, s); s = __dp4a((int)hi[r][3], xb.w, s);
                    acc[r][v] += d[r] * xsc * (float)(s - corr);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= rows) continue;
#pragma unroll
        for (int v = 0; v < NB; ++v) {
            float a = acc[r][v];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
            if (lane == 0) y[v * ystride + row0 + r] = a;
        }
    }
}

// ---------------- delta-net: production (old layout [h][j][i]) ------------------------
__global__ void k_dn_v0(float* state, const float* qkv, const float* g, const float* beta, float* out,
                        int S, int H, int ngroup, float scale, long qkv_stride, long out_stride, int B) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= S) return;
    const int hk = h % ngroup;
    float* st = state + (size_t)h * S * S;
    for (int p = 0; p < B; ++p) {
        const float* qkvp = qkv + (size_t)p * qkv_stride;
        const float* kh = qkvp + (size_t)S * ngroup + (size_t)hk * S;
        const float* qh = qkvp + (size_t)hk * S;
        const float* vh = qkvp + 2 * (size_t)S * ngroup + (size_t)h * S;
        float* row = st + (size_t)j * S;
        const float eg = expf(g[(size_t)p * H + h]);
        for (int i = 0; i < S; ++i) row[i] *= eg;
        float dot = 0.f; for (int i = 0; i < S; ++i) dot += row[i] * kh[i];
        const float dj = (vh[j] - dot) * beta[(size_t)p * H + h];
        for (int i = 0; i < S; ++i) row[i] += dj * kh[i];
        float o = 0.f; for (int i = 0; i < S; ++i) o += row[i] * qh[i];
        out[(size_t)p * out_stride + (size_t)h * S + j] = o * scale;
    }
}
// ---------------- delta-net: E051 transposed state [h][i][j], j-split blocks ----------
// Same FMA sequence per j (i ascending, serial) -> bit-exact vs k_dn_v0; warp lanes map to
// consecutive j so state accesses coalesce (v0 had 512-B lane stride = 32x over-fetch).
__global__ void k_dn_t(float* state, const float* qkv, const float* g, const float* beta, float* out,
                       int S, int H, int ngroup, float scale, long qkv_stride, long out_stride, int B) {
    const int h = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= S) return;
    const int hk = h % ngroup;
    float* st = state + (size_t)h * S * S;
    for (int p = 0; p < B; ++p) {
        const float* qkvp = qkv + (size_t)p * qkv_stride;
        const float* kh = qkvp + (size_t)S * ngroup + (size_t)hk * S;
        const float* qh = qkvp + (size_t)hk * S;
        const float* vh = qkvp + 2 * (size_t)S * ngroup + (size_t)h * S;
        const float eg = expf(g[(size_t)p * H + h]);
        for (int i = 0; i < S; ++i) st[(size_t)i * S + j] *= eg;
        float dot = 0.f; for (int i = 0; i < S; ++i) dot += st[(size_t)i * S + j] * kh[i];
        const float dj = (vh[j] - dot) * beta[(size_t)p * H + h];
        for (int i = 0; i < S; ++i) st[(size_t)i * S + j] += dj * kh[i];
        float o = 0.f; for (int i = 0; i < S; ++i) o += st[(size_t)i * S + j] * qh[i];
        out[(size_t)p * out_stride + (size_t)h * S + j] = o * scale;
    }
}
// E051b: fused scale+dot pass (t = st*eg rounds identically to old scale-then-read), then
// add pass, then output dot — 3 state passes instead of 4. Still bit-exact.
__global__ void k_dn_t2(float* state, const float* qkv, const float* g, const float* beta, float* out,
                        int S, int H, int ngroup, float scale, long qkv_stride, long out_stride, int B) {
    const int h = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= S) return;
    const int hk = h % ngroup;
    float* st = state + (size_t)h * S * S;
    for (int p = 0; p < B; ++p) {
        const float* qkvp = qkv + (size_t)p * qkv_stride;
        const float* kh = qkvp + (size_t)S * ngroup + (size_t)hk * S;
        const float* qh = qkvp + (size_t)hk * S;
        const float* vh = qkvp + 2 * (size_t)S * ngroup + (size_t)h * S;
        const float eg = expf(g[(size_t)p * H + h]);
        float dot = 0.f;
        for (int i = 0; i < S; ++i) { const float t = st[(size_t)i * S + j] * eg; st[(size_t)i * S + j] = t; dot += t * kh[i]; }
        const float dj = (vh[j] - dot) * beta[(size_t)p * H + h];
        for (int i = 0; i < S; ++i) st[(size_t)i * S + j] += dj * kh[i];
        float o = 0.f; for (int i = 0; i < S; ++i) o += st[(size_t)i * S + j] * qh[i];
        out[(size_t)p * out_stride + (size_t)h * S + j] = o * scale;
    }
}
// transposed-in helper for the bench: old [h][j][i] -> new [h][i][j]
__global__ void k_transpose(const float* in, float* out, int S, int H) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long n = (long)H * S * S;
    if (idx >= n) return;
    const int h = idx / (S * S), r = idx % (S * S), j = r / S, i = r % S;
    out[(long)h * S * S + (long)i * S + j] = in[idx];
}

int main() {
    // ================= A) GEMV =================
    {
        const long K = 5120, rows = 11652;
        const long nb = K / 32;
        const size_t raw_bytes = (size_t)rows * nb * 18;
        uint8_t* h_raw = (uint8_t*)malloc(raw_bytes);
        srand(42);
        for (size_t i = 0; i < raw_bytes; ++i) h_raw[i] = rand() & 0xFF;
        for (long r = 0; r < rows; ++r)
            for (long b = 0; b < nb; ++b)
                *(uint16_t*)(h_raw + ((size_t)r * nb + b) * 18) = host_half_bits(0.001f * (float)((r + b) % 97 + 1));
        int8_t* h_x = (int8_t*)malloc(16 * K);
        float* h_xs = (float*)malloc(16 * nb * 4); int* h_xsum = (int*)malloc(16 * nb * 4);
        for (long i = 0; i < 16 * K; ++i) h_x[i] = (rand() % 31) - 15;
        for (long v = 0; v < 16; ++v) for (long b = 0; b < nb; ++b) { h_xs[v*nb+b] = 0.01f; h_xsum[v*nb+b] = (int)(rand() % 7) - 3; }
        uint8_t* d_raw; int8_t* d_x; float *d_xs, *d_y0, *d_y1; int* d_xsum;
        CHECK(cudaMalloc(&d_raw, raw_bytes));
        CHECK(cudaMalloc(&d_x, 16*K)); CHECK(cudaMalloc(&d_xs, 16*nb*4)); CHECK(cudaMalloc(&d_xsum, 16*nb*4));
        CHECK(cudaMalloc(&d_y0, 16*rows*4)); CHECK(cudaMalloc(&d_y1, 16*rows*4));
        CHECK(cudaMemcpy(d_raw, h_raw, raw_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_x, h_x, 16*K, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_xs, h_xs, 16*nb*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_xsum, h_xsum, 16*nb*4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        const int iters = 30;
        float* h_y0 = (float*)malloc(16*rows*4); float* h_y1 = (float*)malloc(16*rows*4);

        // v0 reference + timing
        const unsigned g0 = (unsigned)((rows + 3) / 4);
        k_v0<<<g0,128>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y0, rows);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) k_v0<<<g0,128>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y0, rows);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms0; CHECK(cudaEventElapsedTime(&ms0, e0, e1));
        printf("v0 current      : %.3f ms  (%.1f GB/s equiv)\n", ms0/iters, raw_bytes/1e9/(ms0/iters/1e3));

        struct Cfg { int wrps; int cb; };
        Cfg cfgs[] = {{16,128},{16,64},{16,32}};
        for (Cfg c : cfgs) {
            const size_t smem = (size_t)16 * c.cb * (36 + 8);
            CHECK(cudaFuncSetAttribute(k_tiled<16,16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));
            if (smem > 99*1024) continue;
            const unsigned g = (unsigned)((rows + c.wrps - 1) / c.wrps);
            bool ok = true;
            k_tiled<16,16><<<g, 512, smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows, c.cb);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_y0, d_y0, 16*rows*4, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_y1, d_y1, 16*rows*4, cudaMemcpyDeviceToHost));
            if (memcmp(h_y0, h_y1, 16*rows*4) != 0) { ok = false; for (long i=0;i<16*rows;++i) if (h_y0[i]!=h_y1[i]) { printf("  first diff at %ld: %a vs %a\n", i, h_y0[i], h_y1[i]); break; } }
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < iters; ++i) k_tiled<16,16><<<g, 512, smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows, c.cb);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            printf("tiled w=%d cb=%3d : %.3f ms  (%.1f GB/s equiv)  bitexact=%s\n",
                   c.wrps, c.cb, ms/iters, raw_bytes/1e9/(ms/iters/1e3), ok ? "YES" : "NO");
        }
        // multi-row: R rows per warp amortize xq+xs loads
        CHECK(cudaFuncSetAttribute(k_mr<16,2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));
        CHECK(cudaFuncSetAttribute(k_mr<16,4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));
        CHECK(cudaFuncSetAttribute(k_mr<16,8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));
        const int CB = (int)nb;   // K=5120: single chunk, 20.5 KB smem
        const size_t smem_mr = (size_t)16 * CB * 8;
        for (int R : {2, 4, 8}) {
            const unsigned g = (unsigned)((rows + 4*R - 1) / (4*R));
            bool ok = true;
            if (R == 2) k_mr<16,2><<<g, 128, smem_mr>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows, CB);
            else if (R == 4) k_mr<16,4><<<g, 128, smem_mr>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows, CB);
            else k_mr<16,8><<<g, 128, smem_mr>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows, CB);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_y0, d_y0, 16*rows*4, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_y1, d_y1, 16*rows*4, cudaMemcpyDeviceToHost));
            if (memcmp(h_y0, h_y1, 16*rows*4) != 0) { ok = false; for (long i=0;i<16*rows;++i) if (h_y0[i]!=h_y1[i]) { printf("  first diff at %ld: %a vs %a\n", i, h_y0[i], h_y1[i]); break; } }
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < iters; ++i) {
                if (R == 2) k_mr<16,2><<<g, 128, smem_mr>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows, CB);
                else if (R == 4) k_mr<16,4><<<g, 128, smem_mr>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows, CB);
                else k_mr<16,8><<<g, 128, smem_mr>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows, CB);
            }
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            printf("mr    R=%d      : %.3f ms  (%.1f GB/s equiv)  bitexact=%s\n",
                   R, ms/iters, raw_bytes/1e9/(ms/iters/1e3), ok ? "YES" : "NO");
        }
        free(h_raw); free(h_x); free(h_xs); free(h_xsum); free(h_y0); free(h_y1);
        CHECK(cudaFree(d_raw)); CHECK(cudaFree(d_x)); CHECK(cudaFree(d_xs)); CHECK(cudaFree(d_xsum)); CHECK(cudaFree(d_y0)); CHECK(cudaFree(d_y1));
    }

        // ================= B) delta-net =================
    // Production-shaped inputs: q/k L2-normalized per (position, head) (the model l2-norms
    // them), g <= 0, beta in (0,1) -> contractive dynamics, so bit-comparisons are MEANINGFUL
    // (the first draft of this bench ran one variant on an uninitialized buffer and unnormalized
    // k, which diverged to identical inf/NaN on both sides -- vacuous YES).
    {
        const int S = 128, H = 48, NG = 16, B = 17;
        const long qkv_stride = 2L*S*NG + (long)H*S;   // 6144
        const size_t st_n = (size_t)H * S * S;
        float* h_qkv = (float*)malloc(qkv_stride * B * 4);
        float* h_g = (float*)malloc((size_t)H * B * 4);
        float* h_beta = (float*)malloc((size_t)H * B * 4);
        srand(7);
        for (long i = 0; i < qkv_stride * B; ++i) h_qkv[i] = 0.5f * ((rand() % 2000) - 1000) / 1000.f;
        // l2-normalize q (heads 0..NG-1) and k (heads NG..2NG-1) per position
        for (int p = 0; p < B; ++p) {
            float* base = h_qkv + (size_t)p * qkv_stride;
            for (int hgrp = 0; hgrp < 2 * NG; ++hgrp) {
                float* hh = base + (size_t)hgrp * S;
                double ss = 0; for (int i = 0; i < S; ++i) ss += (double)hh[i] * hh[i];
                const float inv = 1.f / sqrtf((float)ss);
                for (int i = 0; i < S; ++i) hh[i] *= inv;
            }
        }
        for (long i = 0; i < (long)H * B; ++i) { h_g[i] = -0.05f * ((rand() % 100) / 100.f); h_beta[i] = 0.8f + 0.1f * ((rand() % 100) / 100.f); }
        float* h_st_init = (float*)malloc(st_n * 4);
        for (size_t i = 0; i < st_n; ++i) h_st_init[i] = 0.02f * ((rand() % 2000) - 1000) / 1000.f;
        const size_t out_n = (size_t)B * H * S;
        float* h_out0 = (float*)malloc(out_n * 4); float* h_out1 = (float*)malloc(out_n * 4);
        float *h_st0f = (float*)malloc(st_n * 4); float *h_t = (float*)malloc(st_n * 4);
        float *d_qkv, *d_g, *d_beta, *d_st, *d_st0, *d_out0, *d_out1, *d_t;
        CHECK(cudaMalloc(&d_qkv, qkv_stride*B*4)); CHECK(cudaMalloc(&d_g, (size_t)H*B*4)); CHECK(cudaMalloc(&d_beta, (size_t)H*B*4));
        CHECK(cudaMalloc(&d_st, st_n*4)); CHECK(cudaMalloc(&d_st0, st_n*4)); CHECK(cudaMalloc(&d_out0, out_n*4)); CHECK(cudaMalloc(&d_out1, out_n*4)); CHECK(cudaMalloc(&d_t, st_n*4));
        CHECK(cudaMemcpy(d_qkv, h_qkv, qkv_stride*B*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_g, h_g, (size_t)H*B*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_beta, h_beta, (size_t)H*B*4, cudaMemcpyHostToDevice));
        const float scale = 1.f / sqrtf((float)S);
        cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        const int iters = 100;

        auto cmp_run = [&](const char* name, auto launch, float* ms_out) {
            // correctness: ONE run each from the same initial state
            CHECK(cudaMemcpy(d_st0, h_st_init, st_n*4, cudaMemcpyHostToDevice));
            k_dn_v0<<<H, S>>>(d_st0, d_qkv, d_g, d_beta, d_out0, S, H, NG, scale, qkv_stride, (long)H*S, B);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(d_st, h_st_init, st_n*4, cudaMemcpyHostToDevice));
            k_transpose<<<(unsigned)((st_n + 255) / 256), 256>>>(d_st, d_t, S, H);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(d_st, d_t, st_n*4, cudaMemcpyDeviceToDevice));
            launch(d_st, d_out1);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out0, d_out0, out_n*4, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_out1, d_out1, out_n*4, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_st0f, d_st0, st_n*4, cudaMemcpyDeviceToHost));
            k_transpose<<<(unsigned)((st_n + 255) / 256), 256>>>(d_st, d_t, S, H);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_t, d_t, st_n*4, cudaMemcpyDeviceToHost));
            bool out_ok = true, st_ok = true;
            for (size_t i = 0; i < out_n; ++i) if (h_out0[i] != h_out1[i]) { out_ok = false; printf("  out diff at %zu: %a vs %a\n", i, h_out0[i], h_out1[i]); break; }
            for (size_t i = 0; i < st_n; ++i) if (h_st0f[i] != h_t[i]) { st_ok = false; printf("  state diff at %zu: %a vs %a\n", i, h_st0f[i], h_t[i]); break; }
            // timing: repeat on the same buffers (state evolution irrelevant for speed)
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < iters; ++i) launch(d_st, d_out1);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            CHECK(cudaEventElapsedTime(ms_out, e0, e1));
            printf("%s: %.3f ms  out_bitexact=%s state_bitexact=%s\n", name, *ms_out/iters, out_ok ? "YES" : "NO", st_ok ? "YES" : "NO");
        };
        float ms0 = 0, ms1 = 0, ms2 = 0;
        // reference timing first
        CHECK(cudaMemcpy(d_st0, h_st_init, st_n*4, cudaMemcpyHostToDevice));
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) k_dn_v0<<<H, S>>>(d_st0, d_qkv, d_g, d_beta, d_out0, S, H, NG, scale, qkv_stride, (long)H*S, B);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms0, e0, e1));
        printf("dn v0  (48x128) : %.3f ms\n", ms0/iters);

        auto l_t = [&](float* st, float* o1) {
            k_dn_t<<<dim3(H, S/32), 32>>>(st, d_qkv, d_g, d_beta, o1, S, H, NG, scale, qkv_stride, (long)H*S, B);
        };
        auto l_t2 = [&](float* st, float* o1) {
            k_dn_t2<<<dim3(H, S/32), 32>>>(st, d_qkv, d_g, d_beta, o1, S, H, NG, scale, qkv_stride, (long)H*S, B);
        };
        cmp_run("dn t  (48x4x32)", l_t, &ms1);
        printf("       speedup vs v0: %.1fx\n", ms0/ms1);
        cmp_run("dn t2 (fused) ", l_t2, &ms2);
        printf("       speedup vs v0: %.1fx\n", ms0/ms2);
    }
    return 0;
}
