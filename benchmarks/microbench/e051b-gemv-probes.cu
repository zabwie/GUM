// E051b probes: why is NB=16 GEMV stuck at ~50 GB/s? Ladder:
//   v0      : production kernel (baseline)
//   v0n1    : NB=1 calibration (DRAM/weight-stream ceiling, expect ~273 GB/s)
//   pr1     : R=1, xs/xsum smem tile (kills the 32x-overfetch scalar stream) at full occupancy
//   pr1f    : pr1 but FAKE xq (no xq loads) -> isolates xq-global cost
//   pr2/pr4 : R=2/4 row amortization of xq loads, __launch_bounds__ variants for occupancy
// All real-data variants bit-exact vs v0.
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
// NB=1 version for calibration
__global__ void k_v0n1(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32; const size_t rowbytes = (size_t)nb * 18;
    const uint8_t* r = w + (size_t)row * rowbytes;
    float acc = 0.f;
    for (long b = lane; b < nb; b += 32) {
        const uint16_t* blk = (const uint16_t*)(r + (size_t)b * 18);
        const float d = __half2float(__ushort_as_half(blk[0]));
        uint32_t q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
        const uint32_t lo0 = q[0] & 0x0F0F0F0Fu, lo1 = q[1] & 0x0F0F0F0Fu, lo2 = q[2] & 0x0F0F0F0Fu, lo3 = q[3] & 0x0F0F0F0Fu;
        const uint32_t hi0 = (q[0] >> 4) & 0x0F0F0F0Fu, hi1 = (q[1] >> 4) & 0x0F0F0F0Fu, hi2 = (q[2] >> 4) & 0x0F0F0F0Fu, hi3 = (q[3] >> 4) & 0x0F0F0F0Fu;
        const int4* xp = (const int4*)(xq + b * 32);
        const int4 xa = xp[0], xb = xp[1];
        int s = 0;
        s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
        s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
        acc += d * xs[b] * (float)(s - 8 * xsum[b]);
    }
    float a = acc;
    for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    if (lane == 0) y[row] = a;
}

// probe kernel: xs/xsum smem tile (staged once per block, full K), R rows per warp,
// optional fake-xq. 128 threads = 4 warps.
template <int R, bool REAL_XQ, int MINB>
__global__ void __launch_bounds__(128, MINB)
k_pr(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride) {
    extern __shared__ int2 smem[];      // [16][nb]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const long nb = K / 32;
    for (int i = tid; i < 16 * nb; i += 128)
        smem[i] = make_int2(__float_as_int(xs[i / nb * nb + i % nb]), xsum[i / nb * nb + i % nb]);
    __syncthreads();
    const long row0 = (long)blockIdx.x * (4 * R) + (tid >> 5) * R;
    const size_t rowbytes = (size_t)nb * 18;
    float acc[R][16];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int v = 0; v < 16; ++v) acc[r][v] = 0.f;
    const int4 xfake_a = make_int4(1, 2, 3, 4), xfake_b = make_int4(5, 6, 7, 8);
    for (long b = lane; b < nb; b += 32) {
        uint32_t lo[R][4], hi[R][4]; float d[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= rows) continue;
            const uint16_t* blk = (const uint16_t*)(w + (size_t)(row0 + r) * rowbytes + (size_t)b * 18);
            d[r] = __half2float(__ushort_as_half(blk[0]));
            uint32_t q[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
#pragma unroll
            for (int j = 0; j < 4; ++j) { lo[r][j] = q[j] & 0x0F0F0F0Fu; hi[r][j] = (q[j] >> 4) & 0x0F0F0F0Fu; }
        }
#pragma unroll
        for (int v = 0; v < 16; ++v) {
            int4 xa, xb;
            if (REAL_XQ) { const int4* xp = (const int4*)(xq + v * (K / 32 * 32) + b * 32); xa = xp[0]; xb = xp[1]; }
            else { xa = xfake_a; xb = xfake_b; }
            const int2 p = smem[v * nb + b];
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
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= rows) continue;
#pragma unroll
        for (int v = 0; v < 16; ++v) {
            float a = acc[r][v];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
            if (lane == 0) y[v * ystride + row0 + r] = a;
        }
    }
}

int main() {
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
    const size_t smem = (size_t)16 * nb * 8;

    CHECK(cudaFuncSetAttribute(k_pr<1,true,5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));
    CHECK(cudaFuncSetAttribute(k_pr<1,false,5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));
    CHECK(cudaFuncSetAttribute(k_pr<2,true,4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));
    CHECK(cudaFuncSetAttribute(k_pr<4,true,3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));
    CHECK(cudaFuncSetAttribute(k_pr<4,true,2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));
    CHECK(cudaFuncSetAttribute(k_pr<8,true,1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024));

    // v0 reference output + timing
    const unsigned g0 = (unsigned)((rows + 3) / 4);
    k_v0<<<g0,128>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y0, rows);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_y0, d_y0, 16*rows*4, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) k_v0<<<g0,128>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y0, rows);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
    printf("v0 NB=16        : %.3f ms (%.0f GB/s)\n", ms/iters, raw_bytes/1e9/(ms/iters/1e3));
    // NB=1 calibration: only 1 xq vector — reuse kernel with K rows... separate buffer
    {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) k_v0n1<<<g0,128>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        printf("v0 NB=1         : %.3f ms (%.0f GB/s)\n", ms/iters, raw_bytes/1e9/(ms/iters/1e3));
    }

    auto run = [&](const char* name, auto launch) {
        launch();
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_y1, d_y1, 16*rows*4, cudaMemcpyDeviceToHost));
        bool ok = memcmp(h_y0, h_y1, 16*rows*4) == 0;
        if (!ok) for (long i=0;i<16*rows;++i) if (h_y0[i]!=h_y1[i]) { printf("  diff at %ld: %a vs %a\n", i, h_y0[i], h_y1[i]); break; }
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) launch();
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        printf("%s: %.3f ms (%.0f GB/s) bitexact=%s\n", name, ms/iters, raw_bytes/1e9/(ms/iters/1e3), ok ? "YES" : "n/a(fake)");
    };
    const unsigned g1 = (unsigned)((rows + 3) / 4);        // R=1: 4 rows/block
    const unsigned g2 = (unsigned)((rows + 7) / 8);        // R=2
    const unsigned g4 = (unsigned)((rows + 15) / 16);      // R=4
    const unsigned g8 = (unsigned)((rows + 31) / 32);      // R=8
    run("pr1  xs-smem R1", [&] { k_pr<1,true,5><<<g1,128,smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows); });
    run("pr1f fake-xq R1", [&] { k_pr<1,false,5><<<g1,128,smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows); });
    run("pr2  xs-smem R2", [&] { k_pr<2,true,4><<<g2,128,smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows); });
    run("pr4  xs-smem R4", [&] { k_pr<4,true,3><<<g4,128,smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows); });
    run("pr4b R4 occ2    ", [&] { k_pr<4,true,2><<<g4,128,smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows); });
    run("pr8  xs-smem R8", [&] { k_pr<8,true,1><<<g8,128,smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y1, rows); });
    return 0;
}
