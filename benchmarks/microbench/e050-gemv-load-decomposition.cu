// Isolate: (0) current, (1) no-xq-loads (reg const), (2) no-loads-at-all (pure compute)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

__global__ void k_v0(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32;
    const uint8_t* r = w + (size_t)row * nb * 18;
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
            const int4* xp = (const int4*)(xq + v * K + b * 32);
            const int4 xa = xp[0], xb = xp[1];
            int s = 0;
            s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
            s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
            acc[v] += d * xs[v * nb + b] * (float)(s - 8 * xsum[v * nb + b]);
        }
    }
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[v * ystride + row] = a;
    }
}
// v1: xq from registers (fake but same instruction count minus loads); xs/xsum still loaded
__global__ void k_v1(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32;
    const uint8_t* r = w + (size_t)row * nb * 18;
    float acc[16];
#pragma unroll
    for (int v = 0; v < 16; ++v) acc[v] = 0.f;
    const int4 xa = make_int4(1, 2, 3, 4), xb = make_int4(5, 6, 7, 8);
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
            int s = 0;
            s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
            s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
            acc[v] += d * xs[v * nb + b] * (float)(s - 8 * xsum[v * nb + b]);
        }
    }
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[v * ystride + row] = a;
    }
}

// v3: xs/xsum in smem (loaded once per block), xq in regs, weights as-is -> isolates weight-load cost
__global__ void k_v3(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride) {
    extern __shared__ float sxs[];           // [16][nb] floats then [16][nb] ints
    int* sxsum = (int*)(sxs + 16 * (K / 32));
    const int lane = threadIdx.x & 31;
    const long nb = K / 32;
    for (int i = threadIdx.x; i < 16 * nb; i += blockDim.x) { sxs[i] = xs[i]; sxsum[i] = xsum[i]; }
    __syncthreads();
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const uint8_t* r = w + (size_t)row * nb * 18;
    float acc[16];
#pragma unroll
    for (int v = 0; v < 16; ++v) acc[v] = 0.f;
    const int4 xa = make_int4(1, 2, 3, 4), xb = make_int4(5, 6, 7, 8);
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
            int s = 0;
            s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
            s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
            acc[v] += d * sxs[v * nb + b] * (float)(s - 8 * sxsum[v * nb + b]);
        }
    }
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[v * ystride + row] = a;
    }
}
// v2: no loads at all, same ALU
__global__ void k_v2(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32;
    float acc[16];
#pragma unroll
    for (int v = 0; v < 16; ++v) acc[v] = 0.f;
    const int4 xa = make_int4(1, 2, 3, 4), xb = make_int4(5, 6, 7, 8);
    uint32_t lo0 = 0x32103210u, lo1 = 0x54105410u, lo2 = 0x65206520u, lo3 = 0x76307630u;
    uint32_t hi0 = 0x12301230u, hi1 = 0x23402340u, hi2 = 0x34503450u, hi3 = 0x45604560u;
    const float d = 0.5f;
    for (long b = lane; b < nb; b += 32) {
#pragma unroll
        for (int v = 0; v < 16; ++v) {
            int s = 0;
            s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
            s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
            acc[v] += d * (float)v * 0.001f * (float)(s - 8 * (int)b);
        }
    }
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[v * ystride + row] = a;
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
            *(uint16_t*)(h_raw + ((size_t)r * nb + b) * 18) = __half_as_ushort(__float2half(0.001f * (float)((r + b) % 97 + 1)));
    int8_t* h_x = (int8_t*)malloc(16 * K);
    float* h_xs = (float*)malloc(16 * nb * 4); int* h_xsum = (int*)malloc(16 * nb * 4);
    for (long i = 0; i < 16 * K; ++i) h_x[i] = (rand() % 31) - 15;
    for (long v = 0; v < 16; ++v) for (long b = 0; b < nb; ++b) { h_xs[v*nb+b] = 0.01f; h_xsum[v*nb+b] = 0; }
    uint8_t* d_raw; int8_t* d_x; float *d_xs, *d_y; int* d_xsum;
    CHECK(cudaMalloc(&d_raw, raw_bytes));
    CHECK(cudaMalloc(&d_x, 16*K)); CHECK(cudaMalloc(&d_xs, 16*nb*4)); CHECK(cudaMalloc(&d_xsum, 16*nb*4));
    CHECK(cudaMalloc(&d_y, 16*rows*4));
    CHECK(cudaMemcpy(d_raw, h_raw, raw_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h_x, 16*K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_xs, h_xs, 16*nb*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_xsum, h_xsum, 16*nb*4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    const unsigned g = (unsigned)((rows + 3) / 4);
    const int iters = 30;
    struct { const char* name; void(*k)(const uint8_t*, long, long, const int8_t*, const float*, const int*, float*, long); } vs[] = {
        {"v0 current      ", k_v0}, {"v1 no-xq-load    ", k_v1}, {"v2 no-loads-atall", k_v2}};
    for (auto& t : vs) {
        t.k<<<g,128>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y, rows);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) t.k<<<g,128>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y, rows);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        printf("%s: %.3f ms  (%.1f GB/s equiv)\n", t.name, ms/iters, raw_bytes/1e9/(ms/iters/1e3));
    }
    {
        const int smem = 16 * (K/32) * 8;
        const unsigned g = (unsigned)((rows + 3) / 4);
        k_v3<<<g,128,smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y, rows);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) k_v3<<<g,128,smem>>>(d_raw, K, rows, d_x, d_xs, d_xsum, d_y, rows);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        printf("v3 smem-xs/xsum  : %.3f ms  (%.1f GB/s equiv)\n", ms/iters, raw_bytes/1e9/(ms/iters/1e3));
    }
    return 0;
}
