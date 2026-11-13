// GEMV-only + PCIe probe (E033). Splits the streamed-decode per-token cost into its parts:
//  (1) pinned H2D bandwidth vs chunk size / stream count,
//  (2) GEMV kernel time over a VRAM-resident buffer with no copies (current fp32 kernel vs dp4a kernel),
//  (3) copy+GEMV overlapped, (4) pageable->pinned memcpy bandwidth with N threads.
// Weight data is arbitrary bytes (timing only); kernel agreement is checked between kernels.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cstdint>

static void chk(cudaError_t s, const char* w) { if (s != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(s)); exit(1); } }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__global__ void gemv_fp(const uint8_t* w, long K, long rows, const float* x, float* y) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32; const size_t rowbytes = (size_t)nb * 18;
    const uint8_t* r = w + (size_t)row * rowbytes;
    float acc = 0.f;
    for (long b = lane; b < nb; b += 32) {
        const uint16_t* blk = (const uint16_t*)(r + (size_t)b * 18);
        const float d = __half2float(__ushort_as_half(blk[0]));
        const float* xb = x + b * 32; float s = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint16_t v = blk[1 + j]; const uint8_t q0 = v & 0xff, q1 = (v >> 8) & 0xff;
            s += (float)((int)(q0 & 15) - 8) * xb[2 * j] + (float)((int)(q0 >> 4) - 8) * xb[2 * j + 16];
            s += (float)((int)(q1 & 15) - 8) * xb[2 * j + 1] + (float)((int)(q1 >> 4) - 8) * xb[2 * j + 17];
        }
        acc += d * s;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

// x pre-quantized to int8 per 32-block (xq packed so that bytes 0..15 = x[0..15] low-nibble partners,
// i.e. stored in natural order); xs[b] = scale, xsum[b] = sum of int8 values.
__global__ void quant_x(const float* x, long nb, int8_t* xq, float* xs, int* xsum) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nb) return;
    float amax = 0.f; for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
    float d = amax / 127.f; float id = d > 0.f ? 1.f / d : 0.f; int s = 0;
    for (int i = 0; i < 32; ++i) { int q = __float2int_rn(x[b * 32 + i] * id); xq[b * 32 + i] = (int8_t)q; s += q; }
    xs[b] = d; xsum[b] = s;
}
template <int ROWS_PER_WARP>
__global__ void gemv_dp4a(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y) {
    const int lane = threadIdx.x & 31;
    const long row0 = ((long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * ROWS_PER_WARP;
    if (row0 >= rows) return;
    const long nb = K / 32; const size_t rowbytes = (size_t)nb * 18;
    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; ++i) acc[i] = 0.f;
    for (long b = lane; b < nb; b += 32) {
        const int4* xp = (const int4*)(xq + b * 32);
        const int4 xa = xp[0], xb = xp[1];            // x[0..15], x[16..31]
        const float sx = xs[b]; const int sum = xsum[b];
#pragma unroll
        for (int i = 0; i < ROWS_PER_WARP; ++i) {
            const long row = row0 + i; if (row >= rows) break;
            const uint16_t* blk = (const uint16_t*)(w + (size_t)row * rowbytes + (size_t)b * 18);
            const float d = __half2float(__ushort_as_half(blk[0]));
            uint32_t q[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
            int s = 0;
            // low nibbles pair with x[0..15] (bytes 4j..4j+3), high nibbles with x[16..31]
            s = __dp4a((int)(q[0] & 0x0F0F0F0Fu), xa.x, s); s = __dp4a((int)(q[1] & 0x0F0F0F0Fu), xa.y, s);
            s = __dp4a((int)(q[2] & 0x0F0F0F0Fu), xa.z, s); s = __dp4a((int)(q[3] & 0x0F0F0F0Fu), xa.w, s);
            s = __dp4a((int)((q[0] >> 4) & 0x0F0F0F0Fu), xb.x, s); s = __dp4a((int)((q[1] >> 4) & 0x0F0F0F0Fu), xb.y, s);
            s = __dp4a((int)((q[2] >> 4) & 0x0F0F0F0Fu), xb.z, s); s = __dp4a((int)((q[3] >> 4) & 0x0F0F0F0Fu), xb.w, s);
            acc[i] += d * sx * (float)(s - 8 * sum);
        }
    }
#pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; ++i) {
        float a = acc[i];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0 && row0 + i < rows) y[row0 + i] = a;
    }
}

int main(int argc, char** argv) {
    const long K = 5120, N = 17408 * 4;                  // ~4 FFN-sized matrices worth of rows per "item set"
    const size_t rowbytes = (size_t)(K / 32) * 18;
    const size_t item_rows = (32u << 20) / rowbytes, item_bytes = item_rows * rowbytes;
    const double cold_bytes = 5822449920.0;               // per-token cold set from E031/E032
    const int items = (int)(cold_bytes / item_bytes);
    printf("config,K=%ld,item_MiB=%.2f,item_rows=%zu,items_per_token=%d\n", K, item_bytes / 1048576.0, item_rows, items);
    (void)N;

    uint8_t* arena; chk(cudaMalloc(&arena, 4 * item_bytes), "arena"); chk(cudaMemset(arena, 0x5A, 4 * item_bytes), "memset");
    // diversify bytes so dp4a/fp results are non-trivial
    { std::vector<uint8_t> h(item_bytes); for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t)((i * 2654435761u) >> 13);
      // keep scale halves finite & small: force byte1 of each block header to 0x2C-0x30 range
      for (size_t b = 0; b < h.size() / 18; ++b) { h[b * 18] = (uint8_t)(b * 7); h[b * 18 + 1] = 0x2C + (b & 3); }
      for (int s = 0; s < 4; ++s) chk(cudaMemcpy(arena + s * item_bytes, h.data(), item_bytes, cudaMemcpyHostToDevice), "fill"); }
    float *dx, *dy, *dy2; chk(cudaMalloc(&dx, K * 4), "x"); chk(cudaMalloc(&dy, item_rows * 4 * 4), "y"); chk(cudaMalloc(&dy2, item_rows * 4 * 4), "y2");
    { std::vector<float> hx(K); for (long i = 0; i < K; ++i) hx[i] = (float)(((i * 37) % 101) - 50) / 64.f; chk(cudaMemcpy(dx, hx.data(), K * 4, cudaMemcpyHostToDevice), "xc"); }
    int8_t* xq; float* xs; int* xsum; chk(cudaMalloc(&xq, K), "xq"); chk(cudaMalloc(&xs, K / 32 * 4), "xs"); chk(cudaMalloc(&xsum, K / 32 * 4), "xsum");
    quant_x<<<1, 160>>>(dx, K / 32, xq, xs, xsum);
    chk(cudaDeviceSynchronize(), "quant");

    auto time_kernel = [&](const char* name, auto launch) {
        for (int w = 0; w < 3; ++w) for (int i = 0; i < items; ++i) launch(i % 4);
        chk(cudaDeviceSynchronize(), name);
        double best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            double t0 = now(); for (int i = 0; i < items; ++i) launch(i % 4); chk(cudaDeviceSynchronize(), name);
            best = std::min(best, now() - t0);
        }
        printf("gemv,%s,per_token_s=%.4f,GBps=%.1f\n", name, best, items * item_bytes / best / 1e9);
        return best;
    };
    const int TPB = 128;
    time_kernel("fp32_current", [&](int s) { gemv_fp<<<(unsigned)((item_rows + 3) / 4), TPB>>>(arena + s * item_bytes, K, item_rows, dx, dy); });
    time_kernel("dp4a_1row", [&](int s) { gemv_dp4a<1><<<(unsigned)((item_rows + 3) / 4), TPB>>>(arena + s * item_bytes, K, item_rows, xq, xs, xsum, dy2); });
    time_kernel("dp4a_2row", [&](int s) { gemv_dp4a<2><<<(unsigned)((item_rows + 7) / 8), TPB>>>(arena + s * item_bytes, K, item_rows, xq, xs, xsum, dy2); });
    time_kernel("dp4a_4row", [&](int s) { gemv_dp4a<4><<<(unsigned)((item_rows + 15) / 16), TPB>>>(arena + s * item_bytes, K, item_rows, xq, xs, xsum, dy2); });
    // agreement check fp32 vs dp4a_1row on slot 0
    {
        gemv_fp<<<(unsigned)((item_rows + 3) / 4), TPB>>>(arena, K, item_rows, dx, dy);
        gemv_dp4a<1><<<(unsigned)((item_rows + 3) / 4), TPB>>>(arena, K, item_rows, xq, xs, xsum, dy2);
        gemv_dp4a<4><<<(unsigned)((item_rows + 15) / 16), TPB>>>(arena, K, item_rows, xq, xs, xsum, dy2 + item_rows);
        chk(cudaDeviceSynchronize(), "agree");
        std::vector<float> a(item_rows), b(item_rows), c(item_rows);
        cudaMemcpy(a.data(), dy, item_rows * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), dy2, item_rows * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(c.data(), dy2 + item_rows, item_rows * 4, cudaMemcpyDeviceToHost);
        double ma = 0, md = 0, m4 = 0; for (size_t i = 0; i < item_rows; ++i) { ma = fmax(ma, fabs(a[i])); md = fmax(md, fabs(a[i] - b[i])); m4 = fmax(m4, fabs(b[i] - c[i])); }
        printf("agree,max_abs_fp32=%.4f,max_abs_diff_dp4a_vs_fp32=%.5f,rel=%.4g,dp4a1_vs_dp4a4_diff=%.3g\n", ma, md, md / (ma + 1e-9), m4);
    }

    // PCIe / H2D
    const size_t pin_total = 256u << 20;
    for (size_t chunk_mib : {4, 8, 16, 32, 64, 128}) {
        for (int nstreams : {1, 2, 4}) {
            size_t chunk = chunk_mib << 20; if (chunk > pin_total) continue;
            uint8_t* pin; chk(cudaHostAlloc(&pin, pin_total, cudaHostAllocPortable), "pin");
            memset(pin, 1, pin_total);
            uint8_t* dst; chk(cudaMalloc(&dst, 4 * chunk > (256u << 20) ? 4 * chunk : (256u << 20)), "dst");
            std::vector<cudaStream_t> ss(nstreams); for (auto& s : ss) chk(cudaStreamCreate(&s), "s");
            const size_t total = 4ull << 30; double best = 1e9;
            for (int rep = 0; rep < 3; ++rep) {
                double t0 = now(); size_t n = total / chunk;
                for (size_t i = 0; i < n; ++i) chk(cudaMemcpyAsync(dst + (i % 4) * chunk, pin + (i % (pin_total / chunk)) * chunk, chunk, cudaMemcpyHostToDevice, ss[i % nstreams]), "h2d");
                chk(cudaDeviceSynchronize(), "h2d sync"); best = std::min(best, now() - t0);
            }
            printf("h2d,chunk_MiB=%zu,streams=%d,GBps=%.2f\n", chunk_mib, nstreams, total / best / 1e9);
            for (auto s : ss) cudaStreamDestroy(s); cudaFree(dst); cudaFreeHost(pin);
        }
    }
    // pageable -> pinned memcpy bandwidth with T threads (the "staged" path's CPU half)
    {
        const size_t sz = 2ull << 30; uint8_t* src = (uint8_t*)aligned_alloc(4096, sz); memset(src, 3, sz);
        uint8_t* pin; chk(cudaHostAlloc(&pin, 8 * (32u << 20), cudaHostAllocPortable), "pin2");
        for (int T : {1, 2, 4, 6}) {
            double t0 = now(); std::vector<std::thread> th;
            for (int t = 0; t < T; ++t) th.emplace_back([&, t]() { for (size_t i = t; i < sz / (32u << 20); i += T) memcpy(pin + (i % 8) * (32u << 20), src + i * (32u << 20), 32u << 20); });
            for (auto& x : th) x.join();
            double s = now() - t0; printf("memcpy_pageable_to_pinned,threads=%d,GBps=%.2f\n", T, sz / s / 1e9);
        }
    }
    // overlapped copy + GEMV per token (best kernel), 4-deep ring
    {
        uint8_t* pin; chk(cudaHostAlloc(&pin, 8 * item_bytes, cudaHostAllocPortable), "pin3"); memset(pin, 2, 8 * item_bytes);
        cudaStream_t sc, sk; cudaStreamCreate(&sc); cudaStreamCreate(&sk);
        for (int variant = 0; variant < 2; ++variant) {
            cudaEvent_t ec[4], ek[4]; for (int i = 0; i < 4; ++i) { cudaEventCreate(&ec[i]); cudaEventCreate(&ek[i]); cudaEventRecord(ek[i], sk); }
            cudaDeviceSynchronize(); double best = 1e9;
            for (int rep = 0; rep < 4; ++rep) {
                double t0 = now();
                for (int i = 0; i < items; ++i) {
                    int s = i % 4; cudaStreamWaitEvent(sc, ek[s], 0);
                    cudaMemcpyAsync(arena + s * item_bytes, pin + (i % 8) * item_bytes, item_bytes, cudaMemcpyHostToDevice, sc);
                    cudaEventRecord(ec[s], sc); cudaStreamWaitEvent(sk, ec[s], 0);
                    if (variant == 0) gemv_fp<<<(unsigned)((item_rows + 3) / 4), TPB, 0, sk>>>(arena + s * item_bytes, K, item_rows, dx, dy);
                    else gemv_dp4a<2><<<(unsigned)((item_rows + 7) / 8), TPB, 0, sk>>>(arena + s * item_bytes, K, item_rows, xq, xs, xsum, dy2);
                    cudaEventRecord(ek[s], sk);
                }
                chk(cudaDeviceSynchronize(), "overlap"); best = std::min(best, now() - t0);
            }
            printf("overlap,%s,per_token_s=%.4f,GBps=%.2f\n", variant == 0 ? "fp32_current" : "dp4a_2row", best, items * item_bytes / best / 1e9);
        }
    }
    return 0;
}
