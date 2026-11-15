// Compare E195 and E196 on the same packed history, queries and split boundaries.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t error, const char* where) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
    std::exit(1);
}

#ifdef E195_HEADER
#include E195_HEADER
#else
#include "../../benchmarks/tiered_kv.cuh"
#ifndef HEAD_TILE
#define HEAD_TILE 6
#endif
#endif

__device__ __host__ uint32_t hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

__global__ void fill_packed(TierRow* rows, unsigned count, unsigned salt) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    TierRow& packed = rows[row];
    const unsigned seed = hash(row ^ salt);
    for (int i = 0; i < kTierHead / 2; ++i)
        packed.values[i] = hash(seed + i);
    packed.key_scale = 0.005f + (hash(seed) & 255) * 0.0001f;
    for (int i = 0; i < kTierHead / 32; ++i)
        packed.value_range[i] = make_float2(0.02f + (hash(seed + i) & 15) * 0.002f, -0.225f);
#ifdef E195_HEADER
    for (int i = 0; i < kTierHead / 2; ++i)
        packed.keys[i] = (hash(seed + 2*i + 256) & 15) | ((hash(seed + 2*i + 257) & 15) << 4);
#else
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t word = 0;
        for (int part = 0; part < kTierHead / 32; ++part)
            word |= (hash(seed + part * 32 + lane + 256) & 15) << (4 * part);
        packed.keys[lane] = word;
    }
#endif
}

__global__ void fill_recent(float2* data, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] = make_float2((int(hash(i) % 256) - 127) * 0.002f,
                                        (int(hash(i ^ 1234) % 256) - 127) * 0.002f);
}

__global__ void fill_query(float* data, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] = (int(hash(i ^ 711) % 256) - 127) * 0.01f;
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    constexpr int ctx = 262144, q_heads = 24, kv_heads = 4, stride = 512;
    constexpr int old_groups = ctx / kTierOldGroup, batch = 32;
    const int rows = std::atoi(argv[1]);
    const int protect_percent = argc > 3 ? std::atoi(argv[3]) : 32;
    const int pos = ctx - rows;
    const int splits = ctx / kTierSplit;
    if (rows < 1 || rows > 256 || protect_percent < 0 || protect_percent > 100) return 2;
    std::vector<uint8_t> protect(ctx);
    std::vector<int> map(ctx, -1), depth(rows), path((size_t)rows * rows);
    int kept = 0;
    for (int start = 0; start < ctx; start += kTierOldGroup) {
        if (hash(start) % 100 >= protect_percent) continue;
        protect[start] = 1;
        for (int i = 0; i < kTierOldGroup; ++i) map[start + i] = kept++;
        for (int i = 1; i < kTierOldGroup; i += kTierMidGroup) protect[start + i] = 1;
    }
    // Vary the current branch depths and storage paths as in speculative verification.
    for (int row = 0; row < rows; ++row) {
        depth[row] = row / 2;
        for (int t = 0; t <= depth[row]; ++t) path[row * rows + t] = rows - t - 1;
    }
    TierCache cache;
    cache.pooling = true;
    cache.span = ctx;
    cache.capacity = kept ? kept : 1;
    cache.ring = 1040;
    cache.w1_slots = kTierSplit;
    cache.middle_groups = 1156;
    cache.allocate(kv_heads);
    const auto fill = [](TierRow* ptr, unsigned count, unsigned salt) {
        fill_packed<<<(count + 127) / 128, 128>>>(ptr, count, salt);
    };
    fill(cache.rows, kv_heads * cache.capacity, 100);
    fill(cache.old, kv_heads * old_groups, 200);
    fill(cache.middle, kv_heads * cache.middle_groups, 300);
    fill(cache.w1, kv_heads * cache.w1_slots, 400);
    const unsigned recent_cells = kv_heads * cache.ring * kTierHead;
    fill_recent<<<(recent_cells + 127) / 128, 128>>>(cache.recent, recent_cells);
    int *overlay, *dev_depth, *dev_path;
    uint8_t* dev_protect;
    check(cudaMalloc(&overlay, ctx * sizeof(int)), "overlay");
    check(cudaMalloc(&dev_protect, ctx), "protection mask");
    check(cudaMalloc(&dev_depth, depth.size() * sizeof(int)), "depth");
    check(cudaMalloc(&dev_path, path.size() * sizeof(int)), "paths");
    check(cudaMemcpy(overlay, map.data(), ctx * sizeof(int), cudaMemcpyHostToDevice), "overlay copy");
    check(cudaMemcpy(dev_protect, protect.data(), ctx, cudaMemcpyHostToDevice), "protection copy");
    check(cudaMemcpy(dev_depth, depth.data(), depth.size() * sizeof(int), cudaMemcpyHostToDevice), "depth copy");
    check(cudaMemcpy(dev_path, path.data(), path.size() * sizeof(int), cudaMemcpyHostToDevice), "path copy");
    cache.overlay = overlay;

    const size_t query_cells = rows * q_heads * stride;
    const size_t cells = rows * q_heads * kTierHead;
    float *query, *rotated, *output, *partial;
    check(cudaMalloc(&query, query_cells * sizeof(float)), "query");
    check(cudaMalloc(&rotated, cells * sizeof(float)), "rotated");
    check(cudaMalloc(&output, cells * sizeof(float)), "output");
    const size_t scratch = (size_t)batch * q_heads * splits * (kTierHead + 2) * sizeof(float);
    check(cudaMalloc(&partial, scratch), "partials");
    fill_query<<<(query_cells + 127) / 128, 128>>>(query, query_cells);
    k_tier_query<<<dim3(q_heads, rows), kTierHead>>>(rotated, query, q_heads, stride);
    auto launch = [&] {
        for (int first = 0; first < rows; first += batch) {
            const int nr = std::min(batch, rows - first);
#ifdef E195_HEADER
            k_tier_attention<<<dim3(q_heads, nr, splits), kTierHead>>>(
                output, partial, cache, query, rotated, dev_protect, pos, q_heads, kv_heads, stride,
                dev_depth, dev_path, rows, splits, first, 1.f / 16);
#else
            const int old_splits = (pos + 1 - kTierOld) / kTierSplit;
            k_tier_old_attention<HEAD_TILE><<<dim3(q_heads / HEAD_TILE, nr, old_splits), kTierHead>>>(
                partial, cache, rotated, dev_protect, q_heads, kv_heads, splits, first, 1.f / 16);
            k_tier_attention<<<dim3(q_heads, nr, splits - old_splits), kTierHead>>>(
                output, partial, cache, query, rotated, dev_protect, pos, q_heads, kv_heads, stride,
                dev_depth, dev_path, rows, splits, first, 1.f / 16, old_splits);
#endif
            k_tier_merge<<<q_heads * nr, kTierHead>>>(output + (size_t)first * q_heads * kTierHead, partial, splits);
        }
        check(cudaGetLastError(), "kernel launch");
    };
    launch();
    check(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t start, end;
    check(cudaEventCreate(&start), "start");
    check(cudaEventCreate(&end), "end");
    const int samples = rows > 32 ? 3 : 9;
    float best = INFINITY, total = 0;
    for (int i = 0; i < samples; ++i) {
        check(cudaEventRecord(start), "start timing");
        launch();
        check(cudaEventRecord(end), "end timing");
        check(cudaEventSynchronize(end), "sync timing");
        float ms;
        check(cudaEventElapsedTime(&ms, start, end), "get time");
        best = fminf(ms, best);
        total += ms;
    }

    std::vector<float> result(cells);
    check(cudaMemcpy(result.data(), output, cells * sizeof(float), cudaMemcpyDeviceToHost), "outputs");
    for (float v : result) if (!std::isfinite(v)) return 3;
    std::FILE* file = std::fopen(argv[2], "wb");
    if (!file || std::fwrite(result.data(), sizeof(float), cells, file) != cells || std::fclose(file)) return 4;
    std::printf("ctx=%d pos=%d verifier_rows=%d protected_rows=%d cache_bytes=%zu scratch_bytes=%zu "
                "samples=%d min_ms=%.3f mean_ms=%.3f\n",
                ctx, pos, rows, kept, cache.bytes(kv_heads), scratch, samples, best, total / samples);
}
