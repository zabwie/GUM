// Oversubscription test: one cudaMallocManaged buffer larger than VRAM.
// GPU initializes and validates every element. Host monitors RAM/swap between launches
// and aborts if limits are crossed. A cudaMalloc reserve keeps VRAM headroom for the desktop.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static void check(cudaError_t s, const char* what) {
    if (s != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(s)); exit(1); }
}

static long meminfo_mib(const char* key) {
    FILE* f = fopen("/proc/meminfo", "r");
    char line[256]; long v = -1; size_t n = strlen(key);
    while (f && fgets(line, sizeof line, f)) {
        if (!strncmp(line, key, n)) { v = atol(line + n) / 1024; break; }
    }
    if (f) fclose(f);
    return v;
}

static long g_min_avail = 1L << 40, g_swap0 = 0;
static long g_floor_ram_mib = 8192;
static void safety(const char* where) {
    long avail = meminfo_mib("MemAvailable:");
    long swap_used = meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:");
    g_min_avail = std::min(g_min_avail, avail);
    if (avail < g_floor_ram_mib || swap_used > g_swap0 + 12288) {
        fprintf(stderr, "ABORT at %s: MemAvailable=%ld MiB swap_used=%ld MiB (baseline %ld)\n",
                where, avail, swap_used, g_swap0);
        exit(4);
    }
}

__host__ __device__ static inline uint32_t value_at(uint64_t i) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull;
    return (uint32_t)(x >> 32) ^ (uint32_t)i;
}

__global__ void init_k(uint32_t* p, uint64_t base, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x)
        p[base + i] = value_at(base + i);
}

__global__ void validate_k(const uint32_t* p, uint64_t base, uint64_t n, unsigned long long* err) {
    unsigned long long local = 0;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x)
        if (p[base + i] != value_at(base + i)) ++local;
    if (local) atomicAdd(err, local);
}

int main(int argc, char** argv) {
    long total_mib = 13312, reserve_mib = 2048, repeats = 2, ram_floor = 8192;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--mib")) total_mib = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--reserve-mib")) reserve_mib = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--repeats")) repeats = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--ram-floor-mib")) ram_floor = atol(argv[i + 1]);
    }
    g_floor_ram_mib = ram_floor;
    if (total_mib > 14336 || total_mib % 32) { fprintf(stderr, "bad --mib\n"); return 2; }
    g_swap0 = meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:");
    safety("start");

    check(cudaSetDevice(0), "set device");
    size_t freeb, totalb;
    check(cudaMemGetInfo(&freeb, &totalb), "meminfo");
    printf("start,vram_total_MiB=%zu,vram_free_MiB=%zu,ram_avail_MiB=%ld,swap_used_MiB=%ld,alloc_MiB=%ld\n",
           totalb >> 20, freeb >> 20, meminfo_mib("MemAvailable:"), g_swap0, total_mib);

    void* reserve = nullptr;
    if (reserve_mib > 0) check(cudaMalloc(&reserve, (size_t)reserve_mib << 20), "reserve");

    const size_t bytes = (size_t)total_mib << 20;
    const uint64_t elems = bytes / 4;
    uint32_t* p = nullptr;
    check(cudaMallocManaged((void**)&p, bytes), "cudaMallocManaged");
    unsigned long long* err; check(cudaMallocManaged((void**)&err, 8), "err");
    printf("allocated,managed_MiB=%ld,exceeds_vram=%d\n", total_mib, bytes > totalb);

    const uint64_t tile_elems = (2u << 20) / 4, cluster_tiles = 16;
    const size_t tiles = bytes / (2u << 20);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto run_tiles = [&](const std::vector<size_t>& order, bool init) {
        cudaEventRecord(e0);
        int n = 0;
        for (size_t t : order) {
            if (init) init_k<<<128, 256>>>(p, t * tile_elems, tile_elems);
            else validate_k<<<128, 256>>>(p, t * tile_elems, tile_elems, err);
            if ((++n & 63) == 0) { check(cudaDeviceSynchronize(), "sync"); safety("pass"); }
        }
        cudaEventRecord(e1);
        check(cudaEventSynchronize(e1), "pass sync");
        check(cudaGetLastError(), "kernel");
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        return (double)ms;
    };

    std::vector<size_t> seq(tiles); std::iota(seq.begin(), seq.end(), 0);
    std::mt19937_64 rng(1234);
    std::vector<size_t> rnd = seq; std::shuffle(rnd.begin(), rnd.end(), rng);
    std::vector<size_t> clus; {
        size_t nc = tiles / cluster_tiles; std::vector<size_t> co(nc);
        std::iota(co.begin(), co.end(), 0); std::shuffle(co.begin(), co.end(), rng);
        for (size_t c : co) for (size_t k = 0; k < cluster_tiles; ++k) clus.push_back(c * cluster_tiles + k);
        for (size_t t = nc * cluster_tiles; t < tiles; ++t) clus.push_back(t);
    }

    double ms = run_tiles(seq, true);
    printf("init_first_touch,ms=%.1f,GBps=%.2f,min_ram_avail_MiB=%ld\n", ms, bytes / ms / 1e6, g_min_avail);
    safety("after init");

    unsigned long long total_err = 0;
    struct P { const char* name; const std::vector<size_t>* o; } passes[] =
        {{"sequential", &seq}, {"random_tile", &rnd}, {"cluster32", &clus}};
    for (auto& ps : passes) {
        for (long r = 0; r < repeats; ++r) {
            *err = 0;
            ms = run_tiles(*ps.o, false);
            total_err += *err;
            size_t f2, t2; cudaMemGetInfo(&f2, &t2);
            printf("pass,pattern=%s,repeat=%ld,ms=%.1f,GBps=%.2f,errors=%llu,vram_free_MiB=%zu,ram_avail_MiB=%ld,swap_used_MiB=%ld\n",
                   ps.name, r, ms, bytes / ms / 1e6, *err, f2 >> 20, meminfo_mib("MemAvailable:"),
                   meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:"));
            safety(ps.name);
        }
    }
    printf("summary,total_errors=%llu,min_ram_avail_MiB=%ld,final_swap_used_MiB=%ld\n", total_err,
           g_min_avail, meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:"));
    cudaFree(p); cudaFree(err); if (reserve) cudaFree(reserve);
    return total_err ? 1 : 0;
}
