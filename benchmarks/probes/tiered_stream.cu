// Tiered pool streaming test: NVMe (file) -> bounded RAM cache -> pinned staging ring -> VRAM.
// Dataset is a read-only file larger than RAM+VRAM. Kernels only read VRAM. Every byte is
// validated on the GPU. No managed memory, no anonymous RAM beyond the explicit budget.
#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static constexpr size_t kChunk = 32u << 20;
static constexpr int kStage = 8;     // pinned staging buffers (256 MiB)
static constexpr int kSlots = 4;     // VRAM streaming slots (128 MiB)

static void check(cudaError_t s, const char* w) {
    if (s != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(s)); exit(1); }
}
static long meminfo_mib(const char* key) {
    FILE* f = fopen("/proc/meminfo", "r"); char l[256]; long v = -1; size_t n = strlen(key);
    while (f && fgets(l, sizeof l, f)) if (!strncmp(l, key, n)) { v = atol(l + n) / 1024; break; }
    if (f) fclose(f);
    return v;
}
static long g_swap0, g_floor = 8192, g_swap_tol = 1024, g_min_avail = 1L << 40;
static void safety(const char* where) {
    long a = meminfo_mib("MemAvailable:"), s = meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:");
    if (a < g_min_avail) g_min_avail = a;
    if (a < g_floor || s > g_swap0 + g_swap_tol) {
        fprintf(stderr, "ABORT at %s: MemAvailable=%ld MiB swap_used=%ld (baseline %ld)\n", where, a, s, g_swap0);
        _exit(4);
    }
}
static inline uint32_t value_at(uint64_t i) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull;
    return (uint32_t)(x >> 32) ^ (uint32_t)i;
}
__device__ static inline uint32_t dvalue_at(uint64_t i) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull;
    return (uint32_t)(x >> 32) ^ (uint32_t)i;
}
__global__ void validate_k(const uint32_t* p, uint64_t first_elem, uint64_t n, unsigned long long* err) {
    unsigned long long local = 0;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x)
        if (p[i] != dvalue_at(first_elem + i)) ++local;
    if (local) atomicAdd(err, local);
}

static void make_file(const std::string& path, size_t chunks) {
    struct stat_t {};
    int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        off_t sz = lseek(fd, 0, SEEK_END); close(fd);
        if ((size_t)sz == chunks * kChunk) return;
    }
    printf("generating %zu MiB file %s\n", chunks * (kChunk >> 20), path.c_str());
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) { perror("create"); exit(1); }
    void* buf; if (posix_memalign(&buf, 4096, kChunk)) exit(1);
    uint32_t* w = (uint32_t*)buf;
    for (size_t c = 0; c < chunks; ++c) {
        uint64_t base = (uint64_t)c * (kChunk / 4);
        for (size_t i = 0; i < kChunk / 4; ++i) w[i] = value_at(base + i);
        if (pwrite(fd, buf, kChunk, (off_t)(c * kChunk)) != (ssize_t)kChunk) { perror("pwrite"); exit(1); }
        if ((c & 15) == 0) safety("generate");
    }
    close(fd); free(buf);
}

int main(int argc, char** argv) {
    long gib = 24, hot_mib = 4096, cache_mib = 4096, passes = 2, readers = 4;
    std::string path = "tier_data/tier.bin";
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i]; const char* v = argv[i + 1];
        if (k == "--gib") gib = atol(v); else if (k == "--hot-mib") hot_mib = atol(v);
        else if (k == "--cache-mib") cache_mib = atol(v); else if (k == "--passes") passes = atol(v);
        else if (k == "--readers") readers = atol(v); else if (k == "--file") path = v;
        else if (k == "--ram-floor-mib") g_floor = atol(v); else if (k == "--swap-tol-mib") g_swap_tol = atol(v);
    }
    if (gib < 1 || gib > 64) { fprintf(stderr, "bad --gib (1..64)\n"); return 2; }
    const size_t chunks = (size_t)gib * 1024 / (kChunk >> 20);
    const size_t hot_chunks = (size_t)hot_mib / (kChunk >> 20);
    const size_t cache_chunks = (size_t)cache_mib / (kChunk >> 20);
    g_swap0 = meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:");
    safety("start");
    make_file(path, chunks);

    check(cudaSetDevice(0), "device");
    uint8_t* hot = nullptr; uint8_t* slots = nullptr; unsigned long long* err = nullptr;
    if (hot_chunks) check(cudaMalloc((void**)&hot, hot_chunks * kChunk), "hot arena");
    check(cudaMalloc((void**)&slots, (size_t)kSlots * kChunk), "slots");
    check(cudaMalloc((void**)&err, 8), "err");
    check(cudaMemset(err, 0, 8), "err0");
    uint8_t* stage[kStage];
    for (int i = 0; i < kStage; ++i) check(cudaHostAlloc((void**)&stage[i], kChunk, cudaHostAllocPortable), "stage");
    uint8_t* cache = nullptr;
    if (cache_chunks) { if (posix_memalign((void**)&cache, 4096, cache_chunks * kChunk)) { fprintf(stderr, "cache alloc\n"); return 1; } }
    std::vector<char> cached(chunks, 0);

    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); return 1; }
    cudaStream_t st; cudaStreamCreate(&st);
    cudaEvent_t ev[kStage]; for (auto& e : ev) cudaEventCreate(&e);

    printf("config,dataset_GiB=%ld,hot_MiB=%ld,ram_cache_MiB=%ld,staging_MiB=%d,readers=%ld,passes=%ld\n",
           gib, hot_mib, cache_mib, kStage * 32, readers, passes);

    for (long pass = 0; pass < passes; ++pass) {
        std::atomic<size_t> released{0};
        std::atomic<long> ready[kStage]; for (auto& r : ready) r = 0;
        std::atomic<size_t> from_nvme{0}, from_ram{0}, from_hot{0};
        std::atomic<bool> fail{false};
        auto t0 = std::chrono::steady_clock::now();

        // streamed chunks = those not hot-resident (hot chunks load into arena on pass 0)
        std::vector<std::thread> pool;
        for (long t = 0; t < readers; ++t) pool.emplace_back([&, t]() {
            for (size_t c = t; c < chunks; c += readers) {
                bool is_hot = c < hot_chunks;
                if (is_hot && pass > 0) continue;
                int b = (int)(c % kStage);
                while (released.load() + kStage <= c && !fail.load()) std::this_thread::yield();
                if (fail.load()) return;
                size_t cc = c - hot_chunks;
                bool in_cache = !is_hot && cache && cc < cache_chunks;
                if (in_cache && cached[c]) {
                    memcpy(stage[b], cache + cc * kChunk, kChunk); from_ram++;
                } else {
                    ssize_t n = pread(fd, stage[b], kChunk, (off_t)(c * kChunk));
                    if (n != (ssize_t)kChunk) { fprintf(stderr, "pread %zd\n", n); fail = true; return; }
                    from_nvme++;
                    if (in_cache) { memcpy(cache + cc * kChunk, stage[b], kChunk); cached[c] = 1; }
                }
                ready[b].store((long)c + 1);
            }
        });

        size_t rel_next = 0;  // next chunk to release
        std::vector<long> pending;
        for (size_t c = 0; c < chunks; ++c) {
            const uint64_t first = (uint64_t)c * (kChunk / 4);
            if (c < hot_chunks && pass > 0) {   // resident in VRAM already
                validate_k<<<256, 256, 0, st>>>((uint32_t*)(hot + c * kChunk), first, kChunk / 4, err);
                from_hot++;
                released++; rel_next++;   // keeps ordering counters consistent
                continue;
            }
            int b = (int)(c % kStage);
            while (ready[b].load() != (long)c + 1) { if (fail) return 1; std::this_thread::yield(); }
            uint8_t* dst = c < hot_chunks ? hot + c * kChunk : slots + (c % kSlots) * kChunk;
            check(cudaMemcpyAsync(dst, stage[b], kChunk, cudaMemcpyHostToDevice, st), "h2d");
            validate_k<<<256, 256, 0, st>>>((uint32_t*)dst, first, kChunk / 4, err);
            check(cudaEventRecord(ev[b], st), "ev");
            pending.push_back((long)c);
            if (pending.size() >= 3) {      // release oldest after its copy+validate is done
                size_t o = (size_t)pending.front(); pending.erase(pending.begin());
                check(cudaEventSynchronize(ev[o % kStage]), "ev sync");
                released.store(o + 1);
            }
            if ((c & 31) == 0) safety("stream");
        }
        check(cudaStreamSynchronize(st), "sync"); check(cudaGetLastError(), "kernel");
        released.store(chunks + kStage);
        for (auto& th : pool) th.join();
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        unsigned long long e = 0; check(cudaMemcpy(&e, err, 8, cudaMemcpyDeviceToHost), "err read");
        printf("pass,%ld,seconds=%.2f,GBps=%.2f,chunks_nvme=%zu,chunks_ram=%zu,chunks_vram_hot=%zu,errors_cum=%llu,ram_avail_MiB=%ld,swap_used_MiB=%ld\n",
               pass, s, (double)chunks * kChunk / s / 1e9, from_nvme.load(), from_ram.load(), from_hot.load(), e,
               meminfo_mib("MemAvailable:"), meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:"));
        (void)rel_next;
        if (e) { printf("summary,FAIL,errors=%llu\n", e); return 1; }
    }
    printf("summary,PASS,dataset_GiB=%ld,min_ram_avail_MiB=%ld\n", gib, g_min_avail);
    return 0;
}
