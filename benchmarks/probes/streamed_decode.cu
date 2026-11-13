// Streamed decode proxy: per "token", run a Q4_0 GEMV over every weight matrix of a real GGUF
// model (except the embedding table, which a real decode only row-gathers). This measures the
// weight-streaming ceiling. It does not implement attention/SSM math, so it generates no text.
//
// --ram-access staged: VRAM hot arena + pageable RAM cache + NVMe. RAM hits are memcpy'd into a
// pinned stage and cudaMemcpy'd into VRAM slots before the kernel.
// --ram-access direct: the same hot arena, but cold weights stay host-resident and gemv_q4 reads
// them in place. Pinned mapped memory is used up to RLIMIT_MEMLOCK minus the staging ring; any
// remainder is cudaMallocManaged with a CPU preferred location. NVMe fills those buffers once.
// H2D copies run on one stream and GEMV kernels on a second, so copies overlap with compute;
// slot reuse waits on the compute event of the item that last used the slot.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static constexpr size_t kMaxItem = 32u << 20;
static constexpr size_t kStageBytes = kMaxItem + 8192;
static constexpr int kStage = 8;
static constexpr int kSlots = 4;
static constexpr uint64_t kMapMargin = 64ull << 20;
static constexpr long kMigrateDropMiB = 512;

enum Tier { kNvme = 0, kHot = 1, kMapped = 2, kManaged = 3, kStaged = 4 };

static void check(cudaError_t s, const char* w) {
    if (s != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(s)); exit(1); }
}
static long meminfo_mib(const char* key) {
    FILE* f = fopen("/proc/meminfo", "r"); char l[256]; long v = -1; size_t n = strlen(key);
    while (f && fgets(l, sizeof l, f)) if (!strncmp(l, key, n)) { v = atol(l + n) / 1024; break; }
    if (f) fclose(f);
    return v;
}
static long g_swap0, g_floor = 6144, g_swap_tol = 1024;
static void safety(const char* where) {
    long a = meminfo_mib("MemAvailable:"), s = meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:");
    if (a < g_floor || s > g_swap0 + g_swap_tol) {
        fprintf(stderr, "ABORT at %s: MemAvailable=%ld MiB swap_used=%ld (baseline %ld)\n", where, a, s, g_swap0);
        _exit(4);
    }
}
static long free_vram_mib() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return -1;
    return (long)(free_b >> 20);
}

struct Tensor { std::string name; long K, N; uint64_t off, bytes; size_t y_base; };
struct Item {
    int tensor; long row0, rows; uint64_t file_off, bytes, aligned_off, head, read_len;
    int tier = kNvme; int64_t res_off = -1;
};

__device__ __forceinline__ float half_at(const uint8_t* p) {
    unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(u));
}
// x is quantized once to int8 per 32-element block (q8_1 style: scale xs[b], integer sum xsum[b]).
// The old fp32 kernel was compute-bound at ~24.6 GB/s (E033 probe); this dp4a kernel reaches ~275 GB/s.
__global__ void quant_x(const float* x, long nb, int8_t* xq, float* xs, int* xsum) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nb) return;
    float amax = 0.f; for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
    float d = amax / 127.f; float id = d > 0.f ? 1.f / d : 0.f; int s = 0;
    for (int i = 0; i < 32; ++i) { int q = __float2int_rn(x[b * 32 + i] * id); xq[b * 32 + i] = (int8_t)q; s += q; }
    xs[b] = d; xsum[b] = s;
}
// NB decode vectors share one read of each weight block (xq/xs/xsum are [NB][maxK(/32)], y is [NB][ytotal]).
template <int NB>
__global__ void gemv_q4(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum,
                        long xstride, float* y, long ystride) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32; const size_t rowbytes = (size_t)nb * 18;
    const uint8_t* r = w + (size_t)row * rowbytes;
    float acc[NB];
#pragma unroll
    for (int v = 0; v < NB; ++v) acc[v] = 0.f;
    for (long b = lane; b < nb; b += 32) {
        // Q4_0 block = 2-byte scale + 16 data bytes (low nibble = element j, high = j+16); only
        // 2-byte aligned, so build the four 32-bit words from 16-bit loads.
        const uint16_t* blk = (const uint16_t*)(r + (size_t)b * 18);
        const float d = __half2float(__ushort_as_half(blk[0]));
        uint32_t q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
        const uint32_t lo0 = q[0] & 0x0F0F0F0Fu, lo1 = q[1] & 0x0F0F0F0Fu, lo2 = q[2] & 0x0F0F0F0Fu, lo3 = q[3] & 0x0F0F0F0Fu;
        const uint32_t hi0 = (q[0] >> 4) & 0x0F0F0F0Fu, hi1 = (q[1] >> 4) & 0x0F0F0F0Fu, hi2 = (q[2] >> 4) & 0x0F0F0F0Fu, hi3 = (q[3] >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int v = 0; v < NB; ++v) {
            const int4* xp = (const int4*)(xq + v * xstride + b * 32);
            const int4 xa = xp[0], xb = xp[1];
            int s = 0;
            s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
            s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
            acc[v] += d * xs[v * (xstride / 32) + b] * (float)(s - 8 * xsum[v * (xstride / 32) + b]);
        }
    }
#pragma unroll
    for (int v = 0; v < NB; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[v * ystride + row] = a;
    }
}
static int g_batch = 1;
static int hot_stream = 1;
static void launch_gemv(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum,
                        long xstride, float* y, long ystride, cudaStream_t s) {
    const unsigned g = (unsigned)((rows + 3) / 4);
    switch (g_batch) {
        case 1: gemv_q4<1><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, xstride, y, ystride); break;
        case 2: gemv_q4<2><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, xstride, y, ystride); break;
        case 4: gemv_q4<4><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, xstride, y, ystride); break;
        case 8: gemv_q4<8><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, xstride, y, ystride); break;
        case 16: gemv_q4<16><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, xstride, y, ystride); break;
        case 32: gemv_q4<32><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, xstride, y, ystride); break;
        case 64: gemv_q4<64><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, xstride, y, ystride); break;
    }
}

static float xval(long i) { return (float)(((i * 37) % 101) - 50) / 64.0f; }
static float xvalv(int v, long i) { return xval(i + 13L * v); }   // distinct activation per batch vector
static float cpu_half(const uint8_t* p) {
    uint16_t u = (uint16_t)(p[0] | (p[1] << 8)); uint32_t s = (u >> 15) & 1, e = (u >> 10) & 31, m = u & 1023; float v;
    if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf((float)(m + 1024), (int)e - 25);
    return s ? -v : v;
}
static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

static cudaMemLocation loc_host() {
    cudaMemLocation loc{}; loc.type = cudaMemLocationTypeHost; loc.id = 0; return loc;
}
static cudaMemLocation loc_gpu() {
    cudaMemLocation loc{}; loc.type = cudaMemLocationTypeDevice; loc.id = 0; return loc;
}
static void prefer_cpu(uint8_t* p, uint64_t n, cudaStream_t st, int* prefetch_ok) {
    if (!p || !n) return;
    check(cudaMemAdvise(p, n, cudaMemAdviseSetPreferredLocation, loc_host()), "prefer cpu");
    check(cudaMemAdvise(p, n, cudaMemAdviseSetAccessedBy, loc_gpu()), "accessed by gpu");
    cudaError_t pf = cudaMemPrefetchAsync(p, n, loc_host(), 0u, st);
    if (pf != cudaSuccess) {
        fprintf(stderr, "cpu prefetch: %s\n", cudaGetErrorString(pf));
        cudaGetLastError();
        *prefetch_ok = 0;
    } else {
        check(cudaStreamSynchronize(st), "prefetch sync");
    }
}

int main(int argc, char** argv) {
    std::string model, manifest, ram_access = "staged";
    long hot_mib = 8192, cache_mib = 4096, passes = 6, readers = 4;
    int o_direct = 1, fit_cold = -1; long max_gib = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i]; const char* v = argv[i + 1];
        if (k == "--model") model = v; else if (k == "--manifest") manifest = v;
        else if (k == "--hot-mib") hot_mib = atol(v); else if (k == "--cache-mib") cache_mib = atol(v);
        else if (k == "--passes") passes = atol(v); else if (k == "--readers") readers = atol(v);
        else if (k == "--direct") o_direct = atoi(v); else if (k == "--ram-floor-mib") g_floor = atol(v);
        else if (k == "--swap-tol-mib") g_swap_tol = atol(v); else if (k == "--limit-gib") max_gib = atol(v);
        else if (k == "--ram-access") ram_access = v; else if (k == "--fit-cold") fit_cold = atoi(v);
        else if (k == "--batch") g_batch = atoi(v); else if (k == "--hot-stream") hot_stream = atoi(v);
    }
    const bool ram_direct = ram_access == "direct";
    if (ram_access != "staged" && !ram_direct) { fprintf(stderr, "--ram-access must be staged or direct\n"); return 2; }
    if (fit_cold < 0) fit_cold = ram_direct ? 1 : 0;
    if (model.empty() || manifest.empty()) { fprintf(stderr, "need --model and --manifest\n"); return 2; }
    g_swap0 = meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:"); safety("start");

    struct rlimit lim{};
    uint64_t memlock = 0; int memlock_inf = 0;
    if (getrlimit(RLIMIT_MEMLOCK, &lim) != 0) { perror("getrlimit"); return 1; }
    if (lim.rlim_cur == RLIM_INFINITY) memlock_inf = 1; else memlock = lim.rlim_cur;
    const uint64_t stage_lock = (uint64_t)kStage * (kStageBytes + 4096);
    uint64_t map_budget = ~0ull;
    if (!memlock_inf) {
        if (memlock <= stage_lock + kMapMargin) map_budget = 0;
        else map_budget = memlock - stage_lock - kMapMargin;
    }

    std::vector<Tensor> tensors; long maxK = 0; size_t ytotal = 0;
    { FILE* f = fopen(manifest.c_str(), "r"); char nm[256]; long K, N; unsigned long long off, by;
      while (f && fscanf(f, "%255s %ld %ld %llu %llu", nm, &K, &N, &off, &by) == 5) {
          if (!strcmp(nm, "token_embd.weight")) continue;
          if (K % 32) continue;
          tensors.push_back({nm, K, N, off, by, ytotal}); ytotal += (size_t)N; if (K > maxK) maxK = K; }
      if (f) fclose(f); }
    if (tensors.empty()) { fprintf(stderr, "no tensors in manifest\n"); return 1; }
    std::vector<Item> items; uint64_t total_bytes = 0;
    for (size_t t = 0; t < tensors.size(); ++t) {
        const Tensor& T = tensors[t]; const uint64_t rowbytes = (uint64_t)(T.K / 32) * 18;
        const long per = (long)(kMaxItem / rowbytes); if (per < 1) { fprintf(stderr, "row too large\n"); return 1; }
        for (long r = 0; r < T.N; r += per) {
            Item it; it.tensor = (int)t; it.row0 = r; it.rows = std::min<long>(per, T.N - r);
            it.file_off = T.off + (uint64_t)r * rowbytes; it.bytes = (uint64_t)it.rows * rowbytes;
            it.aligned_off = it.file_off & ~4095ull; it.head = it.file_off - it.aligned_off;
            it.read_len = align_up(it.head + it.bytes, 4096); items.push_back(it); total_bytes += it.bytes;
        }
        if (max_gib && total_bytes > (uint64_t)max_gib << 30) break;
    }
    const uint64_t host_cap = fit_cold ? ~0ull : (uint64_t)cache_mib << 20;
    uint64_t hot_used = 0, mapped_used = 0, managed_used = 0, staged_used = 0, host_used = 0;
    for (Item& it : items) {
        if (hot_used + it.bytes + 16 <= (uint64_t)hot_mib << 20) {
            it.tier = kHot; it.res_off = (int64_t)hot_used; hot_used = align_up(hot_used + it.bytes, 256); continue;
        }
        const uint64_t need = align_up(it.bytes, 4096);
        if (ram_direct) {
            if (mapped_used + need <= map_budget && host_used + need <= host_cap) {
                it.tier = kMapped; it.res_off = (int64_t)mapped_used; mapped_used += need; host_used += need;
            } else if (host_used + need <= host_cap) {
                it.tier = kManaged; it.res_off = (int64_t)managed_used; managed_used += need; host_used += need;
            }
        } else if (staged_used + it.bytes + 4096 <= host_cap) {
            it.tier = kStaged; it.res_off = (int64_t)staged_used; staged_used = align_up(staged_used + it.bytes, 4096);
        }
    }
    uint64_t nvme_bytes = 0; for (auto& it : items) if (it.tier == kNvme) nvme_bytes += it.bytes;
    // Pin the head of the staged cache within the memlock budget: cached H2D copies then run
    // straight from the pinned cache with no CPU memcpy through the staging ring.
    const uint64_t pin_bytes = std::min<uint64_t>(staged_used, map_budget);
    const long host_mib = (long)((mapped_used + managed_used + staged_used) >> 20);
    if (meminfo_mib("MemAvailable:") < g_floor + host_mib) {
        fprintf(stderr, "ABORT before alloc: MemAvailable=%ld MiB host_tier=%ld MiB floor=%ld\n",
                meminfo_mib("MemAvailable:"), host_mib, g_floor);
        return 4;
    }
    printf("config,model=%s,tensors=%zu,items=%zu,stream_GiB=%.2f,hot_MiB=%.0f,cache_MiB=%.0f,pinned_MiB=%.0f,mapped_MiB=%.0f,managed_MiB=%.0f,nvme_per_token_GiB=%.2f,o_direct=%d,ram_access=%s,fit_cold=%d,readers=%ld,passes=%ld,memlock_MiB=%.0f,map_budget_MiB=%.0f\n",
           model.c_str(), tensors.size(), items.size(), total_bytes / 1073741824.0, hot_used / 1048576.0,
           staged_used / 1048576.0, pin_bytes / 1048576.0, mapped_used / 1048576.0, managed_used / 1048576.0, nvme_bytes / 1073741824.0,
           o_direct, ram_access.c_str(), fit_cold, readers, passes,
           memlock_inf ? -1.0 : memlock / 1048576.0, map_budget == ~0ull ? -1.0 : map_budget / 1048576.0);

    check(cudaSetDevice(0), "device");
    uint8_t *hot = nullptr, *slots = nullptr, *mapped_host = nullptr, *mapped_dev = nullptr, *managed = nullptr, *staged_cache = nullptr;
    float *dx = nullptr, *dy = nullptr;
    if (hot_used) check(cudaMalloc((void**)&hot, hot_used), "hot arena");
    bool need_slots = false; for (auto& it : items) if (it.tier == kNvme || it.tier == kStaged) need_slots = true;
    if (need_slots) check(cudaMalloc((void**)&slots, (size_t)kSlots * kMaxItem), "slots");
    check(cudaMalloc((void**)&dx, (size_t)maxK * 4 * g_batch), "x"); check(cudaMalloc((void**)&dy, ytotal * 4 * g_batch), "y");
    int8_t* dxq = nullptr; float* dxs = nullptr; int* dxsum = nullptr;
    check(cudaMalloc((void**)&dxq, (size_t)maxK * g_batch), "xq"); check(cudaMalloc((void**)&dxs, (size_t)maxK / 32 * 4 * g_batch), "xs"); check(cudaMalloc((void**)&dxsum, (size_t)maxK / 32 * 4 * g_batch), "xsum");
    { std::vector<float> hx((size_t)maxK * g_batch); for (int v = 0; v < g_batch; ++v) for (long i = 0; i < maxK; ++i) hx[(size_t)v * maxK + i] = xvalv(v, i); check(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x copy"); }
    quant_x<<<(unsigned)((maxK / 32 * g_batch + 127) / 128), 128>>>(dx, maxK / 32 * g_batch, dxq, dxs, dxsum); check(cudaDeviceSynchronize(), "quant x");
    uint8_t* stage[kStage]; for (auto& s : stage) check(cudaHostAlloc((void**)&s, kStageBytes + 4096, cudaHostAllocPortable), "stage");
    if (mapped_used) {
        check(cudaHostAlloc((void**)&mapped_host, mapped_used, cudaHostAllocMapped | cudaHostAllocPortable), "mapped cold");
        check(cudaHostGetDevicePointer((void**)&mapped_dev, mapped_host, 0), "mapped device pointer");
    }
    int prefetch_ok = 1;
    if (managed_used) {
        check(cudaMallocManaged((void**)&managed, managed_used), "managed cold");
        prefer_cpu(managed, managed_used, 0, &prefetch_ok);
    }
    uint8_t* pinned_cache = nullptr;
    if (staged_used) {
        if (posix_memalign((void**)&staged_cache, 4096, staged_used)) { fprintf(stderr, "cache alloc\n"); return 1; }
        if (pin_bytes) check(cudaHostAlloc((void**)&pinned_cache, pin_bytes, cudaHostAllocPortable), "pinned cache");
    }
    std::vector<char> cached(items.size(), 0);
    int fd = open(model.c_str(), O_RDONLY | (o_direct ? O_DIRECT : 0)); if (fd < 0) { perror("open"); return 1; }
    cudaStream_t st, stc, sth; check(cudaStreamCreate(&st), "stream"); check(cudaStreamCreate(&stc), "compute stream"); check(cudaStreamCreate(&sth), "hot stream");
    cudaEvent_t ev[kStage], evc[kSlots]; for (auto& e : ev) check(cudaEventCreate(&e), "event"); for (auto& e : evc) check(cudaEventCreate(&e), "event");
    // Prime each slot event on the compute stream so the first reuse-wait per slot is a no-op.
    for (auto& e : evc) check(cudaEventRecord(e, stc), "evc prime");
    check(cudaStreamSynchronize(stc), "evc prime sync");
    const long free_after_alloc = free_vram_mib();
    printf("alloc,free_vram_MiB=%ld,ram_avail_MiB=%ld,swap_used_MiB=%ld,prefetch_ok=%d\n",
           free_after_alloc, meminfo_mib("MemAvailable:"), meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:"), prefetch_ok);

    const size_t n = items.size(); double pass_s_best = 1e30; uint64_t best_h2d_cold = ~0ull;
    long free_base = free_after_alloc, free_min_steady = free_after_alloc; int managed_migrated = 0;
    for (long pass = 0; pass < passes; ++pass) {
        std::atomic<size_t> released{0}; std::atomic<long> ready[kStage]; for (auto& r : ready) r = 0;
        std::atomic<size_t> n_nvme{0}, n_ram{0}, n_hot{0}, n_mapped{0}, n_managed{0};
        std::atomic<uint64_t> bytes_vram{0}, bytes_mapped{0}, bytes_managed{0}, bytes_nvme{0}, h2d_cold{0};
        std::atomic<bool> fail{false};
        auto t0 = std::chrono::steady_clock::now();
        std::vector<size_t> sl;
        for (size_t c = 0; c < n; ++c) {
            const Item& it = items[c];
            if (ram_direct) { if ((pass == 0 && it.tier != kNvme) || it.tier == kNvme) sl.push_back(c); }
            else if (!(it.tier == kHot && pass > 0)) sl.push_back(c);
        }
        std::vector<std::thread> pool;
        for (long t = 0; t < readers; ++t) pool.emplace_back([&, t]() {
            for (size_t k = t; k < sl.size(); k += (size_t)readers) {
                const size_t c = sl[k]; Item& it = items[c];
                const bool in_pin = it.tier == kStaged && (uint64_t)it.res_off + it.bytes <= pin_bytes;
                if (pass > 0 && in_pin && cached[c]) continue;   // main thread copies straight from the pinned cache
                int b = (int)(k % kStage);
                while (released.load() + kStage <= k && !fail.load()) std::this_thread::yield();
                if (fail.load()) return;
                if (it.tier == kStaged && cached[c]) {
                    memcpy(stage[b] + it.head, staged_cache + it.res_off, it.bytes); n_ram++;
                } else {
                    ssize_t got = pread(fd, stage[b], it.read_len, (off_t)it.aligned_off);
                    if (got < (ssize_t)(it.head + it.bytes)) {
                        fprintf(stderr, "pread %zd < %llu\n", got, (unsigned long long)(it.head + it.bytes)); fail = true; return;
                    }
                    n_nvme++; bytes_nvme += it.bytes;
                    if (it.tier == kStaged) { memcpy((in_pin ? pinned_cache : staged_cache) + it.res_off, stage[b] + it.head, it.bytes); cached[c] = 1; }
                }
                ready[b].store((long)k + 1);
            }
        });
        std::vector<size_t> pending;
        size_t k = 0;
        if (hot_stream && pass > 0) {
            // Hot items need no copy: launch all of their GEMVs on a separate stream up front so they run
            // under the cold H2D transfers rather than ahead of them in the compute stream.
            for (size_t c = 0; c < n; ++c) {
                const Item& it = items[c]; if (it.tier != kHot) continue; const Tensor& T = tensors[it.tensor];
                launch_gemv(hot + it.res_off, T.K, it.rows, dxq, dxs, dxsum, maxK, dy + T.y_base + it.row0, (long)ytotal, sth);
                n_hot++; bytes_vram += it.bytes;
            }
        }
        for (size_t c = 0; c < n; ++c) {
            Item& it = items[c]; const Tensor& T = tensors[it.tensor]; const uint8_t* src = nullptr;
            const bool resident = pass > 0 && (it.tier == kHot || it.tier == kMapped || it.tier == kManaged);
            bool slot_item = false;
            if (hot_stream && pass > 0 && it.tier == kHot) continue;
            if (resident && it.tier == kHot) { src = hot + it.res_off; n_hot++; bytes_vram += it.bytes; }
            else if (resident && it.tier == kMapped) { src = mapped_dev + it.res_off; n_mapped++; n_ram++; bytes_mapped += it.bytes; }
            else if (resident && it.tier == kManaged) { src = managed + it.res_off; n_managed++; n_ram++; bytes_managed += it.bytes; }
            else {
                const bool in_pin = it.tier == kStaged && (uint64_t)it.res_off + it.bytes <= pin_bytes;
                const bool direct_pin = pass > 0 && in_pin && cached[c];
                int b = (int)(k % kStage);
                if (!direct_pin) while (ready[b].load() != (long)k + 1) { if (fail) return 1; std::this_thread::yield(); }
                // Slot reuse: wait for the GEMV that last used this slot (one event per slot).
                if (it.tier == kNvme || it.tier == kStaged) check(cudaStreamWaitEvent(st, evc[k % kSlots], 0), "slot wait");
                uint8_t* dst = nullptr; cudaMemcpyKind kind = cudaMemcpyHostToDevice;
                if (it.tier == kHot) dst = hot + it.res_off;
                else if (it.tier == kMapped) { dst = mapped_host + it.res_off; kind = cudaMemcpyHostToHost; }
                else if (it.tier == kManaged) dst = managed + it.res_off;
                else { dst = slots + (k % kSlots) * kMaxItem; slot_item = true; }
                const uint8_t* csrc = direct_pin ? pinned_cache + it.res_off : stage[b] + it.head;
                check(cudaMemcpyAsync(dst, csrc, it.bytes, kind, st), "copy");
                if (direct_pin) n_ram++;
                if (it.tier == kNvme || it.tier == kStaged) h2d_cold += it.bytes;
                if (it.tier == kMapped) { src = mapped_dev + it.res_off; n_mapped++; n_ram++; bytes_mapped += it.bytes; }
                else if (it.tier == kManaged) { src = dst; n_managed++; n_ram++; bytes_managed += it.bytes; }
                else { src = dst; if (it.tier == kHot) n_hot++; bytes_vram += it.bytes; }
                check(cudaEventRecord(ev[b], st), "ev");
                check(cudaStreamWaitEvent(stc, ev[b], 0), "wait copy");
                pending.push_back(k); ++k;
            }
            launch_gemv(src, T.K, it.rows, dxq, dxs, dxsum, maxK, dy + T.y_base + it.row0, (long)ytotal, stc);
            if (slot_item) check(cudaEventRecord(evc[(k - 1) % kSlots], stc), "evc");
            if (pending.size() >= 3) {
                size_t o = pending.front(); pending.erase(pending.begin());
                check(cudaEventSynchronize(ev[o % kStage]), "ev sync"); released.store(o + 1);
            }
            if ((c & 31) == 0) safety("stream");
        }
        check(cudaStreamSynchronize(st), "sync"); check(cudaStreamSynchronize(stc), "sync c"); check(cudaStreamSynchronize(sth), "sync h"); check(cudaGetLastError(), "kernel");
        released.store(n + kStage); for (auto& th : pool) th.join();
        if (pass == 0 && managed_used) prefer_cpu(managed, managed_used, st, &prefetch_ok);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const long fv = free_vram_mib();
        if (pass == 0) free_base = fv;
        if (pass > 0) {
            if (s < pass_s_best) { pass_s_best = s; best_h2d_cold = h2d_cold.load(); }
            if (fv >= 0 && (free_min_steady < 0 || fv < free_min_steady)) free_min_steady = fv;
            if (managed_used && free_base >= 0 && fv >= 0 && fv + kMigrateDropMiB < free_base) managed_migrated = 1;
        }
        printf("token,%ld,seconds=%.3f,tok_per_s=%.3f,eff_GBps=%.2f,items_nvme=%zu,items_ram=%zu,items_vram=%zu,items_mapped=%zu,items_managed=%zu,bytes_vram=%llu,bytes_mapped=%llu,bytes_managed=%llu,bytes_nvme=%llu,h2d_cold_bytes=%llu,free_vram_MiB=%ld,ram_avail_MiB=%ld,swap_used_MiB=%ld\n",
               pass, s, 1.0 / s, total_bytes / s / 1e9, n_nvme.load(), n_ram.load(), n_hot.load(), n_mapped.load(), n_managed.load(),
               (unsigned long long)bytes_vram.load(), (unsigned long long)bytes_mapped.load(), (unsigned long long)bytes_managed.load(),
               (unsigned long long)bytes_nvme.load(), (unsigned long long)h2d_cold.load(), fv,
               meminfo_mib("MemAvailable:"), meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:"));
    }
    printf("migration_hint,threshold_MiB=%ld,baseline_free_vram_MiB=%ld,min_steady_free_vram_MiB=%ld,managed_migrated=%d,prefetch_ok=%d,note=cudaMemGetInfo_is_a_global_counter\n",
           kMigrateDropMiB, free_base, free_min_steady, managed_migrated, prefetch_ok);

    std::vector<float> hy(ytotal * g_batch); check(cudaMemcpy(hy.data(), dy, ytotal * 4 * g_batch, cudaMemcpyDeviceToHost), "y copy");
    int vfd = open(model.c_str(), O_RDONLY); long checked = 0, bad = 0; double worst = 0;
    for (size_t t = 0; t < tensors.size(); ++t) {
        const Tensor& T = tensors[t]; const size_t rowbytes = (size_t)(T.K / 32) * 18; std::vector<uint8_t> row(rowbytes);
        bool covered = false; for (auto& it : items) if (it.tensor == (int)t) { covered = true; break; } if (!covered) continue;
        for (int s = 0; s < 8; ++s) {
            long r = (long)((T.N - 1) * (long)s / 7);
            bool in_range = false; for (auto& it : items) if (it.tensor == (int)t && r >= it.row0 && r < it.row0 + it.rows) { in_range = true; break; }
            if (!in_range) continue;
            if (pread(vfd, row.data(), rowbytes, (off_t)(T.off + (uint64_t)r * rowbytes)) != (ssize_t)rowbytes) { fprintf(stderr, "verify pread\n"); return 1; }
            for (int v = 0; v < g_batch; ++v) {
            double acc = 0, mag = 0;
            for (long b = 0; b < T.K / 32; ++b) {
                const uint8_t* blk = row.data() + b * 18; float d = cpu_half(blk);
                // Mirror the GPU's int8 activation quantization exactly (same rounding, integer dot).
                float amax = 0.f; for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xvalv(v, b * 32 + i)));
                float dx_ = amax / 127.f, idx = dx_ > 0.f ? 1.f / dx_ : 0.f; int xq_[32];
                for (int i = 0; i < 32; ++i) xq_[i] = (int)nearbyintf(xvalv(v, b * 32 + i) * idx);
                long sb = 0; for (int j = 0; j < 16; ++j) { uint8_t q = blk[2 + j]; sb += (long)((int)(q & 15) - 8) * xq_[j] + (long)((int)(q >> 4) - 8) * xq_[j + 16]; }
                acc += (double)d * dx_ * (double)sb; mag += fabs(d) * 8 * 32 * fabs(dx_) * 127;
            }
            double g = hy[(size_t)v * ytotal + T.y_base + r], err = fabs(g - acc), tol = 1e-4 * mag + 1e-3; ++checked; if (checked % 97 == 1) printf("sample,%s,v=%d,row=%ld,gpu=%.6f,cpu=%.6f\n", T.name.c_str(), v, r, g, acc);
            if (!(err <= tol)) { ++bad; fprintf(stderr, "MISMATCH %s v=%d row %ld gpu=%g cpu=%g\n", T.name.c_str(), v, r, g, acc); }
            worst = std::max(worst, err / (mag + 1e-9));
            }
        }
    }
    printf("verify,rows_checked=%ld,mismatches=%ld,worst_rel_err=%.3g,batch=%d\n", checked, bad, worst, g_batch);
    printf("summary,%s,batch=%d,best_streaming_passes_per_s=%.3f,best_aggregate_tok_per_s=%.3f,best_h2d_cold_bytes=%llu,managed_migrated=%d\n",
           bad ? "FAIL" : "PASS", g_batch, pass_s_best < 1e29 ? 1.0 / pass_s_best : 0.0, pass_s_best < 1e29 ? g_batch / pass_s_best : 0.0,
           best_h2d_cold == ~0ull ? 0ull : (unsigned long long)best_h2d_cold, managed_migrated);
    return bad ? 1 : 0;
}
