#include "../src/residency_manager.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr size_t kPageBytes = 2u * 1024u * 1024u;
constexpr size_t kPressureMiB = 128;
constexpr uint64_t kRequests = 4u * 1024u * 1024u;
constexpr size_t kHotPages = 32;

struct Options {
    int host_mib = 256;
    int cache_mib = 64;
    int passes = 4;
    int repeats = 5;
};

struct BlockStats {
    uint32_t hits;
    uint32_t errors;
};

struct AccessStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t errors = 0;
    double milliseconds = 0.0;
};

[[noreturn]] void fail(const char* operation, cudaError_t status) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) fail(operation, status);
}

int parse_positive(const char* text, const char* option, int maximum) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || value <= 0 || value > maximum) {
        std::fprintf(stderr, "Invalid value for %s: %s (allowed 1..%d)\n",
                     option, text, maximum);
        std::exit(2);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--host-mib") == 0 && i + 1 < argc) {
            options.host_mib = parse_positive(argv[++i], "--host-mib", 512);
        } else if (std::strcmp(argv[i], "--cache-mib") == 0 && i + 1 < argc) {
            options.cache_mib = parse_positive(argv[++i], "--cache-mib", 128);
        } else if (std::strcmp(argv[i], "--passes") == 0 && i + 1 < argc) {
            options.passes = parse_positive(argv[++i], "--passes", 8);
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            options.repeats = parse_positive(argv[++i], "--repeats", 5);
        } else {
            std::fprintf(stderr,
                "Usage: %s [--host-mib 256..512] [--cache-mib 64..128] [--passes 1..8] [--repeats 1..5]\n",
                argv[0]);
            std::exit(2);
        }
    }
    if (options.host_mib < 256 || options.host_mib % 2 != 0 ||
        options.cache_mib < 64 || options.cache_mib % 2 != 0 || options.cache_mib > options.host_mib / 2 ||
        options.passes < 1 || options.repeats < 1) {
        std::fprintf(stderr,
            "Invalid plan: host/cache MiB must be even, host >=128, cache <= half of host, and passes/repeats >=1.\n");
        std::exit(2);
    }
    return options;
}

__host__ __device__ float component_value(uint64_t index) {
    return static_cast<float>((index % 251u) + 1u) * (1.0f / 256.0f);
}

__host__ __device__ float4 expected_value(uint64_t vector_index) {
    float4 value{};
    const uint64_t first = vector_index * 4u;
    value.x = component_value(first);
    value.y = component_value(first + 1u);
    value.z = component_value(first + 2u);
    value.w = component_value(first + 3u);
    return value;
}

__device__ bool same_value(float4 left, float4 right) {
    return fabsf(left.x - right.x) < 1.0e-6f &&
           fabsf(left.y - right.y) < 1.0e-6f &&
           fabsf(left.z - right.z) < 1.0e-6f &&
           fabsf(left.w - right.w) < 1.0e-6f;
}

__global__ void initialize_data(float4* data, uint64_t vectors) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < vectors; index += stride) {
        data[index] = expected_value(index);
    }
}

__global__ void validate_full_range(const float4* data, uint64_t vectors,
                                    unsigned long long* errors) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < vectors; index += stride) {
        if (!same_value(data[index], expected_value(index))) atomicAdd(errors, 1ull);
    }
}

__global__ void access_requests(const float4* mapped_host, float4* const* frame_pointers,
                                const int* page_to_frame, const uint32_t* requests,
                                uint64_t request_count, uint64_t vectors_per_page,
                                bool count_pages, uint32_t* page_access_counts,
                                BlockStats* block_stats) {
    __shared__ uint32_t hit_flags[kThreads];
    __shared__ uint32_t error_flags[kThreads];

    const uint64_t request = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t hit = 0;
    uint32_t error = 0;

    if (request < request_count) {
        const uint32_t vector_index = requests[request];
        const uint32_t page = static_cast<uint32_t>(vector_index / vectors_per_page);
        const uint64_t within_page = vector_index % vectors_per_page;
        const int frame = page_to_frame[page];
        const float4 actual = frame >= 0
            ? frame_pointers[frame][within_page]
            : mapped_host[vector_index];
        if (frame >= 0) hit = 1;
        if (!same_value(actual, expected_value(vector_index))) error = 1;
        if (count_pages) atomicAdd(page_access_counts + page, 1u);
    }

    hit_flags[threadIdx.x] = hit;
    error_flags[threadIdx.x] = error;
    __syncthreads();

    for (unsigned int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            hit_flags[threadIdx.x] += hit_flags[threadIdx.x + offset];
            error_flags[threadIdx.x] += error_flags[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_stats[blockIdx.x] = {hit_flags[0], error_flags[0]};
    }
}

std::vector<uint32_t> make_requests(size_t page_count, size_t hot_start,
                                    size_t hot_pages, uint64_t vectors_per_page,
                                    uint64_t request_count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> requests(static_cast<size_t>(request_count));
    const size_t cold_pages = page_count - hot_pages;

    for (uint64_t i = 0; i < request_count; ++i) {
        size_t page = 0;
        if (rng() % 100u < 90u) {
            page = hot_start + static_cast<size_t>(rng() % hot_pages);
        } else {
            size_t cold_index = static_cast<size_t>(rng() % cold_pages);
            page = cold_index >= hot_start ? cold_index + hot_pages : cold_index;
        }
        const uint64_t within = rng() % vectors_per_page;
        requests[static_cast<size_t>(i)] =
            static_cast<uint32_t>(page * vectors_per_page + within);
    }
    return requests;
}

AccessStats run_access(const float4* host, float4* const* frame_pointers,
                       const int* page_to_frame, const uint32_t* requests,
                       uint64_t request_count, uint64_t vectors_per_page,
                       bool count_pages, uint32_t* device_page_counts,
                       BlockStats* device_blocks, std::vector<BlockStats>& host_blocks,
                       std::vector<uint32_t>* host_page_counts,
                       cudaStream_t stream) {
    if (count_pages) {
        check(cudaMemsetAsync(device_page_counts, 0,
            host_page_counts->size() * sizeof(uint32_t), stream), "clear page counters");
    }
    const int blocks = static_cast<int>((request_count + kThreads - 1) / kThreads);
    const auto start = std::chrono::steady_clock::now();
    access_requests<<<blocks, kThreads, 0, stream>>>(
        host, frame_pointers, page_to_frame, requests, request_count,
        vectors_per_page, count_pages, device_page_counts, device_blocks);
    check(cudaGetLastError(), "launch access/counter kernel");
    check(cudaStreamSynchronize(stream), "wait for access/counter kernel");
    const auto stop = std::chrono::steady_clock::now();

    AccessStats result;
    result.milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();
    check(cudaMemcpy(host_blocks.data(), device_blocks,
        blocks * sizeof(BlockStats), cudaMemcpyDeviceToHost), "copy access validation stats");
    for (const BlockStats& block : host_blocks) {
        result.hits += block.hits;
        result.errors += block.errors;
    }
    result.misses = request_count - result.hits;

    if (count_pages) {
        check(cudaMemcpy(host_page_counts->data(), device_page_counts,
            host_page_counts->size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
            "copy GPU access counters");
    }
    return result;
}

double run_direct(const float4* host, float4* const* frame_pointers,
                  const int* page_to_frame, const uint32_t* requests,
                  uint64_t request_count, uint64_t vectors_per_page, int passes,
                  BlockStats* device_blocks, std::vector<BlockStats>& host_blocks,
                  cudaStream_t stream, uint64_t* errors) {
    double total_ms = 0.0;
    *errors = 0;
    for (int pass = 0; pass < passes; ++pass) {
        AccessStats result = run_access(host, frame_pointers, page_to_frame,
            requests, request_count, vectors_per_page, false, nullptr,
            device_blocks, host_blocks, nullptr, stream);
        total_ms += result.milliseconds;
        *errors += result.errors;
    }
    return total_ms;
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    if (samples.empty()) return 0.0;
    const size_t middle = samples.size() / 2;
    return samples.size() % 2 ? samples[middle] :
        (samples[middle - 1] + samples[middle]) * 0.5;
}

bool check_manager(bool okay, const std::string& error, const char* operation) {
    if (okay) return true;
    std::fprintf(stderr, "%s: %s\n", operation, error.c_str());
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const size_t host_bytes = static_cast<size_t>(options.host_mib) * 1024u * 1024u;
    const size_t cache_bytes = static_cast<size_t>(options.cache_mib) * 1024u * 1024u;
    const size_t page_count = host_bytes / kPageBytes;
    const size_t max_cache_pages = cache_bytes / kPageBytes;
    const uint64_t vectors_per_page = kPageBytes / sizeof(float4);
    const uint64_t vectors = host_bytes / sizeof(float4);
    const uint64_t request_count = kRequests;
    const int blocks = static_cast<int>((request_count + kThreads - 1) / kThreads);
    const size_t request_bytes = request_count * sizeof(uint32_t);

    const size_t planned_device_mib = options.cache_mib + kPressureMiB +
        request_bytes / (1024u * 1024u) + 32;
    const size_t planned_host_mib = options.host_mib +
        request_bytes / (1024u * 1024u) + 16;
    if (page_count <= 2 * kHotPages || max_cache_pages < kHotPages ||
        vectors > UINT32_MAX || planned_device_mib > 2048 || planned_host_mib > 4096) {
        std::fprintf(stderr, "The selected plan violates the benchmark's bounded memory plan.\n");
        return 2;
    }

    check(cudaSetDevice(0), "select GPU 0");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "query GPU");
    int concurrent_managed = 0;
    check(cudaDeviceGetAttribute(&concurrent_managed,
        cudaDevAttrConcurrentManagedAccess, 0), "query managed access support");
    if (!prop.canMapHostMemory || !prop.unifiedAddressing || !concurrent_managed) {
        std::fprintf(stderr, "The GPU lacks mapped-host, UVA, or concurrent-managed support.\n");
        return 3;
    }

    std::printf("device,%s,cc=%d.%d,host_MiB=%d,cache_MiB=%d,page_bytes=%zu,pages=%zu,pressure_MiB=%zu,passes=%d,repeats=%d,requests=%llu\n",
        prop.name, prop.major, prop.minor, options.host_mib, options.cache_mib,
        kPageBytes, page_count, kPressureMiB, options.passes, options.repeats,
        static_cast<unsigned long long>(request_count));
    std::printf("mechanism,GPU_access_counters=frequency_tracking,CPU_role=rank_page_metadata_and_schedule_copies,host_backing=mapped_pinned_RAM,cold_page_policy=direct_host_access,hot_page_policy=VRAM_cache\n");

    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    float4* host_cpu = nullptr;
    float4* host_device = nullptr;
    uint32_t* device_requests = nullptr;
    uint32_t* device_page_counts = nullptr;
    BlockStats* device_blocks = nullptr;
    unsigned long long* device_full_errors = nullptr;
    check(cudaHostAlloc(reinterpret_cast<void**>(&host_cpu), host_bytes,
        cudaHostAllocMapped | cudaHostAllocPortable), "allocate pinned host backing");
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host_cpu, 0),
        "get mapped-host GPU pointer");
    check(cudaMalloc(reinterpret_cast<void**>(&device_requests), request_bytes),
        "allocate request list");
    check(cudaMalloc(reinterpret_cast<void**>(&device_page_counts), page_count * sizeof(uint32_t)),
        "allocate GPU page counters");
    check(cudaMalloc(reinterpret_cast<void**>(&device_blocks), blocks * sizeof(BlockStats)),
        "allocate access statistics");
    check(cudaMalloc(reinterpret_cast<void**>(&device_full_errors), sizeof(unsigned long long)),
        "allocate full-range validation counter");

    const int init_blocks = std::max(1, std::min(prop.multiProcessorCount * 16, 65535));
    initialize_data<<<init_blocks, kThreads, 0, stream>>>(host_device, vectors);
    check(cudaGetLastError(), "launch GPU dataset initialization");
    check(cudaMemsetAsync(device_full_errors, 0, sizeof(unsigned long long), stream),
        "clear full-range validation counter");
    validate_full_range<<<init_blocks, kThreads, 0, stream>>>(
        host_device, vectors, device_full_errors);
    check(cudaGetLastError(), "launch full-range GPU validation");
    check(cudaStreamSynchronize(stream), "wait for full-range validation");
    unsigned long long full_errors = 0;
    check(cudaMemcpy(&full_errors, device_full_errors, sizeof(full_errors),
        cudaMemcpyDeviceToHost), "copy full-range validation result");
    if (full_errors != 0) {
        std::fprintf(stderr, "Initial GPU validation found %llu mismatches.\n", full_errors);
        return 1;
    }

    unified_memory::ResidencyManager manager;
    std::string manager_error;
    if (!check_manager(manager.create(host_device, host_bytes, kPageBytes,
        max_cache_pages, &manager_error), manager_error, "create residency manager")) return 1;

    std::vector<uint32_t> host_page_counts(page_count, 0);
    std::vector<BlockStats> host_blocks(blocks);
    std::vector<uint32_t> empty_counts(page_count, 0);
    unified_memory::ResidencyUpdate no_cache_update;
    if (!check_manager(manager.update(empty_counts, 0, &no_cache_update,
        &manager_error), manager_error, "initialize empty cache map")) return 1;

    std::vector<double> baseline_samples;
    std::vector<double> cached_samples;
    std::vector<double> cached_hit_percent;
    uint64_t last_baseline_errors = 0;
    uint64_t last_cached_errors = 0;
    size_t last_promoted = 0;
    size_t last_cache_bytes = 0;

    for (int repeat = 1; repeat <= options.repeats; ++repeat) {
        const std::vector<uint32_t> requests = make_requests(page_count, 0,
            kHotPages, vectors_per_page, request_count, 0xC001u + repeat);
        check(cudaMemcpy(device_requests, requests.data(), request_bytes,
            cudaMemcpyHostToDevice), "copy phase-one request trace");

        if (!check_manager(manager.update(empty_counts, 0, &no_cache_update,
            &manager_error), manager_error, "clear cache before cold comparison")) return 1;

        uint64_t baseline_errors = 0;
        const double baseline_ms = run_direct(host_device,
            manager.device_frame_pointers(), manager.device_page_to_frame(),
            device_requests, request_count, vectors_per_page, options.passes + 1,
            device_blocks, host_blocks, stream, &baseline_errors);
        baseline_samples.push_back(baseline_ms);

        AccessStats training = run_access(host_device, manager.device_frame_pointers(),
            manager.device_page_to_frame(), device_requests, request_count,
            vectors_per_page, true, device_page_counts, device_blocks,
            host_blocks, &host_page_counts, stream);

        unified_memory::ResidencyUpdate update;
        const auto manager_start = std::chrono::steady_clock::now();
        if (!check_manager(manager.update(host_page_counts, max_cache_pages,
            &update, &manager_error), manager_error, "promote observed hot pages")) return 1;
        const auto manager_stop = std::chrono::steady_clock::now();
        const double manager_ms = std::chrono::duration<double, std::milli>(
            manager_stop - manager_start).count();

        double cached_ms = 0.0;
        uint64_t cached_errors = 0;
        uint64_t hits = 0;
        uint64_t accesses = 0;
        for (int pass = 0; pass < options.passes; ++pass) {
            AccessStats cached = run_access(host_device,
                manager.device_frame_pointers(), manager.device_page_to_frame(),
                device_requests, request_count, vectors_per_page, false,
                nullptr, device_blocks, host_blocks, nullptr, stream);
            cached_ms += cached.milliseconds;
            cached_errors += cached.errors;
            hits += cached.hits;
            accesses += cached.hits + cached.misses;
        }

        const double total_cached_ms = training.milliseconds + manager_ms + cached_ms;
        cached_samples.push_back(total_cached_ms);
        cached_hit_percent.push_back(accesses == 0 ? 0.0 : 100.0 * hits / accesses);
        last_baseline_errors = baseline_errors;
        last_cached_errors = training.errors + cached_errors;
        last_promoted = update.promoted_pages;
        last_cache_bytes = update.host_to_device_bytes;
        std::printf("sample,phase=stable,repeat=%d,baseline_%d_pass_ms=%.6f,cache_total_ms=%.6f,training_ms=%.6f,manager_ms=%.6f,cached_%d_pass_ms=%.6f,cache_hit_pct=%.3f,promoted_pages=%zu,transfer_MiB=%.3f,baseline_errors=%llu,cache_errors=%llu\n",
            repeat, options.passes + 1, baseline_ms, total_cached_ms,
            training.milliseconds, manager_ms, options.passes, cached_ms,
            cached_hit_percent.back(), update.promoted_pages,
            update.host_to_device_bytes / (1024.0 * 1024.0),
            static_cast<unsigned long long>(baseline_errors),
            static_cast<unsigned long long>(training.errors + cached_errors));
    }

    std::printf("summary,phase=stable,n=%d,baseline_median_ms=%.6f,learn_prefetch_reuse_median_ms=%.6f,cache_hit_pct_median=%.3f,initial_cache_promotions=%zu,initial_transfer_MiB=%.3f,baseline_errors=%llu,cache_errors=%llu\n",
        options.repeats, median(baseline_samples), median(cached_samples),
        median(cached_hit_percent), last_promoted,
        last_cache_bytes / (1024.0 * 1024.0),
        static_cast<unsigned long long>(last_baseline_errors),
        static_cast<unsigned long long>(last_cached_errors));

    const std::vector<uint32_t> phase_one = make_requests(page_count, 0,
        kHotPages, vectors_per_page, request_count, 0xD001u);
    check(cudaMemcpy(device_requests, phase_one.data(), request_bytes,
        cudaMemcpyHostToDevice), "copy initial hot-set trace");
    AccessStats phase_one_training = run_access(host_device,
        manager.device_frame_pointers(), manager.device_page_to_frame(),
        device_requests, request_count, vectors_per_page, true,
        device_page_counts, device_blocks, host_blocks, &host_page_counts, stream);
    unified_memory::ResidencyUpdate full_cache_update;
    if (!check_manager(manager.update(host_page_counts, max_cache_pages,
        &full_cache_update, &manager_error), manager_error, "load hot set before pressure")) return 1;

    const size_t pressure_bytes = kPressureMiB * 1024u * 1024u;
    void* pressure_buffer = nullptr;
    check(cudaMalloc(&pressure_buffer, pressure_bytes), "allocate bounded VRAM pressure buffer");
    check(cudaMemset(pressure_buffer, 0, pressure_bytes), "touch bounded VRAM pressure buffer");
    size_t free_before_pressure = 0;
    size_t total_device_bytes = 0;
    check(cudaMemGetInfo(&free_before_pressure, &total_device_bytes), "query VRAM under pressure");

    const size_t reduced_capacity = max_cache_pages / 2;
    unified_memory::ResidencyUpdate pressure_update;
    if (!check_manager(manager.update(host_page_counts, reduced_capacity,
        &pressure_update, &manager_error), manager_error, "evict cold pages under VRAM pressure")) return 1;

    const std::vector<uint32_t> phase_two = make_requests(page_count, 64,
        kHotPages, vectors_per_page, request_count, 0xD002u);
    check(cudaMemcpy(device_requests, phase_two.data(), request_bytes,
        cudaMemcpyHostToDevice), "copy shifted hot-set trace");
    AccessStats phase_two_training = run_access(host_device,
        manager.device_frame_pointers(), manager.device_page_to_frame(),
        device_requests, request_count, vectors_per_page, true,
        device_page_counts, device_blocks, host_blocks, &host_page_counts, stream);
    unified_memory::ResidencyUpdate shifted_update;
    if (!check_manager(manager.update(host_page_counts, reduced_capacity,
        &shifted_update, &manager_error), manager_error, "prefetch shifted hot set")) return 1;

    AccessStats shifted_check = run_access(host_device,
        manager.device_frame_pointers(), manager.device_page_to_frame(),
        device_requests, request_count, vectors_per_page, false, nullptr,
        device_blocks, host_blocks, nullptr, stream);
    size_t free_after_pressure = 0;
    check(cudaMemGetInfo(&free_after_pressure, &total_device_bytes), "query VRAM after page eviction");
    std::printf("pressure,allocated_MiB=%zu,free_before_evict_MiB=%zu,free_after_remap_MiB=%zu,cache_frames_before=%zu,cache_frames_after=%zu,evicted_pages=%zu,shifted_promotions=%zu,shifted_evictions=%zu,shifted_transfer_MiB=%.3f,phase_two_cache_hit_pct=%.3f,phase_two_errors=%llu\n",
        kPressureMiB, free_before_pressure / (1024u * 1024u),
        free_after_pressure / (1024u * 1024u), max_cache_pages,
        manager.capacity_pages(), pressure_update.evicted_pages,
        shifted_update.promoted_pages, shifted_update.evicted_pages,
        shifted_update.host_to_device_bytes / (1024.0 * 1024.0),
        100.0 * shifted_check.hits / request_count,
        static_cast<unsigned long long>(phase_one_training.errors +
            phase_two_training.errors + shifted_check.errors));

    check(cudaFree(pressure_buffer), "free bounded VRAM pressure buffer");
    check(cudaFree(device_full_errors), "free full-range validation counter");
    check(cudaFree(device_blocks), "free access statistics");
    check(cudaFree(device_page_counts), "free GPU page counters");
    check(cudaFree(device_requests), "free request list");
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFreeHost(host_cpu), "free pinned host backing");

    const bool valid = last_baseline_errors == 0 && last_cached_errors == 0 &&
        phase_one_training.errors == 0 && phase_two_training.errors == 0 &&
        shifted_check.errors == 0 && pressure_update.evicted_pages > 0 &&
        shifted_update.promoted_pages > 0 && shifted_check.hits > 0;
    return valid ? 0 : 1;
}
