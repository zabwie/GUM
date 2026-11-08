#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr int kMaxMiB = 960;
constexpr int kMaxRepeats = 10;
constexpr int kMaxSteps = 1'000'000;
constexpr int kMaxChains = 4096;
constexpr size_t kEvictionMiB = 16;

[[noreturn]] void fail(const char* operation, cudaError_t error) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    std::exit(1);
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) fail(operation, error);
}

int parse_positive(const char* value, const char* option, int maximum) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > maximum) {
        std::fprintf(stderr, "Invalid value for %s: %s (expected 1..%d)\n",
                     option, value, maximum);
        std::exit(2);
    }
    return static_cast<int>(parsed);
}

struct Options {
    int mib = 32;
    int steps = 32768;
    int chains = 1;
    int repeats = 5;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mib") == 0 && i + 1 < argc) {
            options.mib = parse_positive(argv[++i], "--mib", kMaxMiB);
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            options.steps = parse_positive(argv[++i], "--steps", kMaxSteps);
        } else if (std::strcmp(argv[i], "--chains") == 0 && i + 1 < argc) {
            options.chains = parse_positive(argv[++i], "--chains", kMaxChains);
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            options.repeats = parse_positive(argv[++i], "--repeats", kMaxRepeats);
        } else {
            std::fprintf(stderr, "Usage: %s [--mib 1..960] [--steps 1..1000000] [--chains 1..4096] [--repeats 1..10]\n", argv[0]);
            std::exit(2);
        }
    }
    return options;
}

__global__ void chase_dependent_loads(const volatile uint32_t* next,
                                      const uint32_t* starts, int steps,
                                      uint32_t* results, int chains) {
    const int chain = blockIdx.x * blockDim.x + threadIdx.x;
    if (chain >= chains) return;
    uint32_t node = starts[chain];
    for (int i = 0; i < steps; ++i) node = next[node];
    results[chain] = node;
}

__global__ void evict_l2(const uint32_t* input, size_t count, uint32_t* output) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    uint32_t sum = 0;
    for (size_t i = first; i < count; i += stride) sum += input[i];
    output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

struct Measurement {
    float elapsed_ms = 0.0f;
    bool valid = false;
};

Measurement measure(const volatile uint32_t* device_next,
                    const uint32_t* device_starts,
                    const std::vector<uint32_t>& expected, int steps,
                    uint32_t* device_results, std::vector<uint32_t>& actual,
                    bool include_managed_prefetch, uint32_t* managed_next,
                    size_t managed_bytes,
                    const uint32_t* eviction, size_t eviction_count,
                    uint32_t* eviction_output, int eviction_blocks,
                    cudaEvent_t begin, cudaEvent_t end) {
    evict_l2<<<eviction_blocks, 256>>>(eviction, eviction_count, eviction_output);
    check(cudaGetLastError(), "launch L2 eviction scan");
    check(cudaDeviceSynchronize(), "wait for L2 eviction scan");

    check(cudaEventRecord(begin), "record trial start");
    if (include_managed_prefetch) {
        cudaMemLocation gpu_location{};
        gpu_location.type = cudaMemLocationTypeDevice;
        gpu_location.id = 0;
        check(cudaMemPrefetchAsync(managed_next, managed_bytes, gpu_location, 0),
              "prefetch managed chain to GPU");
    }
    constexpr int threads = 128;
    const int blocks = static_cast<int>((expected.size() + threads - 1) / threads);
    chase_dependent_loads<<<blocks, threads>>>(device_next, device_starts, steps,
                                                device_results,
                                                static_cast<int>(expected.size()));
    check(cudaGetLastError(), "launch dependent-load chain");
    check(cudaEventRecord(end), "record trial end");
    check(cudaEventSynchronize(end), "wait for dependent-load chain");
    Measurement measurement;
    check(cudaEventElapsedTime(&measurement.elapsed_ms, begin, end),
          "read dependent-load elapsed time");
    check(cudaMemcpy(actual.data(), device_results,
                     actual.size() * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost), "copy chain result");
    measurement.valid = actual == expected;
    return measurement;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle]
                             : (values[middle - 1] + values[middle]) / 2.0;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    check(cudaSetDeviceFlags(cudaDeviceMapHost), "enable mapped host memory");

    int device = 0;
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "query CUDA device");
    if (!properties.canMapHostMemory) {
        std::fprintf(stderr, "CUDA device does not support mapped host memory\n");
        return 3;
    }

    const size_t bytes = static_cast<size_t>(options.mib) * 1024 * 1024;
    const size_t nodes = bytes / sizeof(uint32_t);
    if (nodes < 2 || nodes > std::numeric_limits<uint32_t>::max()) {
        std::fprintf(stderr, "Invalid node count for requested MiB size\n");
        return 2;
    }
    if (static_cast<size_t>(options.steps) > nodes / options.chains) {
        std::fprintf(stderr,
                     "Refusing overlapping chains: steps per chain must not exceed nodes per chain segment\n");
        return 2;
    }

    uint32_t* host_next = nullptr;
    check(cudaHostAlloc(reinterpret_cast<void**>(&host_next), bytes, cudaHostAllocMapped),
          "allocate mapped pinned host chain");
    uint32_t* mapped_next = nullptr;
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped_next), host_next, 0),
          "get mapped host device pointer");

    std::vector<uint32_t> permutation(nodes);
    for (size_t i = 0; i < nodes; ++i) permutation[i] = static_cast<uint32_t>(i);
    std::mt19937_64 random(0x6d61707065645f31ULL);
    std::shuffle(permutation.begin(), permutation.end(), random);
    for (size_t i = 0; i < nodes; ++i) {
        host_next[permutation[i]] = permutation[(i + 1) % nodes];
    }
    std::vector<uint32_t> starts(options.chains);
    std::vector<uint32_t> expected(options.chains);
    std::vector<uint32_t> actual(options.chains);
    for (int chain = 0; chain < options.chains; ++chain) {
        const size_t cycle_position = static_cast<size_t>(chain) * nodes / options.chains;
        starts[chain] = permutation[cycle_position];
        uint32_t node = starts[chain];
        for (int i = 0; i < options.steps; ++i) node = host_next[node];
        expected[chain] = node;
    }
    uint32_t* managed_next = nullptr;
    check(cudaMallocManaged(reinterpret_cast<void**>(&managed_next), bytes,
                            cudaMemAttachGlobal),
          "allocate managed pointer chain");
    std::memcpy(managed_next, host_next, bytes);
    permutation.clear();
    permutation.shrink_to_fit();

    uint32_t* device_next = nullptr;
    uint32_t* eviction = nullptr;
    uint32_t* device_starts = nullptr;
    uint32_t* device_results = nullptr;
    uint32_t* eviction_output = nullptr;
    const size_t eviction_bytes = kEvictionMiB * 1024 * 1024;
    const size_t eviction_count = eviction_bytes / sizeof(uint32_t);
    check(cudaMalloc(&device_next, bytes), "allocate VRAM chain");
    check(cudaMalloc(&eviction, eviction_bytes), "allocate L2 eviction buffer");
    check(cudaMalloc(&device_starts, starts.size() * sizeof(uint32_t)),
          "allocate chain starts");
    check(cudaMalloc(&device_results, actual.size() * sizeof(uint32_t)),
          "allocate chain results");
    const int eviction_blocks = 256;
    check(cudaMalloc(&eviction_output, eviction_blocks * 256 * sizeof(uint32_t)),
          "allocate L2 eviction output");
    check(cudaMemcpy(device_next, host_next, bytes, cudaMemcpyHostToDevice),
          "copy randomized chain to VRAM");
    check(cudaMemcpy(device_starts, starts.data(), starts.size() * sizeof(uint32_t),
                     cudaMemcpyHostToDevice), "copy chain starts");
    check(cudaMemset(eviction, 0x5a, eviction_bytes), "initialize L2 eviction buffer");

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    check(cudaEventCreate(&begin), "create start event");
    check(cudaEventCreate(&end), "create stop event");

    std::printf("host_latency_meta,device=%s,bytes=%zu,nodes=%zu,steps_per_chain=%d,chains=%d,total_loads=%d,repeats=%d,host_allocation_mib=%d,vram_allocation_mib=%d,eviction_mib=%zu\n",
                properties.name, bytes, nodes, options.steps, options.chains,
                options.steps * options.chains, options.repeats,
                2 * options.mib, 2 * options.mib, kEvictionMiB);
    std::printf("host_latency_format,summary_or_sample,path,sample,elapsed_ms,ns_per_load,million_loads_per_s,valid\n");

    struct Path {
        const char* name;
        const volatile uint32_t* chain;
        bool managed;
        bool prefetch_to_gpu;
    };
    const Path paths[] = {
        {"mapped_host", mapped_next, false, false},
        {"managed_demand", managed_next, true, false},
        {"managed_prefetch_plus_kernel", managed_next, true, true},
        {"vram_resident", device_next, false, false},
    };
    constexpr int path_count = sizeof(paths) / sizeof(paths[0]);
    std::vector<std::vector<double>> path_times(path_count);

    auto prepare_managed_host = [&]() {
        cudaMemLocation host_location{};
        host_location.type = cudaMemLocationTypeHost;
        check(cudaMemPrefetchAsync(managed_next, bytes, host_location, 0),
              "prefetch managed chain to host before trial");
        check(cudaDeviceSynchronize(), "wait for host prefetch");
    };
    auto run_path = [&](const Path& path, int sample, bool write_sample) {
        if (path.managed) prepare_managed_host();
        const Measurement measurement = measure(
            path.chain, device_starts, expected, options.steps,
            device_results, actual, path.prefetch_to_gpu, managed_next, bytes,
            eviction, eviction_count, eviction_output, eviction_blocks,
            begin, end);
        const double total_loads = static_cast<double>(options.steps) * options.chains;
        const double ns_per_load = measurement.elapsed_ms * 1.0e6 / total_loads;
        if (write_sample) {
            path_times[&path - paths].push_back(ns_per_load);
            std::printf("host_latency_sample,%s,%d,%.6f,%.2f,%.3f,%s\n",
                        path.name, sample, measurement.elapsed_ms, ns_per_load,
                        1000.0 / ns_per_load, measurement.valid ? "yes" : "no");
        }
        if (!measurement.valid) {
            std::fprintf(stderr, "Dependent-load chain result mismatch for %s\n", path.name);
            std::exit(1);
        }
    };

    for (int warmup = 0; warmup < 2; ++warmup) {
        for (const Path& path : paths) run_path(path, 0, false);
    }
    for (int sample = 1; sample <= options.repeats; ++sample) {
        const int start = (sample - 1) % path_count;
        for (int order = 0; order < path_count; ++order)
            run_path(paths[(start + order) % path_count], sample, true);
    }

    auto range = [](const std::vector<double>& values) {
        const auto bounds = std::minmax_element(values.begin(), values.end());
        return std::pair<double, double>{*bounds.first, *bounds.second};
    };
    for (int path = 0; path < path_count; ++path) {
        const double path_median = median(path_times[path]);
        const auto path_range = range(path_times[path]);
        std::printf("host_latency_summary,%s,n=%d,median_ns_per_load=%.2f,million_loads_per_s=%.3f,min=%.2f,max=%.2f\n",
                    paths[path].name, options.repeats, path_median,
                    1000.0 / path_median, path_range.first, path_range.second);
    }

    check(cudaEventDestroy(begin), "destroy start event");
    check(cudaEventDestroy(end), "destroy stop event");
    check(cudaFree(eviction_output), "free L2 eviction output");
    check(cudaFree(device_results), "free chain results");
    check(cudaFree(device_starts), "free chain starts");
    check(cudaFree(eviction), "free L2 eviction buffer");
    check(cudaFree(device_next), "free VRAM chain");
    check(cudaFree(managed_next), "free managed pointer chain");
    check(cudaFreeHost(host_next), "free mapped pinned host chain");
    return 0;
}
