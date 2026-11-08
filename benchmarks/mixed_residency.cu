#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <sys/resource.h>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarmups = 2;
constexpr size_t kEvictionMiB = 16;
constexpr size_t kMaxDeviceMiB = 2048;
constexpr size_t kMaxHostMiB = 4096;

[[noreturn]] void fail(const char* what, cudaError_t error) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    std::exit(1);
}

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) fail(what, error);
}

struct Options {
    int device_mib = 1024;
    int host_mib = 2048;
    int repeats = 5;
};

int parse_positive(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || value <= 0 || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", name, text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device-mib") == 0 && i + 1 < argc) {
            options.device_mib = parse_positive(argv[++i], "--device-mib");
        } else if (std::strcmp(argv[i], "--host-mib") == 0 && i + 1 < argc) {
            options.host_mib = parse_positive(argv[++i], "--host-mib");
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            options.repeats = parse_positive(argv[++i], "--repeats");
        } else {
            std::fprintf(stderr,
                "Usage: %s [--device-mib 1..2031] [--host-mib 1..4096] [--repeats 1..10]\n",
                argv[0]);
            std::exit(2);
        }
    }
    if (options.device_mib + static_cast<int>(kEvictionMiB) > static_cast<int>(kMaxDeviceMiB) ||
        options.host_mib > static_cast<int>(kMaxHostMiB) ||
        options.repeats > 10) {
        std::fprintf(stderr,
            "Resource cap exceeded: device buffer plus 16 MiB scratch <= 2048 MiB, host <= 4096 MiB, repeats <= 10\n");
        std::exit(2);
    }
    return options;
}

__device__ __forceinline__ float initial_component(size_t index) {
    return static_cast<float>((index % 255) + 1) * (1.0f / 256.0f);
}

__device__ __forceinline__ float4 initial_value(size_t vector_index) {
    const size_t first = vector_index * 4;
    return make_float4(initial_component(first), initial_component(first + 1),
                       initial_component(first + 2), initial_component(first + 3));
}

__device__ __forceinline__ float4 transform_value(float4 value) {
    value.x = fmaf(value.x, 0.875f, 0.125f);
    value.y = fmaf(value.y, 0.875f, 0.125f);
    value.z = fmaf(value.z, 0.875f, 0.125f);
    value.w = fmaf(value.w, 0.875f, 0.125f);
    return value;
}

__global__ void initialize_regions(float4* device_region, float4* host_region,
                                   size_t device_vectors, size_t host_vectors) {
    const size_t total = device_vectors + host_vectors;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    bool touched_host = false;
    for (size_t index = first; index < total; index += stride) {
        if (index < device_vectors) {
            device_region[index] = initial_value(index);
        } else {
            touched_host = true;
            host_region[index - device_vectors] = initial_value(index - device_vectors);
        }
    }
    if (touched_host) __threadfence_system();
}

// A single launch operates on both address ranges. Threads select the correct
// backing region from the global index; no host-side staging or math occurs.
__global__ void transform_regions(float4* device_region, float4* host_region,
                                  size_t device_vectors, size_t host_vectors) {
    const size_t total = device_vectors + host_vectors;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    bool touched_host = false;
    for (size_t index = first; index < total; index += stride) {
        if (index < device_vectors) {
            device_region[index] = transform_value(device_region[index]);
        } else {
            touched_host = true;
            const size_t host_index = index - device_vectors;
            host_region[host_index] = transform_value(host_region[host_index]);
        }
    }
    if (touched_host) __threadfence_system();
}

__global__ void verify_regions(const float4* device_region, const float4* host_region,
                               size_t device_vectors, size_t host_vectors, int passes,
                               unsigned int* errors) {
    const size_t total = device_vectors + host_vectors;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < total; index += stride) {
        const bool is_device = index < device_vectors;
        const size_t local_index = is_device ? index : index - device_vectors;
        const float4 actual = is_device ? device_region[local_index] : host_region[local_index];
        float4 expected = initial_value(local_index);
        for (int pass = 0; pass < passes; ++pass) expected = transform_value(expected);
        const bool mismatch = fabsf(actual.x - expected.x) > 1.0e-6f ||
                              fabsf(actual.y - expected.y) > 1.0e-6f ||
                              fabsf(actual.z - expected.z) > 1.0e-6f ||
                              fabsf(actual.w - expected.w) > 1.0e-6f;
        if (mismatch) atomicAdd(errors + (is_device ? 0 : 1), 1u);
    }
}

__global__ void evict_l2(const uint4* input, size_t count, unsigned int* sink) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int sum = 0;
    for (size_t index = first; index < count; index += stride) {
        const uint4 value = input[index];
        sum += value.x ^ value.y ^ value.z ^ value.w;
    }
    if (sum != 0) atomicXor(sink, sum);
}

struct RegionCase {
    const char* name;
    size_t device_vectors;
    size_t host_vectors;
};

struct Measurement {
    std::vector<float> milliseconds;
    bool valid = true;
    unsigned int device_mismatches = 0;
    unsigned int host_mismatches = 0;
};

int block_count(const cudaDeviceProp& prop) {
    return std::max(1, std::min(prop.multiProcessorCount * 16, 65535));
}

void initialize(const RegionCase& region, float4* device_region, float4* host_region,
                int blocks, cudaStream_t stream) {
    initialize_regions<<<blocks, kThreads, 0, stream>>>(
        device_region, host_region, region.device_vectors, region.host_vectors);
    check(cudaGetLastError(), "launch GPU-only initialization");
    check(cudaStreamSynchronize(stream), "wait for GPU-only initialization");
}

bool verify(const RegionCase& region, float4* device_region, float4* host_region,
            int passes, int blocks, unsigned int* error_counters,
            unsigned int* device_mismatch_count, unsigned int* host_mismatch_count,
            cudaStream_t stream) {
    check(cudaMemsetAsync(error_counters, 0, 2 * sizeof(unsigned int), stream),
          "clear validation counters");
    verify_regions<<<blocks, kThreads, 0, stream>>>(
        device_region, host_region, region.device_vectors, region.host_vectors,
        passes, error_counters);
    check(cudaGetLastError(), "launch GPU validation");
    unsigned int errors[2] = {};
    check(cudaMemcpyAsync(errors, error_counters, sizeof(errors), cudaMemcpyDeviceToHost, stream),
          "copy two GPU validation counters");
    check(cudaStreamSynchronize(stream), "wait for GPU validation");
    *device_mismatch_count = errors[0];
    *host_mismatch_count = errors[1];
    if (errors[0] || errors[1]) {
        std::fprintf(stderr, "validation mismatch: device=%u host=%u\n", errors[0], errors[1]);
        return false;
    }
    return true;
}

Measurement measure_case(const RegionCase& region, float4* device_region,
                         float4* host_region, uint4* eviction, size_t eviction_vectors,
                         unsigned int* eviction_sink, unsigned int* device_errors,
                         int passes, int repeats, int blocks, cudaStream_t stream) {
    Measurement result;
    result.milliseconds.reserve(repeats);
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");

    for (int sample = -kWarmups; sample < repeats; ++sample) {
        initialize(region, device_region, host_region, blocks, stream);
        evict_l2<<<blocks, kThreads, 0, stream>>>(eviction, eviction_vectors, eviction_sink);
        check(cudaGetLastError(), "launch L2 eviction");
        check(cudaStreamSynchronize(stream), "wait for L2 eviction");

        check(cudaEventRecord(start, stream), "record start event");
        for (int pass = 0; pass < passes; ++pass) {
            transform_regions<<<blocks, kThreads, 0, stream>>>(
                device_region, host_region, region.device_vectors, region.host_vectors);
            check(cudaGetLastError(), "launch mixed-residency transform");
        }
        check(cudaEventRecord(stop, stream), "record stop event");
        check(cudaEventSynchronize(stop), "wait for mixed-residency transform");

        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, start, stop), "measure kernel time");
        unsigned int device_mismatches = 0;
        unsigned int host_mismatches = 0;
        evict_l2<<<blocks, kThreads, 0, stream>>>(eviction, eviction_vectors, eviction_sink);
        check(cudaGetLastError(), "launch L2 eviction before full validation");
        check(cudaStreamSynchronize(stream), "wait for pre-validation L2 eviction");
        result.valid = verify(region, device_region, host_region, passes, blocks,
                              device_errors, &device_mismatches, &host_mismatches, stream) &&
                       result.valid;
        result.device_mismatches = device_mismatches;
        result.host_mismatches = host_mismatches;
        if (sample >= 0) result.milliseconds.push_back(elapsed);
        if (!result.valid) break;
    }

    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(stop), "destroy stop event");
    return result;
}

void print_measurement(const RegionCase& region, const Measurement& result,
                       int passes, int sample_count) {
    std::vector<float> sorted = result.milliseconds;
    std::sort(sorted.begin(), sorted.end());
    const double median_ms = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
    const double min_ms = sorted.empty() ? 0.0 : sorted.front();
    const double max_ms = sorted.empty() ? 0.0 : sorted.back();
    const double device_bytes = static_cast<double>(region.device_vectors) * sizeof(float4);
    const double host_bytes = static_cast<double>(region.host_vectors) * sizeof(float4);
    const double total_logical_gbps = median_ms > 0.0
        ? 2.0 * (device_bytes + host_bytes) * passes / (median_ms * 1.0e6) : 0.0;
    const double device_logical_gbps = median_ms > 0.0
        ? 2.0 * device_bytes * passes / (median_ms * 1.0e6) : 0.0;
    const double host_logical_gbps = median_ms > 0.0
        ? 2.0 * host_bytes * passes / (median_ms * 1.0e6) : 0.0;
    std::printf("summary,%s,passes=%d,n=%zu,median_ms=%.6f,min_ms=%.6f,max_ms=%.6f,logical_total_GB_s=%.3f,logical_device_GB_s=%.3f,logical_host_GB_s=%.3f,device_mismatches=%u,host_mismatches=%u,valid=%s\n",
                region.name, passes, sorted.size(), median_ms, min_ms, max_ms,
                total_logical_gbps, device_logical_gbps, host_logical_gbps,
                result.device_mismatches, result.host_mismatches,
                result.valid && sorted.size() == static_cast<size_t>(sample_count) ? "yes" : "no");
    for (size_t index = 0; index < result.milliseconds.size(); ++index) {
        const double elapsed = result.milliseconds[index];
        const double total = elapsed > 0.0
            ? 2.0 * (device_bytes + host_bytes) * passes / (elapsed * 1.0e6) : 0.0;
        const double host = elapsed > 0.0
            ? 2.0 * host_bytes * passes / (elapsed * 1.0e6) : 0.0;
        std::printf("sample,%s,%d,%zu,%.6f,%.3f,%.3f,device_mismatches=%u,host_mismatches=%u,valid=%s\n",
                    region.name, passes, index + 1, elapsed, total, host,
                    result.device_mismatches, result.host_mismatches,
                    result.valid ? "yes" : "no");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const size_t device_bytes = static_cast<size_t>(options.device_mib) * 1024 * 1024;
    const size_t host_bytes = static_cast<size_t>(options.host_mib) * 1024 * 1024;
    if (device_bytes % sizeof(float4) || host_bytes % sizeof(float4)) {
        std::fprintf(stderr, "Region sizes must be divisible by 16 bytes.\n");
        return 2;
    }
    struct rlimit memlock_limit{};
    if (getrlimit(RLIMIT_MEMLOCK, &memlock_limit) == 0 &&
        memlock_limit.rlim_cur != RLIM_INFINITY && host_bytes > memlock_limit.rlim_cur) {
        std::fprintf(stderr, "Host mapping exceeds this process's RLIMIT_MEMLOCK.\n");
        return 2;
    }

    check(cudaSetDevice(0), "select GPU 0");
    int driver_version = 0;
    int runtime_version = 0;
    cudaDeviceProp prop{};
    check(cudaDriverGetVersion(&driver_version), "query driver level");
    check(cudaRuntimeGetVersion(&runtime_version), "query runtime level");
    check(cudaGetDeviceProperties(&prop, 0), "query device properties");
    if (!prop.canMapHostMemory) {
        std::fprintf(stderr, "Selected GPU does not support mapped pinned host memory.\n");
        return 2;
    }
    const int blocks = block_count(prop);
    const size_t device_vectors = device_bytes / sizeof(float4);
    const size_t host_vectors = host_bytes / sizeof(float4);
    const size_t eviction_bytes = kEvictionMiB * 1024 * 1024;
    const size_t eviction_vectors = eviction_bytes / sizeof(uint4);

    float4* device_region = nullptr;
    float4* host_region = nullptr;
    float4* mapped_host_region = nullptr;
    uint4* eviction = nullptr;
    unsigned int* eviction_sink = nullptr;
    unsigned int* device_errors = nullptr;
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check(cudaMalloc(reinterpret_cast<void**>(&device_region), device_bytes),
          "allocate device-resident region");
    check(cudaHostAlloc(reinterpret_cast<void**>(&host_region), host_bytes, cudaHostAllocMapped),
          "allocate mapped pinned host region");
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped_host_region), host_region, 0),
          "get GPU pointer for pinned host region");
    check(cudaMalloc(reinterpret_cast<void**>(&eviction), eviction_bytes), "allocate L2 eviction buffer");
    check(cudaMalloc(reinterpret_cast<void**>(&eviction_sink), sizeof(unsigned int)),
          "allocate L2 eviction sink");
    check(cudaMalloc(reinterpret_cast<void**>(&device_errors), 2 * sizeof(unsigned int)),
          "allocate GPU validation counters");
    check(cudaMemset(eviction, 0x5a, eviction_bytes), "initialize L2 eviction buffer");
    check(cudaMemset(eviction_sink, 0, sizeof(unsigned int)), "initialize L2 eviction sink");

    const double mib = 1024.0 * 1024.0;
    const double actual_device_mib =
        (static_cast<double>(device_bytes + eviction_bytes) + 3.0 * sizeof(unsigned int)) / mib;
    struct rlimit locked{};
    getrlimit(RLIMIT_MEMLOCK, &locked);
    const double memlock_mib = locked.rlim_cur == RLIM_INFINITY
        ? -1.0 : static_cast<double>(locked.rlim_cur) / mib;
    std::printf("device,%s,cc=%d.%d,SMs=%d,VRAM_MiB=%zu,driver=%d,runtime=%d,canMapHostMemory=%d,unifiedAddressing=%d\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / (1024 * 1024), driver_version, runtime_version,
                prop.canMapHostMemory, prop.unifiedAddressing);
    std::printf("regions,device_MiB=%d,host_pinned_MiB=%d,total_working_set_MiB=%d,device_allocation_plan_MiB=%.3f,host_memlock_limit_MiB=%.0f\n",
                options.device_mib, options.host_mib,
                options.device_mib + options.host_mib, actual_device_mib, memlock_mib);
    std::printf("addressing,device_ptr=%p,host_cpu_ptr=%p,host_gpu_ptr=%p,host_alias_equal=%d\n",
                static_cast<void*>(device_region), static_cast<void*>(host_region),
                static_cast<void*>(mapped_host_region), host_region == mapped_host_region);
    std::printf("execution,cpu_dataset_math=none,initializer=gpu_kernel,transform=gpu_kernel,validator=gpu_kernel,cpu_reads_only_two_error_counters\n");
    std::printf("result_format,sample,mode,passes,run,elapsed_ms,logical_total_GB_s,logical_host_GB_s,device_mismatches,host_mismatches,valid\n");

    const RegionCase cases[] = {
        {"device_only", device_vectors, 0},
        {"mapped_host_only", 0, host_vectors},
        {"mixed_single_kernel", device_vectors, host_vectors},
    };
    bool all_valid = true;
    for (const RegionCase& region : cases) {
        for (int passes : {1, 4}) {
            const Measurement result = measure_case(
                region, device_region, mapped_host_region, eviction, eviction_vectors,
                eviction_sink, device_errors, passes, options.repeats, blocks, stream);
            print_measurement(region, result, passes, options.repeats);
            all_valid = all_valid && result.valid &&
                        result.milliseconds.size() == static_cast<size_t>(options.repeats);
        }
    }

    check(cudaFree(device_errors), "free validation counters");
    check(cudaFree(eviction_sink), "free L2 eviction sink");
    check(cudaFree(eviction), "free L2 eviction buffer");
    check(cudaFreeHost(host_region), "free mapped pinned host region");
    check(cudaFree(device_region), "free device-resident region");
    check(cudaStreamDestroy(stream), "destroy stream");
    return all_valid ? 0 : 1;
}
