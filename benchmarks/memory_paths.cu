#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxMiB = 256;
constexpr int kWarmups = 3;

[[noreturn]] void fail(const char* what, cudaError_t error) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    std::exit(1);
}

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) fail(what, error);
}

__global__ void scan(const float4* input, size_t vector_count, float* block_sums) {
    __shared__ float sums[kThreads];
    const unsigned int lane = threadIdx.x;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + lane;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float sum = 0.0f;

    for (size_t index = first; index < vector_count; index += stride) {
        const float4 value = input[index];
        sum += value.x + value.y + value.z + value.w;
    }

    sums[lane] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        __syncthreads();
    }
    if (lane == 0) block_sums[blockIdx.x] = sums[0];
}

struct Options {
    int mib = 64;
    int repeats = 5;
    bool capabilities_only = false;
};

int parse_positive(const char* value, const char* option) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", option, value);
        std::exit(2);
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--capabilities-only") == 0) {
            options.capabilities_only = true;
        } else if (std::strcmp(argv[i], "--mib") == 0 && i + 1 < argc) {
            options.mib = parse_positive(argv[++i], "--mib");
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            options.repeats = parse_positive(argv[++i], "--repeats");
        } else {
            std::fprintf(stderr, "Usage: %s [--capabilities-only] [--mib 1..256] [--repeats 1..10]\n", argv[0]);
            std::exit(2);
        }
    }
    if (options.mib > kMaxMiB || options.repeats > 10) {
        std::fprintf(stderr, "Resource cap exceeded: --mib <= %d, --repeats <= 10\n", kMaxMiB);
        std::exit(2);
    }
    return options;
}

int attribute(cudaDeviceAttr attr) {
    int value = -1;
    const cudaError_t result = cudaDeviceGetAttribute(&value, attr, 0);
    return result == cudaSuccess ? value : -1;
}

struct Measurement {
    std::vector<float> milliseconds;
    double checksum = 0.0;
    bool valid = false;
};

Measurement measure(const char* path, const float4* input, size_t vector_count,
                    float* device_input, bool stage, double expected,
                    size_t bytes, int blocks, int repeats,
                    cudaStream_t stream, float* block_sums,
                    std::vector<float>& host_sums) {
    Measurement result;
    result.milliseconds.reserve(repeats);
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int sample = -kWarmups; sample < repeats; ++sample) {
        check(cudaEventRecord(start, stream), "record start");
        const float4* source = input;
        if (stage) {
            check(cudaMemcpyAsync(device_input, input, bytes, cudaMemcpyHostToDevice, stream),
                  "stage pinned host buffer");
            source = reinterpret_cast<const float4*>(device_input);
        }
        scan<<<blocks, kThreads, 0, stream>>>(source, vector_count, block_sums);
        check(cudaGetLastError(), "launch scan kernel");
        check(cudaEventRecord(stop, stream), "record stop");
        check(cudaEventSynchronize(stop), "wait for scan");

        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, start, stop), "measure elapsed time");
        if (sample >= 0) result.milliseconds.push_back(elapsed);

        check(cudaMemcpyAsync(host_sums.data(), block_sums,
                              blocks * sizeof(float), cudaMemcpyDeviceToHost, stream),
              "copy checksum partials");
        check(cudaStreamSynchronize(stream), "wait for checksum");
        double checksum = 0.0;
        for (int block = 0; block < blocks; ++block) checksum += host_sums[block];
        result.checksum = checksum;
        const double relative_error = std::abs(checksum - expected) / expected;
        result.valid = relative_error < 1.0e-4;
        if (!result.valid) {
            std::fprintf(stderr, "%s checksum mismatch: got %.9g expected %.9g (relative error %.3g)\n",
                         path, checksum, expected, relative_error);
            break;
        }
    }

    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(stop), "destroy stop event");
    return result;
}

Measurement measure_copy_only(const float4* host_input, size_t vector_count,
                              float* device_input, double expected,
                              size_t bytes, int blocks, int repeats,
                              cudaStream_t stream, float* block_sums,
                              std::vector<float>& host_sums) {
    Measurement result;
    result.milliseconds.reserve(repeats);
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "cudaEventCreate(copy start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(copy stop)");

    for (int sample = -kWarmups; sample < repeats; ++sample) {
        check(cudaEventRecord(start, stream), "record copy start");
        check(cudaMemcpyAsync(device_input, host_input, bytes, cudaMemcpyHostToDevice, stream),
              "copy pinned host buffer to VRAM");
        check(cudaEventRecord(stop, stream), "record copy stop");
        check(cudaEventSynchronize(stop), "wait for H2D copy");

        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, start, stop), "measure H2D copy time");
        if (sample >= 0) result.milliseconds.push_back(elapsed);

        scan<<<blocks, kThreads, 0, stream>>>(reinterpret_cast<const float4*>(device_input),
                                               vector_count, block_sums);
        check(cudaGetLastError(), "launch copy validation scan");
        check(cudaMemcpyAsync(host_sums.data(), block_sums,
                              blocks * sizeof(float), cudaMemcpyDeviceToHost, stream),
              "copy validation partials");
        check(cudaStreamSynchronize(stream), "wait for copy validation");
        double checksum = 0.0;
        for (int block = 0; block < blocks; ++block) checksum += host_sums[block];
        result.checksum = checksum;
        const double relative_error = std::abs(checksum - expected) / expected;
        result.valid = relative_error < 1.0e-4;
        if (!result.valid) {
            std::fprintf(stderr, "H2D copy checksum mismatch: got %.9g expected %.9g (relative error %.3g)\n",
                         checksum, expected, relative_error);
            break;
        }
    }

    check(cudaEventDestroy(start), "destroy copy start event");
    check(cudaEventDestroy(stop), "destroy copy stop event");
    return result;
}

void print_results(const char* path, const Measurement& m, size_t bytes) {
    std::vector<float> sorted = m.milliseconds;
    std::sort(sorted.begin(), sorted.end());
    const double median_ms = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
    const double min_ms = sorted.empty() ? 0.0 : sorted.front();
    const double effective_gbps = median_ms > 0.0 ? bytes / (median_ms * 1.0e6) : 0.0;
    std::printf("summary,%s,n=%zu,median_ms=%.6f,min_ms=%.6f,effective_input_GB_s=%.3f,checksum=%.9g,valid=%s\n",
                path, sorted.size(), median_ms, min_ms, effective_gbps, m.checksum,
                m.valid && !sorted.empty() ? "yes" : "no");
    for (size_t i = 0; i < m.milliseconds.size(); ++i) {
        const double gbps = bytes / (static_cast<double>(m.milliseconds[i]) * 1.0e6);
        std::printf("sample,%s,%zu,%.6f,%.3f\n", path, i + 1, m.milliseconds[i], gbps);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const size_t bytes = static_cast<size_t>(options.mib) * 1024 * 1024;
    if (bytes % sizeof(float4) != 0) {
        std::fprintf(stderr, "Buffer size must be divisible by 16 bytes.\n");
        return 2;
    }
    const size_t vector_count = bytes / sizeof(float4);

    check(cudaSetDevice(0), "select GPU 0");
    int driver_version = 0;
    int runtime_version = 0;
    check(cudaDriverGetVersion(&driver_version), "query driver API level");
    check(cudaRuntimeGetVersion(&runtime_version), "query CUDA runtime level");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "query GPU properties");
    const int blocks = std::max(1, std::min(prop.multiProcessorCount * 8, 65535));

    std::printf("device,%s,cc=%d.%d,SMs=%d,VRAM_MiB=%zu,driver=%d,runtime=%d,buffer_MiB=%d,repeats=%d\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / (1024 * 1024), driver_version, runtime_version,
                options.mib, options.repeats);
    std::printf("capability,canMapHostMemory,%d\n", prop.canMapHostMemory);
    std::printf("capability,unifiedAddressing,%d\n", prop.unifiedAddressing);
    std::printf("capability,concurrentManagedAccess,%d\n", attribute(cudaDevAttrConcurrentManagedAccess));
    std::printf("capability,pageableMemoryAccess,%d\n", attribute(cudaDevAttrPageableMemoryAccess));
    std::printf("capability,pageableMemoryAccessUsesHostPageTables,%d\n",
                attribute(cudaDevAttrPageableMemoryAccessUsesHostPageTables));
    std::printf("capability,hostNativeAtomicSupported,%d\n", attribute(cudaDevAttrHostNativeAtomicSupported));

    if (options.capabilities_only) return 0;

    float4* host = nullptr;
    float* device = nullptr;
    float* block_sums = nullptr;
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped),
          "allocate mapped pinned host memory");
    float4* mapped_device = nullptr;
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped_device), host, 0),
          "get mapped device pointer");
    check(cudaMalloc(reinterpret_cast<void**>(&device), bytes), "allocate VRAM buffer");
    check(cudaMalloc(reinterpret_cast<void**>(&block_sums), blocks * sizeof(float)),
          "allocate checksum buffer");

    double expected = 0.0;
    float* scalar = reinterpret_cast<float*>(host);
    const size_t elements = bytes / sizeof(float);
    for (size_t i = 0; i < elements; ++i) {
        const float value = 1.0f + static_cast<float>(i % 13) * 0.001f;
        scalar[i] = value;
        expected += value;
    }
    check(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "initialize VRAM reference");
    std::vector<float> host_sums(blocks, 0.0f);

    std::printf("result_format,sample,path,run,elapsed_ms,effective_input_GB_s\n");
    const Measurement mapped = measure("mapped_host", mapped_device, vector_count,
                                        device, false, expected, bytes, blocks,
                                        options.repeats, stream, block_sums, host_sums);
    const Measurement copy_only = measure_copy_only(reinterpret_cast<const float4*>(host),
                                                     vector_count, device, expected, bytes,
                                                     blocks, options.repeats, stream,
                                                     block_sums, host_sums);
    const Measurement staged = measure("staged_h2d_plus_scan", reinterpret_cast<const float4*>(host), vector_count,
                                       device, true, expected, bytes, blocks,
                                       options.repeats, stream, block_sums, host_sums);
    const Measurement local = measure("vram_resident", reinterpret_cast<const float4*>(device),
                                      vector_count, device, false, expected, bytes, blocks,
                                      options.repeats, stream, block_sums, host_sums);
    print_results("mapped_host", mapped, bytes);
    print_results("h2d_copy_only", copy_only, bytes);
    print_results("staged_h2d_plus_scan", staged, bytes);
    print_results("vram_resident", local, bytes);

    check(cudaFree(block_sums), "free checksum buffer");
    check(cudaFree(device), "free VRAM buffer");
    check(cudaFreeHost(host), "free mapped pinned memory");
    check(cudaStreamDestroy(stream), "destroy stream");
    if (!mapped.valid || !copy_only.valid || !staged.valid || !local.valid) return 1;
    return 0;
}
