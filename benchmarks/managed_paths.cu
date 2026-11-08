#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxMiB = 256;

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
        if (std::strcmp(argv[i], "--mib") == 0 && i + 1 < argc) {
            options.mib = parse_positive(argv[++i], "--mib");
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            options.repeats = parse_positive(argv[++i], "--repeats");
        } else {
            std::fprintf(stderr, "Usage: %s [--mib 1..256] [--repeats 1..10]\n", argv[0]);
            std::exit(2);
        }
    }
    if (options.mib > kMaxMiB || options.repeats > 10) {
        std::fprintf(stderr, "Resource cap exceeded: --mib <= %d, --repeats <= 10\n", kMaxMiB);
        std::exit(2);
    }
    return options;
}

struct ScanResult {
    float milliseconds = 0.0f;
    double checksum = 0.0;
    bool valid = false;
};

ScanResult run_scan(const float4* input, size_t vector_count, size_t bytes,
                    double expected, int blocks, cudaStream_t stream,
                    float* block_sums, std::vector<float>& host_sums,
                    cudaEvent_t start, cudaEvent_t stop) {
    check(cudaEventRecord(start, stream), "record scan start");
    scan<<<blocks, kThreads, 0, stream>>>(input, vector_count, block_sums);
    check(cudaGetLastError(), "launch scan kernel");
    check(cudaEventRecord(stop, stream), "record scan stop");
    check(cudaEventSynchronize(stop), "wait for scan");

    ScanResult result;
    check(cudaEventElapsedTime(&result.milliseconds, start, stop), "measure scan time");
    check(cudaMemcpy(host_sums.data(), block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost),
          "copy checksum partials");
    for (int block = 0; block < blocks; ++block) result.checksum += host_sums[block];
    const double error = std::abs(result.checksum - expected);
    result.valid = expected == 0.0 ? error < 1.0e-4 : error / expected < 1.0e-4;
    return result;
}

void print_summary(const char* path, const std::vector<float>& samples, size_t bytes) {
    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted.empty() ? 0.0f : sorted[sorted.size() / 2];
    const float minimum = sorted.empty() ? 0.0f : sorted.front();
    const float maximum = sorted.empty() ? 0.0f : sorted.back();
    const double effective_gbps = median > 0.0f ? bytes / (static_cast<double>(median) * 1.0e6) : 0.0;
    std::printf("summary,%s,n=%zu,median_ms=%.6f,min_ms=%.6f,max_ms=%.6f,effective_input_GB_s=%.3f\n",
                path, sorted.size(), median, minimum, maximum, effective_gbps);
    for (size_t i = 0; i < samples.size(); ++i) {
        const double gbps = bytes / (static_cast<double>(samples[i]) * 1.0e6);
        std::printf("sample,%s,%zu,%.6f,%.3f\n", path, i + 1, samples[i], gbps);
    }
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop, const char* what) {
    float value = 0.0f;
    check(cudaEventElapsedTime(&value, start, stop), what);
    return value;
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
    check(cudaDriverGetVersion(&driver_version), "query driver version");
    check(cudaRuntimeGetVersion(&runtime_version), "query runtime version");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "query GPU properties");
    const int blocks = std::max(1, prop.multiProcessorCount * 8);

    std::printf("device,%s,cc=%d.%d,SMs=%d,driver=%d,runtime=%d,managed_buffer_MiB=%d,repeats=%d\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                driver_version, runtime_version, options.mib, options.repeats);

    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    float* block_sums = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&block_sums), blocks * sizeof(float)),
          "allocate checksum buffer");
    std::vector<float> host_sums(blocks, 0.0f);

    cudaEvent_t start{}, stop{}, whole_start{}, prefetch_start{}, prefetch_stop{};
    cudaEvent_t scan_start{}, scan_stop{}, cpu_start{}, cpu_stop{};
    check(cudaEventCreate(&start), "create scan start event");
    check(cudaEventCreate(&stop), "create scan stop event");
    check(cudaEventCreate(&whole_start), "create whole-path start event");
    check(cudaEventCreate(&prefetch_start), "create prefetch start event");
    check(cudaEventCreate(&prefetch_stop), "create prefetch stop event");
    check(cudaEventCreate(&scan_start), "create staged scan start event");
    check(cudaEventCreate(&scan_stop), "create staged scan stop event");
    check(cudaEventCreate(&cpu_start), "create host prefetch start event");
    check(cudaEventCreate(&cpu_stop), "create host prefetch stop event");

    // Warm the CUDA context and kernel on a separate VRAM allocation. The managed
    // benchmark buffer remains untouched by the GPU until the first-touch sample.
    float4* warm_buffer = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&warm_buffer), 1u << 20), "allocate warm-up buffer");
    check(cudaMemset(warm_buffer, 0, 1u << 20), "initialize warm-up buffer");
    const ScanResult warm = run_scan(warm_buffer, (1u << 20) / sizeof(float4), 1u << 20,
                                     0.0, blocks, stream, block_sums, host_sums, start, stop);
    if (!warm.valid) {
        std::fprintf(stderr, "Warm-up buffer checksum was not zero (got %.9g).\n", warm.checksum);
        return 1;
    }
    check(cudaFree(warm_buffer), "free warm-up buffer");

    float4* managed = nullptr;
    check(cudaMallocManaged(reinterpret_cast<void**>(&managed), bytes, cudaMemAttachGlobal),
          "allocate managed buffer");
    double expected = 0.0;
    const size_t elements = bytes / sizeof(float);
    auto host_init_start = std::chrono::steady_clock::now();
    float* scalar = reinterpret_cast<float*>(managed);
    for (size_t i = 0; i < elements; ++i) {
        const float value = 1.0f + static_cast<float>(i % 13) * 0.001f;
        scalar[i] = value;
        expected += value;
    }
    const double host_init_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - host_init_start).count();
    std::printf("host_init,CPU_first_touch,%.3f_ms,expected_checksum=%.9g\n", host_init_ms, expected);

    // First GPU access after CPU initialization is isolated from explicit prefetch.
    const ScanResult first = run_scan(managed, vector_count, bytes, expected, blocks,
                                      stream, block_sums, host_sums, start, stop);
    std::printf("first_gpu_access,cpu_initialized_no_prefetch,%.6f_ms,checksum=%.9g,valid=%s\n",
                first.milliseconds, first.checksum, first.valid ? "yes" : "no");
    std::vector<float> cold_samples{first.milliseconds};
    bool all_valid = first.valid;
    for (int i = 1; i < options.repeats; ++i) {
        float4* cold_buffer = nullptr;
        check(cudaMallocManaged(reinterpret_cast<void**>(&cold_buffer), bytes, cudaMemAttachGlobal),
              "allocate independent cold managed buffer");
        double cold_expected = 0.0;
        float* cold_scalar = reinterpret_cast<float*>(cold_buffer);
        for (size_t element = 0; element < elements; ++element) {
            const float value = 1.0f + static_cast<float>(element % 13) * 0.001f;
            cold_scalar[element] = value;
            cold_expected += value;
        }
        const ScanResult cold = run_scan(cold_buffer, vector_count, bytes, cold_expected, blocks,
                                         stream, block_sums, host_sums, start, stop);
        cold_samples.push_back(cold.milliseconds);
        all_valid = all_valid && cold.valid;
        std::printf("cold_trial,%d,cpu_initialized_no_prefetch,%.6f_ms,checksum=%.9g,valid=%s\n",
                    i + 1, cold.milliseconds, cold.checksum, cold.valid ? "yes" : "no");
        check(cudaFree(cold_buffer), "free independent cold managed buffer");
    }
    print_summary("cpu_initialized_no_prefetch_first_gpu_scan", cold_samples, bytes);

    std::vector<float> repeat_samples;
    for (int i = 0; i < options.repeats; ++i) {
        const ScanResult sample = run_scan(managed, vector_count, bytes, expected, blocks,
                                           stream, block_sums, host_sums, start, stop);
        repeat_samples.push_back(sample.milliseconds);
        all_valid = all_valid && sample.valid;
        std::printf("checksum,repeat_after_first_gpu_access,%d,%.9g,%s\n",
                    i + 1, sample.checksum, sample.valid ? "valid" : "INVALID");
    }
    print_summary("repeat_after_first_gpu_access", repeat_samples, bytes);

    cudaMemLocation host_location{};
    host_location.type = cudaMemLocationTypeHost;
    cudaMemLocation gpu_location{};
    gpu_location.type = cudaMemLocationTypeDevice;
    gpu_location.id = 0;
    std::vector<float> host_prefetch_samples;
    std::vector<float> gpu_prefetch_samples;
    std::vector<float> prefetched_scan_samples;
    std::vector<float> prefetched_total_samples;

    for (int i = 0; i < options.repeats; ++i) {
        // Return the region to host memory before each device-prefetch trial so
        // each measured H2D prefetch starts from the same requested location.
        check(cudaEventRecord(cpu_start, stream), "record CPU prefetch start");
        check(cudaMemPrefetchAsync(managed, bytes, host_location, 0, stream),
              "prefetch managed buffer to CPU");
        check(cudaEventRecord(cpu_stop, stream), "record CPU prefetch stop");
        check(cudaEventSynchronize(cpu_stop), "wait for CPU prefetch");
        host_prefetch_samples.push_back(elapsed_ms(cpu_start, cpu_stop, "measure CPU prefetch"));

        check(cudaEventRecord(whole_start, stream), "record prefetch-plus-scan start");
        check(cudaEventRecord(prefetch_start, stream), "record GPU prefetch start");
        check(cudaMemPrefetchAsync(managed, bytes, gpu_location, 0, stream),
              "prefetch managed buffer to GPU");
        check(cudaEventRecord(prefetch_stop, stream), "record GPU prefetch stop");
        check(cudaEventRecord(scan_start, stream), "record prefetched scan start");
        scan<<<blocks, kThreads, 0, stream>>>(managed, vector_count, block_sums);
        check(cudaGetLastError(), "launch prefetched scan");
        check(cudaEventRecord(scan_stop, stream), "record prefetched scan stop");
        check(cudaEventRecord(stop, stream), "record prefetch-plus-scan stop");
        check(cudaEventSynchronize(stop), "wait for prefetch and scan");

        const float prefetch_ms = elapsed_ms(prefetch_start, prefetch_stop, "measure GPU prefetch");
        const float scan_ms = elapsed_ms(scan_start, scan_stop, "measure scan after prefetch");
        const float total_ms = elapsed_ms(whole_start, stop, "measure prefetch plus scan");
        gpu_prefetch_samples.push_back(prefetch_ms);
        prefetched_scan_samples.push_back(scan_ms);
        prefetched_total_samples.push_back(total_ms);

        check(cudaMemcpy(host_sums.data(), block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost),
              "copy prefetched checksum partials");
        double checksum = 0.0;
        for (int block = 0; block < blocks; ++block) checksum += host_sums[block];
        const bool valid = std::abs(checksum - expected) / expected < 1.0e-4;
        all_valid = all_valid && valid;
        std::printf("prefetch_trial,%d,cpu_prefetch_ms=%.6f,gpu_prefetch_ms=%.6f,scan_ms=%.6f,total_ms=%.6f,checksum=%.9g,valid=%s\n",
                    i + 1, host_prefetch_samples.back(), prefetch_ms, scan_ms, total_ms,
                    checksum, valid ? "yes" : "no");
    }

    print_summary("prefetch_to_cpu_reset", host_prefetch_samples, bytes);
    print_summary("prefetch_to_gpu", gpu_prefetch_samples, bytes);
    print_summary("scan_after_gpu_prefetch", prefetched_scan_samples, bytes);
    print_summary("gpu_prefetch_plus_scan", prefetched_total_samples, bytes);

    check(cudaFree(managed), "free managed buffer");
    check(cudaFree(block_sums), "free checksum buffer");
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaEventDestroy(start), "destroy scan start event");
    check(cudaEventDestroy(stop), "destroy scan stop event");
    check(cudaEventDestroy(whole_start), "destroy whole-path start event");
    check(cudaEventDestroy(prefetch_start), "destroy prefetch start event");
    check(cudaEventDestroy(prefetch_stop), "destroy prefetch stop event");
    check(cudaEventDestroy(scan_start), "destroy staged scan start event");
    check(cudaEventDestroy(scan_stop), "destroy staged scan stop event");
    check(cudaEventDestroy(cpu_start), "destroy host prefetch start event");
    check(cudaEventDestroy(cpu_stop), "destroy host prefetch stop event");
    return all_valid ? 0 : 1;
}
