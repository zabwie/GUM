#include <cuda_runtime.h>

#include <algorithm>
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
constexpr int kRepeatsMax = 10;

[[noreturn]] void fail(const char* what, cudaError_t error) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    std::exit(1);
}

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) fail(what, error);
}

__global__ void scan_selected(const float4* input, size_t selected_vectors,
                              size_t stride_vectors, float* block_sums) {
    __shared__ float sums[kThreads];
    const unsigned int lane = threadIdx.x;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + lane;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float sum = 0.0f;

    for (size_t selected = first; selected < selected_vectors; selected += stride) {
        const float4 value = input[selected * stride_vectors];
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

__global__ void gather_selected(const float4* input, size_t selected_vectors,
                                size_t stride_vectors, float4* compact_cache) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t selected = first; selected < selected_vectors; selected += stride) {
        compact_cache[selected] = input[selected * stride_vectors];
    }
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
    if (options.mib > kMaxMiB || options.repeats > kRepeatsMax) {
        std::fprintf(stderr, "Resource cap exceeded: --mib <= %d, --repeats <= %d\n",
                     kMaxMiB, kRepeatsMax);
        std::exit(2);
    }
    return options;
}

struct TimedResult {
    float milliseconds = 0.0f;
    double checksum = 0.0;
    bool valid = false;
};

TimedResult run_scans(const float4* input, size_t selected_vectors,
                      size_t stride_vectors, double expected, int passes,
                      int blocks, cudaStream_t stream, float* block_sums,
                      std::vector<float>& host_sums,
                      cudaEvent_t start, cudaEvent_t stop) {
    check(cudaEventRecord(start, stream), "record path start");
    for (int pass = 0; pass < passes; ++pass) {
        scan_selected<<<blocks, kThreads, 0, stream>>>(input, selected_vectors,
                                                        stride_vectors, block_sums);
        check(cudaGetLastError(), "launch selected scan");
    }
    check(cudaEventRecord(stop, stream), "record path stop");
    check(cudaEventSynchronize(stop), "wait for selected scans");

    TimedResult result;
    check(cudaEventElapsedTime(&result.milliseconds, start, stop), "measure path time");
    check(cudaMemcpy(host_sums.data(), block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost),
          "copy selected-scan checksum");
    for (int block = 0; block < blocks; ++block) result.checksum += host_sums[block];
    result.valid = std::abs(result.checksum - expected) / expected < 1.0e-4;
    return result;
}

void evict_l2(const float4* eviction_buffer, size_t eviction_vectors,
              int blocks, cudaStream_t stream, float* block_sums) {
    scan_selected<<<blocks, kThreads, 0, stream>>>(eviction_buffer,
        eviction_vectors, 1, block_sums);
    check(cudaGetLastError(), "launch L2 eviction scan");
    check(cudaStreamSynchronize(stream), "wait for L2 eviction scan");
}

void print_summary(const char* path, int selected_pct, int passes,
                   const std::vector<float>& samples, size_t useful_bytes_per_pass,
                   size_t explicit_copy_bytes, size_t gathered_payload_bytes = 0) {
    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted.empty() ? 0.0f : sorted[sorted.size() / 2];
    const float minimum = sorted.empty() ? 0.0f : sorted.front();
    const float maximum = sorted.empty() ? 0.0f : sorted.back();
    const size_t useful_total = useful_bytes_per_pass * static_cast<size_t>(passes);
    const size_t transfer_total = explicit_copy_bytes * static_cast<size_t>(passes > 0 ? 1 : 0);
    const double useful_gbps = median > 0.0f ? useful_total / (static_cast<double>(median) * 1.0e6) : 0.0;
    std::printf("summary,%s,selectivity_pct=%d,passes=%d,n=%zu,median_ms=%.6f,min_ms=%.6f,max_ms=%.6f,useful_GB_s=%.3f,explicit_copy_MiB=%.3f,gathered_payload_MiB=%.3f\n",
                path, selected_pct, passes, sorted.size(), median, minimum, maximum,
                useful_gbps, transfer_total / (1024.0 * 1024.0),
                gathered_payload_bytes / (1024.0 * 1024.0));
    for (size_t i = 0; i < samples.size(); ++i) {
        const double gbps = samples[i] > 0.0f ? useful_total / (static_cast<double>(samples[i]) * 1.0e6) : 0.0;
        std::printf("sample,%s,selectivity_pct=%d,passes=%d,%zu,%.6f,useful_GB_s=%.3f\n",
                    path, selected_pct, passes, i + 1, samples[i], gbps);
    }
}

double expected_for_stride(size_t selected_vectors, size_t stride_vectors) {
    double expected = 0.0;
    for (size_t selected = 0; selected < selected_vectors; ++selected) {
        const size_t first_scalar = selected * stride_vectors * 4;
        for (size_t component = 0; component < 4; ++component) {
            expected += 1.0 + static_cast<double>((first_scalar + component) % 13) * 0.001;
        }
    }
    return expected;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const size_t bytes = static_cast<size_t>(options.mib) * 1024 * 1024;
    if (bytes % sizeof(float4) != 0) {
        std::fprintf(stderr, "Buffer size must be divisible by 16 bytes.\n");
        return 2;
    }
    const size_t total_vectors = bytes / sizeof(float4);

    check(cudaSetDevice(0), "select GPU 0");
    int driver_version = 0;
    int runtime_version = 0;
    check(cudaDriverGetVersion(&driver_version), "query driver version");
    check(cudaRuntimeGetVersion(&runtime_version), "query runtime version");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "query GPU properties");
    const int blocks = std::max(1, prop.multiProcessorCount * 8);
    const size_t eviction_bytes = std::max<size_t>(16u * 1024 * 1024,
        static_cast<size_t>(prop.l2CacheSize) * 4);

    std::printf("device,%s,cc=%d.%d,driver=%d,runtime=%d,buffer_MiB=%d,repeats=%d,device_buffers_MiB=%zu,L2_MiB=%zu\n",
                prop.name, prop.major, prop.minor, driver_version, runtime_version,
                options.mib, options.repeats,
                options.mib * 3 + eviction_bytes / (1024 * 1024),
                static_cast<size_t>(prop.l2CacheSize) / (1024 * 1024));

    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    float4* host = nullptr;
    float4* mapped_device = nullptr;
    float* staged_device = nullptr;
    float4* managed = nullptr;
    float4* compact_cache = nullptr;
    float4* eviction_buffer = nullptr;
    float* block_sums = nullptr;
    check(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped),
          "allocate mapped pinned input");
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped_device), host, 0),
          "get mapped input device pointer");
    check(cudaMalloc(reinterpret_cast<void**>(&staged_device), bytes), "allocate staging buffer");
    check(cudaMallocManaged(reinterpret_cast<void**>(&managed), bytes, cudaMemAttachGlobal),
          "allocate managed input");
    check(cudaMalloc(reinterpret_cast<void**>(&compact_cache), bytes), "allocate compact cache buffer");
    check(cudaMalloc(reinterpret_cast<void**>(&eviction_buffer), eviction_bytes),
          "allocate L2 eviction buffer");
    check(cudaMemset(eviction_buffer, 0, eviction_bytes), "initialize L2 eviction buffer");
    check(cudaMalloc(reinterpret_cast<void**>(&block_sums), blocks * sizeof(float)),
          "allocate checksum buffer");

    const size_t elements = bytes / sizeof(float);
    float* host_scalar = reinterpret_cast<float*>(host);
    float* managed_scalar = reinterpret_cast<float*>(managed);
    double full_expected = 0.0;
    for (size_t i = 0; i < elements; ++i) {
        const float value = 1.0f + static_cast<float>(i % 13) * 0.001f;
        host_scalar[i] = value;
        managed_scalar[i] = value;
        full_expected += value;
    }

    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");
    std::vector<float> host_sums(blocks, 0.0f);
    const int strides[] = {1, 4, 10, 100};
    const int percentages[] = {100, 25, 10, 1};
    const int pass_counts[] = {1, 2, 4};
    bool all_valid = true;

    std::printf("result_format,mode,selectivity_pct,passes,trial,elapsed_ms,useful_GB_s,valid\n");
    for (size_t pattern = 0; pattern < sizeof(strides) / sizeof(strides[0]); ++pattern) {
        const size_t stride_vectors = static_cast<size_t>(strides[pattern]);
        const size_t selected_vectors = (total_vectors + stride_vectors - 1) / stride_vectors;
        const size_t useful_bytes = selected_vectors * sizeof(float4);
        const double expected = stride_vectors == 1 ? full_expected : expected_for_stride(selected_vectors, stride_vectors);

        for (int passes : pass_counts) {
            std::vector<float> mapped_samples;
            std::vector<float> staged_samples;
            std::vector<float> managed_samples;
            std::vector<float> cache_samples;

            for (int trial = 0; trial < options.repeats; ++trial) {
                evict_l2(eviction_buffer, eviction_bytes / sizeof(float4), blocks,
                         stream, block_sums);
                const TimedResult result = run_scans(mapped_device, selected_vectors,
                    stride_vectors, expected, passes, blocks, stream, block_sums,
                    host_sums, start, stop);
                mapped_samples.push_back(result.milliseconds);
                all_valid = all_valid && result.valid;
                const double useful_gbps = useful_bytes * static_cast<size_t>(passes) /
                    (static_cast<double>(result.milliseconds) * 1.0e6);
                std::printf("sample,mapped_host,%d,%d,%d,%.6f,%.3f,%s\n", percentages[pattern],
                            passes, trial + 1, result.milliseconds, useful_gbps,
                            result.valid ? "yes" : "no");
            }
            print_summary("mapped_host", percentages[pattern], passes, mapped_samples,
                          useful_bytes, 0);

            for (int trial = 0; trial < options.repeats; ++trial) {
                evict_l2(eviction_buffer, eviction_bytes / sizeof(float4), blocks,
                         stream, block_sums);
                check(cudaEventRecord(start, stream), "record staged path start");
                check(cudaMemcpyAsync(staged_device, host, bytes, cudaMemcpyHostToDevice, stream),
                      "copy full buffer to staging VRAM");
                for (int pass = 0; pass < passes; ++pass) {
                    scan_selected<<<blocks, kThreads, 0, stream>>>(
                        reinterpret_cast<const float4*>(staged_device), selected_vectors,
                        stride_vectors, block_sums);
                    check(cudaGetLastError(), "launch staged selected scan");
                }
                check(cudaEventRecord(stop, stream), "record staged path stop");
                check(cudaEventSynchronize(stop), "wait for staged path");
                TimedResult result;
                check(cudaEventElapsedTime(&result.milliseconds, start, stop),
                      "measure staged path");
                check(cudaMemcpy(host_sums.data(), block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost),
                      "copy staged checksum");
                for (int block = 0; block < blocks; ++block) result.checksum += host_sums[block];
                result.valid = std::abs(result.checksum - expected) / expected < 1.0e-4;
                staged_samples.push_back(result.milliseconds);
                all_valid = all_valid && result.valid;
                const double useful_gbps = useful_bytes * static_cast<size_t>(passes) /
                    (static_cast<double>(result.milliseconds) * 1.0e6);
                std::printf("sample,staged_h2d,%d,%d,%d,%.6f,%.3f,%s\n", percentages[pattern],
                            passes, trial + 1, result.milliseconds, useful_gbps,
                            result.valid ? "yes" : "no");
            }
            print_summary("staged_h2d", percentages[pattern], passes, staged_samples,
                          useful_bytes, bytes);

            for (int trial = 0; trial < options.repeats; ++trial) {
                cudaMemLocation host_location{};
                host_location.type = cudaMemLocationTypeHost;
                cudaMemLocation gpu_location{};
                gpu_location.type = cudaMemLocationTypeDevice;
                gpu_location.id = 0;
                check(cudaMemPrefetchAsync(managed, bytes, host_location, 0, stream),
                      "reset managed input to CPU");
                check(cudaStreamSynchronize(stream), "wait for CPU managed reset");
                evict_l2(eviction_buffer, eviction_bytes / sizeof(float4), blocks,
                         stream, block_sums);
                check(cudaEventRecord(start, stream), "record managed path start");
                check(cudaMemPrefetchAsync(managed, bytes, gpu_location, 0, stream),
                      "prefetch managed input to GPU");
                for (int pass = 0; pass < passes; ++pass) {
                    scan_selected<<<blocks, kThreads, 0, stream>>>(managed, selected_vectors,
                                                                    stride_vectors, block_sums);
                    check(cudaGetLastError(), "launch managed selected scan");
                }
                check(cudaEventRecord(stop, stream), "record managed path stop");
                check(cudaEventSynchronize(stop), "wait for managed path");
                TimedResult result;
                check(cudaEventElapsedTime(&result.milliseconds, start, stop),
                      "measure managed path");
                check(cudaMemcpy(host_sums.data(), block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost),
                      "copy managed checksum");
                for (int block = 0; block < blocks; ++block) result.checksum += host_sums[block];
                result.valid = std::abs(result.checksum - expected) / expected < 1.0e-4;
                managed_samples.push_back(result.milliseconds);
                all_valid = all_valid && result.valid;
                const double useful_gbps = useful_bytes * static_cast<size_t>(passes) /
                    (static_cast<double>(result.milliseconds) * 1.0e6);
                std::printf("sample,managed_prefetch,%d,%d,%d,%.6f,%.3f,%s\n", percentages[pattern],
                            passes, trial + 1, result.milliseconds, useful_gbps,
                            result.valid ? "yes" : "no");
            }
            print_summary("managed_prefetch", percentages[pattern], passes, managed_samples,
                          useful_bytes, bytes);

            for (int trial = 0; trial < options.repeats; ++trial) {
                evict_l2(eviction_buffer, eviction_bytes / sizeof(float4), blocks,
                         stream, block_sums);
                check(cudaEventRecord(start, stream), "record GPU-cache path start");
                gather_selected<<<blocks, kThreads, 0, stream>>>(mapped_device,
                    selected_vectors, stride_vectors, compact_cache);
                check(cudaGetLastError(), "launch GPU-cache gather");
                for (int pass = 0; pass < passes; ++pass) {
                    scan_selected<<<blocks, kThreads, 0, stream>>>(compact_cache,
                        selected_vectors, 1, block_sums);
                    check(cudaGetLastError(), "launch GPU-cache reuse scan");
                }
                check(cudaEventRecord(stop, stream), "record GPU-cache path stop");
                check(cudaEventSynchronize(stop), "wait for GPU-cache path");
                TimedResult result;
                check(cudaEventElapsedTime(&result.milliseconds, start, stop),
                      "measure GPU-cache path");
                check(cudaMemcpy(host_sums.data(), block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost),
                      "copy GPU-cache checksum");
                for (int block = 0; block < blocks; ++block) result.checksum += host_sums[block];
                result.valid = std::abs(result.checksum - expected) / expected < 1.0e-4;
                cache_samples.push_back(result.milliseconds);
                all_valid = all_valid && result.valid;
                const double useful_gbps = useful_bytes * static_cast<size_t>(passes) /
                    (static_cast<double>(result.milliseconds) * 1.0e6);
                std::printf("sample,gpu_side_cache,%d,%d,%d,%.6f,%.3f,%s\n", percentages[pattern],
                            passes, trial + 1, result.milliseconds, useful_gbps,
                            result.valid ? "yes" : "no");
            }
            print_summary("gpu_side_cache", percentages[pattern], passes, cache_samples,
                          useful_bytes, 0, useful_bytes);
        }
    }

    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(stop), "destroy stop event");
    check(cudaFree(block_sums), "free checksum buffer");
    check(cudaFree(compact_cache), "free compact cache buffer");
    check(cudaFree(eviction_buffer), "free L2 eviction buffer");
    check(cudaFree(managed), "free managed buffer");
    check(cudaFree(staged_device), "free staging buffer");
    check(cudaFreeHost(host), "free mapped pinned input");
    check(cudaStreamDestroy(stream), "destroy stream");
    return all_valid ? 0 : 1;
}
