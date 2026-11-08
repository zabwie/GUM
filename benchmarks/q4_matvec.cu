#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int kRows = 4096;
constexpr int kQK = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxMiB = 256;
constexpr int kRepeatsMax = 10;
constexpr int kPasses[] = {1, 2, 4};
constexpr size_t kEvictionMiB = 16;

struct BlockQ4_0 {
    uint16_t d;
    uint8_t qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18, "Q4_0 block must be 18 bytes");

[[noreturn]] void fail(const char* what, cudaError_t error) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    std::exit(1);
}

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) fail(what, error);
}

__global__ void q4_matvec(const BlockQ4_0* weights, const float* input,
                          float* output, int blocks_per_row) {
    __shared__ float warp_sums[kWarps];
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.0f;

    for (int block_index = warp; block_index < blocks_per_row; block_index += kWarps) {
        const BlockQ4_0* block = weights + static_cast<size_t>(row) * blocks_per_row + block_index;
        const uint8_t packed = block->qs[lane & 15];
        const int quant = ((lane < 16) ? (packed & 0x0f) : (packed >> 4)) - 8;
        const float scale = __half2float(__ushort_as_half(block->d));
        sum += (static_cast<float>(quant) * scale) *
               input[static_cast<size_t>(block_index) * kQK + lane];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = lane < kWarps ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) output[row] = sum;
    }
}

__global__ void evict_l2(const float4* input, size_t count, float* sink) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float sum = 0.0f;
    for (size_t index = first; index < count; index += stride) {
        const float4 value = input[index];
        sum += value.x + value.y + value.z + value.w;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = sum;
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

enum class Mode { Mapped, Staged, Managed, Vram };

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::Mapped: return "mapped_host";
        case Mode::Staged: return "staged_h2d";
        case Mode::Managed: return "managed_prefetch";
        case Mode::Vram: return "vram_resident";
    }
    return "unknown";
}

struct TimedResult {
    float milliseconds = 0.0f;
    float max_abs_error = 0.0f;
    double checksum = 0.0;
    bool valid = false;
};

bool validate_output(const float* expected, const std::vector<float>& actual,
                     float& max_abs_error, double& checksum) {
    float max_reference = 0.0f;
    for (int row = 0; row < kRows; ++row) {
        const float reference = expected[row];
        checksum += actual[row];
        max_abs_error = std::max(max_abs_error, std::abs(actual[row] - reference));
        max_reference = std::max(max_reference, std::abs(reference));
    }
    return max_abs_error <= 5.0e-4f * (1.0f + max_reference);
}

void flush_l2(const float4* eviction, size_t eviction_vectors, float* sink,
              cudaStream_t stream, int eviction_blocks) {
    evict_l2<<<eviction_blocks, 256, 0, stream>>>(eviction, eviction_vectors, sink);
    check(cudaGetLastError(), "launch L2 eviction scan");
    check(cudaStreamSynchronize(stream), "wait for L2 eviction scan");
}

TimedResult run_sample(Mode mode, int passes, int sample,
                       const BlockQ4_0* mapped, const BlockQ4_0* staged,
                       BlockQ4_0* managed, int blocks_per_row, size_t weight_bytes,
                       const float* device_inputs, int columns, float* device_output,
                       const float* expected, std::vector<float>& host_output,
                       const float4* eviction, size_t eviction_vectors, float* eviction_sink,
                       int eviction_blocks, cudaStream_t stream,
                       cudaEvent_t start, cudaEvent_t stop) {
    flush_l2(eviction, eviction_vectors, eviction_sink, stream, eviction_blocks);

    if (mode == Mode::Managed) {
        cudaMemLocation host_location{};
        host_location.type = cudaMemLocationTypeHost;
        check(cudaMemPrefetchAsync(managed, weight_bytes, host_location, 0, stream),
              "prefetch managed weights to host before trial");
        check(cudaStreamSynchronize(stream), "wait for host prefetch");
    }

    check(cudaEventRecord(start, stream), "record trial start");
    const BlockQ4_0* weights = mapped;
    if (mode == Mode::Staged) {
        check(cudaMemcpyAsync(const_cast<BlockQ4_0*>(staged), mapped, weight_bytes,
                              cudaMemcpyHostToDevice, stream), "stage Q4_0 weights");
        weights = staged;
    } else if (mode == Mode::Managed) {
        cudaMemLocation gpu_location{};
        gpu_location.type = cudaMemLocationTypeDevice;
        gpu_location.id = 0;
        check(cudaMemPrefetchAsync(managed, weight_bytes, gpu_location, 0, stream),
              "prefetch managed weights to GPU");
        weights = managed;
    } else if (mode == Mode::Vram) {
        weights = staged;
    }

    for (int pass = 0; pass < passes; ++pass) {
        q4_matvec<<<kRows, kThreads, 0, stream>>>(weights,
            device_inputs + static_cast<size_t>(pass) * columns,
            device_output, blocks_per_row);
        check(cudaGetLastError(), "launch Q4_0 matvec");
    }
    check(cudaEventRecord(stop, stream), "record trial stop");
    check(cudaEventSynchronize(stop), "wait for Q4_0 trial");

    TimedResult result;
    check(cudaEventElapsedTime(&result.milliseconds, start, stop), "measure Q4_0 trial");
    check(cudaMemcpy(host_output.data(), device_output, kRows * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy Q4_0 outputs for validation");
    result.valid = validate_output(expected, host_output, result.max_abs_error,
                                   result.checksum);
    if (!result.valid) {
        std::fprintf(stderr,
            "Q4_0 output mismatch mode=%s passes=%d sample=%d max_abs_error=%.8g\n",
            mode_name(mode), passes, sample, result.max_abs_error);
    }
    return result;
}

void print_summary(Mode mode, int passes, const std::vector<float>& samples,
                   size_t weight_bytes) {
    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted[sorted.size() / 2];
    const float minimum = sorted.front();
    const float maximum = sorted.back();
    const double logical_gbps = (weight_bytes * static_cast<double>(passes)) /
                                (static_cast<double>(median) * 1.0e6);
    const double moved_mib = mode == Mode::Mapped || mode == Mode::Vram
        ? 0.0 : static_cast<double>(weight_bytes) / (1024.0 * 1024.0);
    std::printf("summary,%s,passes=%d,n=%zu,median_ms=%.6f,min_ms=%.6f,max_ms=%.6f,logical_weight_GB_s=%.3f,explicit_weight_move_MiB=%.3f\n",
                mode_name(mode), passes, samples.size(), median, minimum, maximum,
                logical_gbps, moved_mib);
    for (size_t i = 0; i < samples.size(); ++i) {
        const double gbps = (weight_bytes * static_cast<double>(passes)) /
                            (static_cast<double>(samples[i]) * 1.0e6);
        std::printf("sample,%s,%d,%zu,%.6f,logical_weight_GB_s=%.3f\n",
                    mode_name(mode), passes, i + 1, samples[i], gbps);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const size_t target_bytes = static_cast<size_t>(options.mib) * 1024 * 1024;
    const size_t blocks_per_row = std::max<size_t>(1,
        (target_bytes / sizeof(BlockQ4_0)) / kRows);
    const size_t block_count = static_cast<size_t>(kRows) * blocks_per_row;
    const size_t weight_bytes = block_count * sizeof(BlockQ4_0);
    const int columns = static_cast<int>(blocks_per_row * kQK);
    const size_t max_input_bytes = static_cast<size_t>(columns) * 4 * sizeof(float);

    check(cudaSetDevice(0), "select GPU 0");
    int driver_version = 0;
    int runtime_version = 0;
    check(cudaDriverGetVersion(&driver_version), "query driver version");
    check(cudaRuntimeGetVersion(&runtime_version), "query runtime version");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "query GPU properties");
    const size_t eviction_bytes = kEvictionMiB * 1024 * 1024;
    const int eviction_blocks = prop.multiProcessorCount * 8;
    const size_t device_buffers = 2 * weight_bytes + eviction_bytes + max_input_bytes +
                                  kRows * sizeof(float) +
                                  static_cast<size_t>(eviction_blocks * 256) * sizeof(float);

    std::printf("device,%s,cc=%d.%d,driver=%d,runtime=%d,target_MiB=%d,weight_bytes=%zu,rows=%d,columns=%d,blocks_per_row=%zu,repeats=%d,device_buffers_MiB=%.3f,L2_MiB=%zu\n",
                prop.name, prop.major, prop.minor, driver_version, runtime_version,
                options.mib, weight_bytes, kRows, columns, blocks_per_row, options.repeats,
                device_buffers / (1024.0 * 1024.0),
                static_cast<size_t>(prop.l2CacheSize) / (1024 * 1024));
    std::printf("format,Q4_0,block_values=32,block_bytes=18,packed_layout=low_nibble_values_0_to_15_high_nibble_values_16_to_31,scale_half_bits=0x2c00\n");

    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    BlockQ4_0* host_weights = nullptr;
    BlockQ4_0* mapped_weights = nullptr;
    BlockQ4_0* managed_weights = nullptr;
    BlockQ4_0* staged_weights = nullptr;
    float* device_inputs = nullptr;
    float* device_output = nullptr;
    float4* eviction = nullptr;
    float* eviction_sink = nullptr;
    check(cudaHostAlloc(reinterpret_cast<void**>(&host_weights), weight_bytes,
                        cudaHostAllocMapped), "allocate mapped pinned Q4_0 weights");
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped_weights), host_weights, 0),
          "get mapped Q4_0 weight pointer");
    check(cudaMallocManaged(reinterpret_cast<void**>(&managed_weights), weight_bytes,
                            cudaMemAttachGlobal), "allocate managed Q4_0 weights");
    check(cudaMalloc(reinterpret_cast<void**>(&staged_weights), weight_bytes),
          "allocate staged Q4_0 weights");
    check(cudaMalloc(reinterpret_cast<void**>(&device_inputs), max_input_bytes),
          "allocate input vectors");
    check(cudaMalloc(reinterpret_cast<void**>(&device_output), kRows * sizeof(float)),
          "allocate output vector");
    check(cudaMalloc(reinterpret_cast<void**>(&eviction), eviction_bytes),
          "allocate L2 eviction buffer");
    check(cudaMalloc(reinterpret_cast<void**>(&eviction_sink),
                     static_cast<size_t>(eviction_blocks * 256) * sizeof(float)),
          "allocate L2 eviction sink");

    constexpr uint16_t kScaleHalf = 0x2c00;  // exactly 1/16
    for (size_t index = 0; index < block_count; ++index) {
        BlockQ4_0& block = host_weights[index];
        block.d = kScaleHalf;
        const size_t row = index / blocks_per_row;
        const size_t within_row = index % blocks_per_row;
        for (int packed_index = 0; packed_index < 16; ++packed_index) {
            const uint8_t low = static_cast<uint8_t>((row * 3 + within_row * 5 + packed_index * 7) & 15);
            const uint8_t high = static_cast<uint8_t>((row * 11 + within_row * 9 + packed_index * 13 + 1) & 15);
            block.qs[packed_index] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    check(cudaMemcpy(managed_weights, host_weights, weight_bytes, cudaMemcpyHostToHost),
          "initialize managed Q4_0 weights");

    std::vector<float> host_inputs(static_cast<size_t>(columns) * 4);
    for (int pass = 0; pass < 4; ++pass) {
        for (int column = 0; column < columns; ++column) {
            const int value = ((column * 5 + pass * 3) % 31) - 15;
            host_inputs[static_cast<size_t>(pass) * columns + column] =
                static_cast<float>(value) * (1.0f / 16.0f);
        }
    }
    check(cudaMemcpy(device_inputs, host_inputs.data(), max_input_bytes,
                     cudaMemcpyHostToDevice), "copy input vectors to GPU");
    check(cudaMemset(eviction, 0, eviction_bytes), "initialize L2 eviction buffer");
    check(cudaMemcpyAsync(staged_weights, host_weights, weight_bytes,
                          cudaMemcpyHostToDevice, stream), "initialize resident weights");
    check(cudaStreamSynchronize(stream), "wait for resident-weight initialization");

    std::vector<float> expected[4];
    for (int input_index = 0; input_index < 4; ++input_index) {
        expected[input_index].resize(kRows, 0.0f);
        for (int row = 0; row < kRows; ++row) {
            double sum = 0.0;
            for (size_t b = 0; b < blocks_per_row; ++b) {
                const BlockQ4_0& block = host_weights[static_cast<size_t>(row) * blocks_per_row + b];
                for (int lane = 0; lane < kQK; ++lane) {
                    const uint8_t packed = block.qs[lane & 15];
                    const int quant = ((lane < 16) ? (packed & 15) : (packed >> 4)) - 8;
                    const float input = host_inputs[static_cast<size_t>(input_index) * columns + b * kQK + lane];
                    sum += static_cast<double>(quant) * (1.0 / 16.0) * input;
                }
            }
            expected[input_index][row] = static_cast<float>(sum);
        }
    }

    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");
    const size_t eviction_vectors = eviction_bytes / sizeof(float4);
    std::vector<float> host_output(kRows);
    std::printf("result_format,summary_or_sample,mode,passes,n_or_trial,median_or_elapsed_ms,logical_weight_GB_s,valid_or_explicit_weight_move_MiB\n");
    const Mode modes[] = {Mode::Mapped, Mode::Staged, Mode::Managed, Mode::Vram};
    bool all_valid = true;

    for (int passes : kPasses) {
        std::vector<float> samples[4];
        for (int sample = -1; sample < options.repeats; ++sample) {
            const int first_mode = sample < 0 ? (passes % 4) : (sample % 4);
            for (int order = 0; order < 4; ++order) {
                const int mode_index = (first_mode + order) % 4;
                const Mode mode = modes[mode_index];
                const TimedResult result = run_sample(mode, passes, sample + 1,
                    mapped_weights, staged_weights, managed_weights,
                    static_cast<int>(blocks_per_row), weight_bytes, device_inputs,
                    columns, device_output, expected[passes - 1].data(), host_output, eviction,
                    eviction_vectors, eviction_sink, eviction_blocks, stream, start, stop);
                all_valid = all_valid && result.valid;
                if (sample >= 0) {
                    samples[mode_index].push_back(result.milliseconds);
                    const double logical_gbps = weight_bytes * static_cast<double>(passes) /
                        (static_cast<double>(result.milliseconds) * 1.0e6);
                    const double moved_mib = mode == Mode::Mapped || mode == Mode::Vram
                        ? 0.0 : static_cast<double>(weight_bytes) / (1024.0 * 1024.0);
                    std::printf("sample,%s,%d,%d,%.6f,%.3f,%s;explicit_weight_move_MiB=%.3f,max_abs_error=%.8g,checksum=%.9g\n",
                        mode_name(mode), passes, sample + 1, result.milliseconds,
                        logical_gbps, result.valid ? "yes" : "no", moved_mib,
                        result.max_abs_error, result.checksum);
                }
            }
        }
        for (int mode_index = 0; mode_index < 4; ++mode_index) {
            print_summary(modes[mode_index], passes, samples[mode_index], weight_bytes);
        }
    }

    std::printf("transfer_only_format,summary_or_sample,path,n_or_trial,elapsed_ms,effective_GB_s,valid\n");
    std::vector<float> transfer_samples[2];
    for (int sample = -1; sample < options.repeats; ++sample) {
        const int first_path = sample < 0 ? 0 : sample % 2;
        for (int order = 0; order < 2; ++order) {
            const int path = (first_path + order) % 2;
            const bool managed_path = path == 1;
            if (managed_path) {
                cudaMemLocation host_location{};
                host_location.type = cudaMemLocationTypeHost;
                check(cudaMemPrefetchAsync(managed_weights, weight_bytes,
                                           host_location, 0, stream),
                      "reset managed weights to host before transfer-only sample");
                check(cudaStreamSynchronize(stream), "wait for managed host reset");
            }
            flush_l2(eviction, eviction_vectors, eviction_sink, stream, eviction_blocks);
            check(cudaEventRecord(start, stream), "record transfer-only start");
            if (managed_path) {
                cudaMemLocation gpu_location{};
                gpu_location.type = cudaMemLocationTypeDevice;
                gpu_location.id = 0;
                check(cudaMemPrefetchAsync(managed_weights, weight_bytes,
                                           gpu_location, 0, stream),
                      "prefetch managed weights for transfer-only sample");
            } else {
                check(cudaMemcpyAsync(staged_weights, host_weights, weight_bytes,
                                      cudaMemcpyHostToDevice, stream),
                      "copy pinned weights for transfer-only sample");
            }
            check(cudaEventRecord(stop, stream), "record transfer-only stop");
            check(cudaEventSynchronize(stop), "wait for transfer-only operation");
            float transfer_ms = 0.0f;
            check(cudaEventElapsedTime(&transfer_ms, start, stop),
                  "measure transfer-only operation");

            const BlockQ4_0* validation_weights = managed_path
                ? managed_weights : staged_weights;
            q4_matvec<<<kRows, kThreads, 0, stream>>>(validation_weights,
                device_inputs + static_cast<size_t>(3) * columns,
                device_output, static_cast<int>(blocks_per_row));
            check(cudaGetLastError(), "launch transfer-only validation matvec");
            check(cudaStreamSynchronize(stream), "wait for transfer-only validation matvec");
            check(cudaMemcpy(host_output.data(), device_output, kRows * sizeof(float),
                             cudaMemcpyDeviceToHost), "copy transfer-only outputs");
            float max_abs_error = 0.0f;
            double checksum = 0.0;
            const bool valid = validate_output(expected[3].data(), host_output,
                                                max_abs_error, checksum);
            all_valid = all_valid && valid;
            if (!valid) {
                std::fprintf(stderr,
                    "transfer-only Q4_0 validation failed path=%s sample=%d max_abs_error=%.8g\n",
                    managed_path ? "managed_prefetch" : "staged_h2d",
                    sample + 1, max_abs_error);
            }
            if (sample >= 0) {
                transfer_samples[path].push_back(transfer_ms);
                const double gbps = weight_bytes / (static_cast<double>(transfer_ms) * 1.0e6);
                std::printf("transfer_sample,%s,%d,%.6f,%.3f,%s;max_abs_error=%.8g,checksum=%.9g\n",
                    managed_path ? "managed_prefetch" : "staged_h2d",
                    sample + 1, transfer_ms, gbps, valid ? "yes" : "no",
                    max_abs_error, checksum);
            }
        }
    }
    for (int path = 0; path < 2; ++path) {
        std::vector<float> sorted = transfer_samples[path];
        std::sort(sorted.begin(), sorted.end());
        const float median = sorted[sorted.size() / 2];
        const double gbps = weight_bytes / (static_cast<double>(median) * 1.0e6);
        std::printf("transfer_summary,%s,n=%zu,median_ms=%.6f,min_ms=%.6f,max_ms=%.6f,effective_GB_s=%.3f\n",
            path == 0 ? "staged_h2d" : "managed_prefetch", sorted.size(), median,
            sorted.front(), sorted.back(), gbps);
    }

    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(stop), "destroy stop event");
    check(cudaFree(eviction_sink), "free L2 eviction sink");
    check(cudaFree(eviction), "free L2 eviction buffer");
    check(cudaFree(device_output), "free output vector");
    check(cudaFree(device_inputs), "free input vectors");
    check(cudaFree(staged_weights), "free staged weights");
    check(cudaFree(managed_weights), "free managed weights");
    check(cudaFreeHost(host_weights), "free mapped pinned weights");
    check(cudaStreamDestroy(stream), "destroy stream");
    return all_valid ? 0 : 1;
}
