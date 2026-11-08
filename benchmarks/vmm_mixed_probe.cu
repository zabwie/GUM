#include "../src/unified_range.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Options {
    int device_mib = 64;
    int host_mib = 128;
};

int parse_positive(const char* text, const char* option) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || value <= 0 || value > 4096) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", option, text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--device-mib") == 0 && index + 1 < argc) {
            options.device_mib = parse_positive(argv[++index], "--device-mib");
        } else if (std::strcmp(argv[index], "--host-mib") == 0 && index + 1 < argc) {
            options.host_mib = parse_positive(argv[++index], "--host-mib");
        } else {
            std::fprintf(stderr, "Usage: %s [--device-mib MiB] [--host-mib MiB]\n", argv[0]);
            std::exit(2);
        }
    }
    if (options.device_mib + 17 > 2048 || options.host_mib > 4096) {
        std::fprintf(stderr, "Project memory caps exceeded (2 GiB device, 4 GiB host).\n");
        std::exit(2);
    }
    return options;
}

__device__ float initial_component(size_t index) {
    return static_cast<float>((index % 251) + 1) * (1.0f / 256.0f);
}

__device__ float4 initial_value(size_t index) {
    const size_t first = index * 4;
    return make_float4(initial_component(first), initial_component(first + 1),
                       initial_component(first + 2), initial_component(first + 3));
}

__device__ float4 transform_value(float4 value) {
    value.x = fmaf(value.x, 0.875f, 0.125f);
    value.y = fmaf(value.y, 0.875f, 0.125f);
    value.z = fmaf(value.z, 0.875f, 0.125f);
    value.w = fmaf(value.w, 0.875f, 0.125f);
    return value;
}

__global__ void initialize(float4* range, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t item = index; item < count; item += stride) range[item] = initial_value(item);
}

__global__ void transform(float4* range, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t item = index; item < count; item += stride) {
        range[item] = transform_value(range[item]);
    }
}

__global__ void validate(const float4* range, size_t count, size_t device_count,
                         unsigned int* errors) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t item = index; item < count; item += stride) {
        const float4 expected = transform_value(initial_value(item));
        const float4 actual = range[item];
        const bool mismatch = fabsf(actual.x - expected.x) > 1.0e-6f ||
                              fabsf(actual.y - expected.y) > 1.0e-6f ||
                              fabsf(actual.z - expected.z) > 1.0e-6f ||
                              fabsf(actual.w - expected.w) > 1.0e-6f;
        if (mismatch) atomicAdd(errors + (item < device_count ? 0 : 1), 1u);
    }
}

const char* driver_error(CUresult result) {
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    return name ? name : "CUDA_ERROR_UNKNOWN";
}

bool check_cuda(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    if (!check_cuda(cudaSetDevice(0), "select device 0")) return 1;
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuInit: %s\n", driver_error(CUDA_ERROR_NOT_INITIALIZED));
        return 1;
    }
    CUdevice device{};
    CUresult result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuDeviceGet: %s\n", driver_error(result));
        return 1;
    }

    int vmm_supported = 0;
    int host_vmm_supported = 0;
    result = cuDeviceGetAttribute(&vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "query VMM support: %s\n", driver_error(result));
        return 1;
    }
    result = cuDeviceGetAttribute(&host_vmm_supported, CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "query host VMM support: %s\n", driver_error(result));
        return 1;
    }
    if (!vmm_supported || !host_vmm_supported) {
        std::printf("capabilities,vmm=%d,host_vmm=%d,supported=no\n", vmm_supported, host_vmm_supported);
        return 3;
    }

    unified_memory::UnifiedRange range;
    std::string error;
    const size_t requested_device = static_cast<size_t>(options.device_mib) * 1024 * 1024;
    const size_t requested_host = static_cast<size_t>(options.host_mib) * 1024 * 1024;
    if (!range.create(requested_device, requested_host, 0, &error)) {
        std::fprintf(stderr, "VMM range creation failed: %s\n", error.c_str());
        return 3;
    }

    cudaDeviceProp runtime_prop{};
    if (!check_cuda(cudaGetDeviceProperties(&runtime_prop, 0), "query runtime device")) return 1;
    const size_t total_vectors = range.total_size() / sizeof(float4);
    const size_t device_vectors = range.device_size() / sizeof(float4);
    const int blocks = std::max(1, std::min(runtime_prop.multiProcessorCount * 16, 65535));
    float4* gpu_range = reinterpret_cast<float4*>(static_cast<uintptr_t>(range.base()));
    unsigned int* errors = nullptr;
    if (!check_cuda(cudaMalloc(&errors, 2 * sizeof(unsigned int)), "allocate GPU validation counters")) return 1;

    std::printf("capabilities,vmm=%d,host_vmm=%d,supported=yes\n", vmm_supported, host_vmm_supported);
    std::printf("vmm_range,base=%p,host_segment_offset_bytes=%zu,device_granularity_bytes=%zu,host_granularity_bytes=%zu,device_mapped_MiB=%.3f,host_mapped_MiB=%.3f,total_mapped_MiB=%.3f,one_gpu_va=yes\n",
                reinterpret_cast<void*>(static_cast<uintptr_t>(range.base())),
                range.device_size(), range.device_granularity(), range.host_granularity(),
                static_cast<double>(range.device_size()) / (1024.0 * 1024.0),
                static_cast<double>(range.host_size()) / (1024.0 * 1024.0),
                static_cast<double>(range.total_size()) / (1024.0 * 1024.0));
    std::printf("execution,one_pointer_for_device_and_host=yes,all_data_math=GPU_kernels,cpu_dataset_math=none\n");

    initialize<<<blocks, 256>>>(gpu_range, total_vectors);
    if (!check_cuda(cudaGetLastError(), "launch GPU initialization") ||
        !check_cuda(cudaDeviceSynchronize(), "wait for GPU initialization")) return 1;

    cudaEvent_t start{}, stop{};
    if (!check_cuda(cudaEventCreate(&start), "create timer") ||
        !check_cuda(cudaEventCreate(&stop), "create timer")) return 1;
    cudaEventRecord(start);
    transform<<<blocks, 256>>>(gpu_range, total_vectors);
    cudaEventRecord(stop);
    if (!check_cuda(cudaGetLastError(), "launch unified-range transform") ||
        !check_cuda(cudaEventSynchronize(stop), "wait for unified-range transform")) return 1;
    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "read transform time")) return 1;

    if (!check_cuda(cudaMemset(errors, 0, 2 * sizeof(unsigned int)), "clear error counters")) return 1;
    validate<<<blocks, 256>>>(gpu_range, total_vectors, device_vectors, errors);
    if (!check_cuda(cudaGetLastError(), "launch GPU validation") ||
        !check_cuda(cudaDeviceSynchronize(), "wait for GPU validation")) return 1;
    unsigned int host_errors[2]{};
    if (!check_cuda(cudaMemcpy(host_errors, errors, sizeof(host_errors), cudaMemcpyDeviceToHost),
                     "read GPU validation counters")) return 1;

    std::printf("result,elapsed_ms=%.6f,device_mismatches=%u,host_mismatches=%u,valid=%s\n",
                elapsed_ms, host_errors[0], host_errors[1],
                host_errors[0] == 0 && host_errors[1] == 0 ? "yes" : "no");

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(errors);
    return host_errors[0] == 0 && host_errors[1] == 0 ? 0 : 1;
}
