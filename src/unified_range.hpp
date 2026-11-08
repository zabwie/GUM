#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace unified_memory {

class UnifiedRange {
public:
    UnifiedRange() = default;
    UnifiedRange(const UnifiedRange&) = delete;
    UnifiedRange& operator=(const UnifiedRange&) = delete;

    ~UnifiedRange() { reset(); }

    bool create(size_t device_bytes, size_t host_bytes, int device_ordinal,
                std::string* error) {
        reset();
        if (device_bytes == 0 || host_bytes == 0) {
            if (error) *error = "both VMM regions must be nonempty";
            return false;
        }

        CUdevice device{};
        CUresult result = cuDeviceGet(&device, device_ordinal);
        if (!succeeded(result, "cuDeviceGet", error)) return false;

        int vmm_supported = 0;
        int host_vmm_supported = 0;
        result = cuDeviceGetAttribute(&vmm_supported,
                                      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                      device);
        if (!succeeded(result, "query device VMM support", error)) return false;
        result = cuDeviceGetAttribute(&host_vmm_supported,
                                      CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                      device);
        if (!succeeded(result, "query host VMM support", error)) return false;
        if (!vmm_supported || !host_vmm_supported) {
            if (error) *error = "device does not advertise VMM plus host-backed VMM support";
            return false;
        }

        CUmemAllocationProp device_prop{};
        device_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        device_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        device_prop.location.id = device_ordinal;

        CUmemAllocationProp host_prop{};
        host_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        host_prop.location.type = CU_MEM_LOCATION_TYPE_HOST;

        result = cuMemGetAllocationGranularity(&device_granularity_, &device_prop,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (!succeeded(result, "query device allocation granularity", error)) return false;
        result = cuMemGetAllocationGranularity(&host_granularity_, &host_prop,
                                              CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (!succeeded(result, "query host allocation granularity", error)) return false;

        size_t common_step = 0;
        if (!checked_lcm(device_granularity_, host_granularity_, &common_step)) {
            if (error) *error = "allocation granularity overflow";
            return false;
        }
        if (!round_up(device_bytes, common_step, &device_size_) ||
            !round_up(host_bytes, host_granularity_, &host_size_) ||
            device_size_ > std::numeric_limits<size_t>::max() - host_size_) {
            if (error) *error = "VMM allocation size overflow";
            reset();
            return false;
        }
        const size_t total_size = device_size_ + host_size_;
        size_t reserve_alignment = 0;
        if (!checked_lcm(device_granularity_, host_granularity_, &reserve_alignment) ||
            (reserve_alignment & (reserve_alignment - 1)) != 0) {
            if (error) *error = "CUDA mapping granularities cannot share a power-of-two aligned VA";
            reset();
            return false;
        }

        result = cuMemAddressReserve(&base_, total_size, reserve_alignment, 0, 0);
        if (!succeeded(result, "reserve unified GPU virtual range", error)) {
            reset();
            return false;
        }

        result = cuMemCreate(&device_handle_, device_size_, &device_prop, 0);
        if (!succeeded(result, "create device-backed VMM allocation", error)) {
            reset();
            return false;
        }
        result = cuMemCreate(&host_handle_, host_size_, &host_prop, 0);
        if (!succeeded(result, "create host-backed VMM allocation", error)) {
            reset();
            return false;
        }
        result = cuMemMap(base_, device_size_, 0, device_handle_, 0);
        if (!succeeded(result, "map device-backed VMM region", error)) {
            reset();
            return false;
        }
        device_mapped_ = true;

        result = cuMemMap(base_ + device_size_, host_size_, 0, host_handle_, 0);
        if (!succeeded(result, "map host-backed VMM region", error)) {
            reset();
            return false;
        }
        host_mapped_ = true;

        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_ordinal;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        result = cuMemSetAccess(base_, total_size, &access, 1);
        if (!succeeded(result, "grant GPU read/write access to unified range", error)) {
            reset();
            return false;
        }
        return true;
    }

    void reset() {
        if (host_mapped_) {
            cuMemUnmap(base_ + device_size_, host_size_);
            host_mapped_ = false;
        }
        if (device_mapped_) {
            cuMemUnmap(base_, device_size_);
            device_mapped_ = false;
        }
        if (host_handle_) {
            cuMemRelease(host_handle_);
            host_handle_ = {};
        }
        if (device_handle_) {
            cuMemRelease(device_handle_);
            device_handle_ = {};
        }
        if (base_) {
            cuMemAddressFree(base_, device_size_ + host_size_);
            base_ = 0;
        }
        device_size_ = 0;
        host_size_ = 0;
        device_granularity_ = 0;
        host_granularity_ = 0;
    }

    CUdeviceptr base() const { return base_; }
    CUdeviceptr host_region() const { return base_ + device_size_; }
    size_t device_size() const { return device_size_; }
    size_t host_size() const { return host_size_; }
    size_t total_size() const { return device_size_ + host_size_; }
    size_t device_granularity() const { return device_granularity_; }
    size_t host_granularity() const { return host_granularity_; }

private:
    static bool succeeded(CUresult result, const char* operation, std::string* error) {
        if (result == CUDA_SUCCESS) return true;
        const char* name = nullptr;
        const char* detail = nullptr;
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &detail);
        if (error) {
            *error = operation;
            *error += ": ";
            *error += name ? name : "CUDA_ERROR_UNKNOWN";
            if (detail) {
                *error += " (";
                *error += detail;
                *error += ")";
            }
        }
        return false;
    }

    static size_t gcd(size_t a, size_t b) {
        while (b != 0) {
            const size_t remainder = a % b;
            a = b;
            b = remainder;
        }
        return a;
    }

    static bool checked_lcm(size_t a, size_t b, size_t* out) {
        if (a == 0 || b == 0) return false;
        const size_t reduced = a / gcd(a, b);
        if (reduced > std::numeric_limits<size_t>::max() / b) return false;
        *out = reduced * b;
        return true;
    }

    static bool round_up(size_t value, size_t granularity, size_t* out) {
        if (granularity == 0) return false;
        const size_t remainder = value % granularity;
        const size_t padding = remainder == 0 ? 0 : granularity - remainder;
        if (value > std::numeric_limits<size_t>::max() - padding) return false;
        *out = value + padding;
        return true;
    }

    CUdeviceptr base_ = 0;
    CUmemGenericAllocationHandle device_handle_{};
    CUmemGenericAllocationHandle host_handle_{};
    size_t device_size_ = 0;
    size_t host_size_ = 0;
    size_t device_granularity_ = 0;
    size_t host_granularity_ = 0;
    bool device_mapped_ = false;
    bool host_mapped_ = false;
};

}  // namespace unified_memory
