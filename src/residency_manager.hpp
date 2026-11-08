#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace unified_memory {

struct ResidencyUpdate {
    size_t retained_pages = 0;
    size_t promoted_pages = 0;
    size_t evicted_pages = 0;
    size_t host_to_device_bytes = 0;
};

class ResidencyManager {
public:
    ResidencyManager() = default;
    ResidencyManager(const ResidencyManager&) = delete;
    ResidencyManager& operator=(const ResidencyManager&) = delete;

    ~ResidencyManager() { reset(); }

    bool create(const float4* mapped_host_device_pointer, size_t host_bytes,
                size_t page_bytes, size_t maximum_cache_pages,
                std::string* error) {
        reset();
        if (mapped_host_device_pointer == nullptr || host_bytes == 0 ||
            page_bytes == 0 || maximum_cache_pages == 0 ||
            host_bytes % page_bytes != 0) {
            if (error) *error = "invalid host range, page size, or cache capacity";
            return false;
        }
        host_ = mapped_host_device_pointer;
        host_bytes_ = host_bytes;
        page_bytes_ = page_bytes;
        page_count_ = host_bytes / page_bytes;
        maximum_cache_pages_ = maximum_cache_pages;
        page_to_frame_.assign(page_count_, -1);
        frames_.resize(maximum_cache_pages_);

        cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device_page_to_frame_),
                                        page_count_ * sizeof(int));
        if (status != cudaSuccess) {
            if (error) *error = std::string("allocate device page map: ") + cudaGetErrorString(status);
            reset();
            return false;
        }
        status = cudaMalloc(reinterpret_cast<void**>(&device_frame_pointers_),
                            maximum_cache_pages_ * sizeof(float4*));
        if (status != cudaSuccess) {
            if (error) *error = std::string("allocate device frame table: ") + cudaGetErrorString(status);
            reset();
            return false;
        }

        std::vector<float4*> empty_frames(maximum_cache_pages_, nullptr);
        status = cudaMemcpy(device_page_to_frame_, page_to_frame_.data(),
                            page_count_ * sizeof(int), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            status = cudaMemcpy(device_frame_pointers_, empty_frames.data(),
                                maximum_cache_pages_ * sizeof(float4*), cudaMemcpyHostToDevice);
        }
        if (status != cudaSuccess) {
            if (error) *error = std::string("initialize device cache tables: ") + cudaGetErrorString(status);
            reset();
            return false;
        }
        return true;
    }

    bool update(const std::vector<uint32_t>& page_access_counts,
                size_t requested_capacity_pages, ResidencyUpdate* update,
                std::string* error) {
        if (page_access_counts.size() != page_count_ ||
            requested_capacity_pages > maximum_cache_pages_) {
            if (error) *error = "access-count size or requested cache capacity is invalid";
            return false;
        }
        if (update) *update = {};

        std::vector<size_t> ranked_pages(page_count_);
        for (size_t page = 0; page < page_count_; ++page) ranked_pages[page] = page;
        std::stable_sort(ranked_pages.begin(), ranked_pages.end(),
            [&](size_t left, size_t right) {
                if (page_access_counts[left] != page_access_counts[right]) {
                    return page_access_counts[left] > page_access_counts[right];
                }
                return left < right;
            });

        std::vector<unsigned char> wanted(page_count_, 0);
        size_t wanted_count = 0;
        for (size_t page : ranked_pages) {
            if (wanted_count == requested_capacity_pages ||
                page_access_counts[page] == 0) break;
            wanted[page] = 1;
            ++wanted_count;
        }

        capacity_pages_ = requested_capacity_pages;
        for (size_t frame = 0; frame < frames_.size(); ++frame) {
            const int page = frames_[frame].page;
            if (page < 0) continue;
            if (frame >= capacity_pages_ || !wanted[static_cast<size_t>(page)]) {
                page_to_frame_[static_cast<size_t>(page)] = -1;
                cudaError_t status = cudaFree(frames_[frame].data);
                if (status != cudaSuccess) {
                    if (error) *error = std::string("free evicted GPU page: ") + cudaGetErrorString(status);
                    return false;
                }
                frames_[frame] = {};
                if (update) ++update->evicted_pages;
            } else if (update) {
                ++update->retained_pages;
            }
        }

        for (size_t page : ranked_pages) {
            if (wanted[page] == 0 || page_to_frame_[page] >= 0) continue;
            size_t frame = 0;
            while (frame < capacity_pages_ && frames_[frame].page >= 0) ++frame;
            if (frame == capacity_pages_) break;

            float4* device_page = nullptr;
            cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device_page), page_bytes_);
            if (status != cudaSuccess) {
                if (error) *error = std::string("allocate GPU cache page: ") + cudaGetErrorString(status);
                return false;
            }
            const size_t offset_vectors = page * (page_bytes_ / sizeof(float4));
            status = cudaMemcpy(device_page, host_ + offset_vectors, page_bytes_, cudaMemcpyDefault);
            if (status != cudaSuccess) {
                cudaFree(device_page);
                if (error) *error = std::string("prefetch host page into GPU cache: ") + cudaGetErrorString(status);
                return false;
            }

            frames_[frame].page = static_cast<int>(page);
            frames_[frame].data = device_page;
            page_to_frame_[page] = static_cast<int>(frame);
            if (update) {
                ++update->promoted_pages;
                update->host_to_device_bytes += page_bytes_;
            }
        }

        std::vector<float4*> frame_pointers(maximum_cache_pages_, nullptr);
        for (size_t frame = 0; frame < frames_.size(); ++frame) {
            frame_pointers[frame] = frames_[frame].data;
        }
        cudaError_t status = cudaMemcpy(device_page_to_frame_, page_to_frame_.data(),
                                        page_count_ * sizeof(int), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            status = cudaMemcpy(device_frame_pointers_, frame_pointers.data(),
                                maximum_cache_pages_ * sizeof(float4*), cudaMemcpyHostToDevice);
        }
        if (status != cudaSuccess) {
            if (error) *error = std::string("upload GPU cache tables: ") + cudaGetErrorString(status);
            return false;
        }
        return true;
    }

    void reset() {
        for (Frame& frame : frames_) {
            if (frame.data) cudaFree(frame.data);
            frame = {};
        }
        if (device_frame_pointers_) cudaFree(device_frame_pointers_);
        if (device_page_to_frame_) cudaFree(device_page_to_frame_);
        device_frame_pointers_ = nullptr;
        device_page_to_frame_ = nullptr;
        frames_.clear();
        page_to_frame_.clear();
        host_ = nullptr;
        host_bytes_ = 0;
        page_bytes_ = 0;
        page_count_ = 0;
        maximum_cache_pages_ = 0;
        capacity_pages_ = 0;
    }

    const int* device_page_to_frame() const { return device_page_to_frame_; }
    float4* const* device_frame_pointers() const { return device_frame_pointers_; }
    size_t page_count() const { return page_count_; }
    size_t capacity_pages() const { return capacity_pages_; }

private:
    struct Frame {
        int page = -1;
        float4* data = nullptr;
    };

    const float4* host_ = nullptr;
    size_t host_bytes_ = 0;
    size_t page_bytes_ = 0;
    size_t page_count_ = 0;
    size_t maximum_cache_pages_ = 0;
    size_t capacity_pages_ = 0;
    std::vector<int> page_to_frame_;
    std::vector<Frame> frames_;
    int* device_page_to_frame_ = nullptr;
    float4** device_frame_pointers_ = nullptr;
};

}  // namespace unified_memory
