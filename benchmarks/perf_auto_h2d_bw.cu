// Diagnostic: pinned-host -> device copy bandwidth at decoder-realistic sizes.
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("device=%s cc=%d.%d copyEngines~%d\n", prop.name, prop.major, prop.minor,
           prop.asyncEngineCount);
    const size_t total = 1ull << 30;  // 1 GiB
    void *host, *dev;
    cudaHostAlloc(&host, total, cudaHostAllocPortable);
    cudaMalloc(&dev, total);
    memset(host, 1, total);
    cudaStream_t st; cudaStreamCreate(&st);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // warmup
    for (int i = 0; i < 3; ++i) cudaMemcpyAsync(dev, host, total, cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);
    for (size_t chunk : {16ull << 20, 12ull << 20, 4ull << 20, 1ull << 20}) {
        const int iters = (int)(total / chunk);
        cudaEventRecord(e0, st);
        for (int i = 0; i < iters; ++i)
            cudaMemcpyAsync((char*)dev + (size_t)i * chunk, (char*)host + (size_t)i * chunk,
                            chunk, cudaMemcpyHostToDevice, st);
        cudaEventRecord(e1, st);
        cudaStreamSynchronize(st);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        printf("chunk=%zuMiB iters=%d total=%.1fGiB time=%.1fms bw=%.1fGB/s\n",
               chunk >> 20, iters, total / 1073741824.0, ms, total / (ms * 1e6));
    }
    return 0;
}
