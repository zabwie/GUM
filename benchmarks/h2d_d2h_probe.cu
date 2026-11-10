// H2D vs D2H pinned bandwidth asymmetry probe (E035). H2D = GPU DMA-reads host (PCIe MemRd,
// subject to MaxReadReq); D2H = GPU DMA-writes host (posted MemWr). Asymmetry implicates MRRS/IOMMU.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>

static void chk(cudaError_t s, const char* w) { if (s != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(s)); exit(1); } }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    int engines = 0, async_engines = 0;
    chk(cudaDeviceGetAttribute(&async_engines, cudaDevAttrAsyncEngineCount, 0), "attr");
    printf("asyncEngineCount=%d\n", async_engines);
    const size_t sz = 256u << 20;
    uint8_t *pin, *dev;
    chk(cudaHostAlloc(&pin, sz, cudaHostAllocPortable), "pin");
    chk(cudaMalloc(&dev, sz), "dev");
    memset(pin, 1, sz);
    cudaStream_t st; chk(cudaStreamCreate(&st), "stream");
    const size_t total = 2ull << 30;
    for (int dir = 0; dir < 2; ++dir) {
        cudaMemcpyKind k = dir == 0 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        double best = 1e9;
        for (int rep = 0; rep < 4; ++rep) {
            double t0 = now();
            for (size_t i = 0; i < total / sz; ++i)
                chk(cudaMemcpyAsync(dir == 0 ? dev : pin, dir == 0 ? pin : dev, sz, k, st), "copy");
            chk(cudaStreamSynchronize(st), "sync");
            best = std::min(best, now() - t0);
        }
        printf("%s,GBps=%.2f\n", dir == 0 ? "h2d" : "d2h", total / best / 1e9);
    }
    return 0;
}