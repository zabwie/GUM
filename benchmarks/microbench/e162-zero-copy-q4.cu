// E162: staged DMA vs zero-copy mapped pinned memory — the memory-path question.
//
// E160 established that the verify pass is ~94% exposed PCIe: copy 397 us vs
// GEMV 200 us per 16 MiB chunk. Replacing staging with mapped reads is only
// viable if SM-issued reads of sequential pinned host memory sustain ~the DMA
// rate (24.9 GB/s measured, E099). This microbenchmark measures exactly that,
// with a GEMV-like consumer (each weight byte read once, shared by 24 rows,
// matching the packed verifier's access pattern) and checksum validation.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

__global__ void k_read_sum(const uint4* __restrict__ src, size_t n16, unsigned long long* __restrict__ sink) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long acc = 0;
    for (size_t k = i; k < n16; k += stride) {
        const uint4 v = src[k];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefdeadbeefULL) *sink = acc;
}

// Packed-verifier consumer: each 18-byte Q4 block read once, dotted against 24
// rows (activations indexed cyclically to bound memory; instruction mix kept).
__global__ void k_gemv_q4_packed(const uint8_t* __restrict__ weights, const float* __restrict__ acts,
                                 float* __restrict__ out, long nblocks, int rows, long act_span) {
    const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t* blk = weights + (size_t)b * 18;
    const unsigned dh = blk[12] | (blk[13] << 8);
    const unsigned sign = (dh >> 15) & 1, exp = (dh >> 10) & 31, man = dh & 1023;
    float d = (exp == 0) ? man * (1.f / 16384.f) * (1.f / 16384.f)
                         : ldexpf(1.f + man / 1024.f, (int)exp - 15);
    if (sign) d = -d;
    float acc[24];
    #pragma unroll
    for (int r = 0; r < 24; ++r) acc[r] = 0.f;
    const long col = (b * 32) % (act_span / 4);
    for (int j = 0; j < 16; ++j) {
        const uint8_t byte = blk[j];
        const float w0 = ((byte & 15) - 8) * d;
        const float w1 = ((byte >> 4) - 8) * d;
        #pragma unroll
        for (int r = 0; r < rows; ++r) {
            const float* a = acts + (size_t)r * (act_span / 4) + col + 2 * j;
            acc[r] = fmaf(w0, a[0], acc[r]);
            acc[r] = fmaf(w1, a[1], acc[r]);
        }
    }
    for (int r = 0; r < rows; ++r) atomicAdd(&out[r], acc[r]);
}

int main() {
    const long pass_bytes = 5241526272LL;
    const size_t bytes = (size_t)512 << 20;
    printf("e162_zero_copy,size_MiB=%zu\n", bytes >> 20);

    const char* model = getenv("GUM_MODEL");
    std::string path = model ? model : "/home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open model\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    long off = ((long)(fsize * 0.60)) / 18 * 18;
    std::vector<uint8_t> host(bytes);
    fseek(f, off, SEEK_SET);
    if (fread(host.data(), 1, bytes, f) != bytes) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    uint8_t* mapped = nullptr;
    CHECK(cudaHostAlloc(&mapped, bytes, cudaHostAllocMapped));
    memcpy(mapped, host.data(), bytes);
    uint8_t* mapped_dev = nullptr;
    CHECK(cudaHostGetDevicePointer((void**)&mapped_dev, mapped, 0));

    uint8_t* d_buf = nullptr;
    CHECK(cudaMalloc(&d_buf, bytes));
    unsigned long long* d_sink; CHECK(cudaMalloc(&d_sink, 8));
    CHECK(cudaMemset(d_sink, 0, 8));

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));
    const int reps = 8;
    const size_t n16 = bytes / 16;

    // pinned control: DMA from a second pinned buffer (E099's 24.9 GB/s condition)
    uint8_t* pinned_src = nullptr;
    CHECK(cudaHostAlloc(&pinned_src, bytes, cudaHostAllocDefault));
    memcpy(pinned_src, host.data(), bytes);
    float copy_ms = 0.f;
    CHECK(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) CHECK(cudaMemcpyAsync(d_buf, pinned_src, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(ev1)); CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&copy_ms, ev0, ev1)); copy_ms /= reps;
    printf("staged_dma,ms=%.2f,GBps=%.2f\n", copy_ms, bytes / copy_ms / 1e6);

    float map_ms = 0.f;
    CHECK(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) k_read_sum<<<1184, 256>>>((const uint4*)mapped_dev, n16, d_sink);
    CHECK(cudaEventRecord(ev1)); CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&map_ms, ev0, ev1)); map_ms /= reps;
    printf("mapped_read,ms=%.2f,GBps=%.2f\n", map_ms, bytes / map_ms / 1e6);

    const long nblocks = bytes / 18;
    const int rows = 24;
    const long act_span = 5120 * 4;   // 24 rows x 5120 floats
    float* d_acts; CHECK(cudaMalloc(&d_acts, act_span));
    {
        std::vector<float> a(act_span / 4);
        srand(7);
        for (auto& v : a) v = (rand() % 2000 - 1000) / 512.f;
        CHECK(cudaMemcpy(d_acts, a.data(), act_span, cudaMemcpyHostToDevice));
    }
    float *d_out_map, *d_out_vram;
    CHECK(cudaMalloc(&d_out_map, rows * 4)); CHECK(cudaMalloc(&d_out_vram, rows * 4));
    const int threads = 256;
    const dim3 g((unsigned)((nblocks + threads - 1) / threads));

    float gemv_vram_ms = 0.f, gemv_map_ms = 0.f;
    CHECK(cudaMemset(d_out_vram, 0, rows * 4));
    CHECK(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) k_gemv_q4_packed<<<g, threads>>>(d_buf, d_acts, d_out_vram, nblocks, rows, act_span);
    CHECK(cudaEventRecord(ev1)); CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&gemv_vram_ms, ev0, ev1)); gemv_vram_ms /= reps;
    printf("gemv_vram,ms=%.2f\n", gemv_vram_ms);

    CHECK(cudaMemset(d_out_map, 0, rows * 4));
    CHECK(cudaEventRecord(ev0));
    for (int r = 0; r < reps; ++r) k_gemv_q4_packed<<<g, threads>>>(mapped_dev, d_acts, d_out_map, nblocks, rows, act_span);
    CHECK(cudaEventRecord(ev1)); CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&gemv_map_ms, ev0, ev1)); gemv_map_ms /= reps;
    printf("gemv_mapped,ms=%.2f,effective_GBps=%.2f,pass_equiv_s=%.3f\n",
           gemv_map_ms, bytes / gemv_map_ms / 1e6, pass_bytes / (bytes / gemv_map_ms / 1e6) / 1e9);

    std::vector<float> o1(rows), o2(rows);
    CHECK(cudaMemcpy(o1.data(), d_out_vram, rows * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(o2.data(), d_out_map, rows * 4, cudaMemcpyDeviceToHost));
    long mismatch = 0;
    for (int i = 0; i < rows; ++i) if (memcmp(&o1[i], &o2[i], 4)) ++mismatch;
    printf("validate,bit_exact_mismatches=%ld/%d\n", mismatch, rows);

    printf("summary,mapped_gemv_vs_dma=%.2fx,staged_dma_pass_equiv_s=%.3f,zero_copy_pass_equiv_s=%.3f\n",
           gemv_map_ms / copy_ms, pass_bytes / (bytes / copy_ms / 1e6) / 1e9,
           pass_bytes / (bytes / gemv_map_ms / 1e6) / 1e9);
    return 0;
}
