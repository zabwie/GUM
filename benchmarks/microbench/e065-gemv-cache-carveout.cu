// E065: test the L1/shared carveout on the E058 full-shape GEMV sweep.
// The production kernel uses little/no shared memory and repeatedly reads a
// small activation working set. PreferL1 may reduce L1 eviction from weights.
#define main e058_original_main
#include "e058-gemv-vector-warps.cu"
#undef main

int main() {
    CHECK(cudaFuncSetCacheConfig(k_ref, cudaFuncCachePreferL1));
    CHECK(cudaFuncSetCacheConfig(k_vwarp<16>, cudaFuncCachePreferL1));
    CHECK(cudaFuncSetCacheConfig(k_vwarp<8>, cudaFuncCachePreferL1));
    CHECK(cudaFuncSetCacheConfig(k_vwarp<4>, cudaFuncCachePreferL1));
    CHECK(cudaFuncSetCacheConfig(k_vwarp<2>, cudaFuncCachePreferL1));
    CHECK(cudaFuncSetCacheConfig(k_vwarp<1>, cudaFuncCachePreferL1));
    puts("E065 cache carveout: PreferL1");
    return e058_original_main();
}
