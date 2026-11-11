// E052: test a warp-per-vector Q4_0 GEMV mapping against the current NB=16 kernel.
// Each CTA owns one output row and has one warp per input vector. The quantized
// weight row is staged once in shared memory, while each warp preserves the same
// per-lane K-block order and final warp reduction as the production kernel.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CHECK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

static uint16_t host_half_bits(float x) {
    const __half h = __float2half_rn(x);
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

template <int NB>
__global__ void k_ref(const uint8_t* w, long K, long rows, const int8_t* xq,
                      const float* xs, const int* xsum, float* y, long ystride) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32;
    const size_t rowbytes = (size_t)nb * 18;
    const uint8_t* r = w + (size_t)row * rowbytes;
    float acc[NB];
#pragma unroll
    for (int v = 0; v < NB; ++v) acc[v] = 0.f;
    for (long b = lane; b < nb; b += 32) {
        const uint16_t* blk = (const uint16_t*)(r + (size_t)b * 18);
        const float d = __half2float(__ushort_as_half(blk[0]));
        uint32_t q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
        const uint32_t lo0=q[0]&0x0F0F0F0Fu, lo1=q[1]&0x0F0F0F0Fu, lo2=q[2]&0x0F0F0F0Fu, lo3=q[3]&0x0F0F0F0Fu;
        const uint32_t hi0=(q[0]>>4)&0x0F0F0F0Fu, hi1=(q[1]>>4)&0x0F0F0F0Fu, hi2=(q[2]>>4)&0x0F0F0F0Fu, hi3=(q[3]>>4)&0x0F0F0F0Fu;
#pragma unroll
        for (int v = 0; v < NB; ++v) {
            const int4* xp = (const int4*)(xq + (size_t)v * K + b * 32);
            const int4 xa = xp[0], xb = xp[1];
            int s = 0;
            s=__dp4a((int)lo0,xa.x,s); s=__dp4a((int)lo1,xa.y,s); s=__dp4a((int)lo2,xa.z,s); s=__dp4a((int)lo3,xa.w,s);
            s=__dp4a((int)hi0,xb.x,s); s=__dp4a((int)hi1,xb.y,s); s=__dp4a((int)hi2,xb.z,s); s=__dp4a((int)hi3,xb.w,s);
            acc[v] += d * xs[(size_t)v * nb + b] * (float)(s - 8 * xsum[(size_t)v * nb + b]);
        }
    }
#pragma unroll
    for (int v = 0; v < NB; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[(size_t)v * ystride + row] = a;
    }
}

template <int NB, int VPW>
__global__ void k_warp_per_vector(const uint8_t* w, long K, long rows, const int8_t* xq,
                                 const float* xs, const int* xsum, float* y, long ystride) {
    extern __shared__ uint8_t wrow[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int v0 = (tid >> 5) * VPW;
    const long row = blockIdx.x;
    const long nb = K / 32;
    const size_t rowbytes = (size_t)nb * 18;
    const uint16_t* src = (const uint16_t*)(w + (size_t)row * rowbytes);
    uint16_t* dst = (uint16_t*)wrow;
    for (size_t i = (size_t)tid; i < rowbytes / 2; i += blockDim.x) dst[i] = src[i];
    __syncthreads();

    float acc[VPW];
#pragma unroll
    for (int u = 0; u < VPW; ++u) acc[u] = 0.f;
    const uint16_t* rowp = (const uint16_t*)wrow;
    for (long b = lane; b < nb; b += 32) {
        const uint16_t* blk = rowp + (size_t)b * 9;
        const float d = __half2float(__ushort_as_half(blk[0]));
        uint32_t q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
        const uint32_t lo0=q[0]&0x0F0F0F0Fu, lo1=q[1]&0x0F0F0F0Fu, lo2=q[2]&0x0F0F0F0Fu, lo3=q[3]&0x0F0F0F0Fu;
        const uint32_t hi0=(q[0]>>4)&0x0F0F0F0Fu, hi1=(q[1]>>4)&0x0F0F0F0Fu, hi2=(q[2]>>4)&0x0F0F0F0Fu, hi3=(q[3]>>4)&0x0F0F0F0Fu;
#pragma unroll
        for (int u = 0; u < VPW; ++u) {
            const int v = v0 + u;
            const int4* xp = (const int4*)(xq + (size_t)v * K + b * 32);
            const int4 xa = xp[0], xb = xp[1];
            int s = 0;
            s=__dp4a((int)lo0,xa.x,s); s=__dp4a((int)lo1,xa.y,s); s=__dp4a((int)lo2,xa.z,s); s=__dp4a((int)lo3,xa.w,s);
            s=__dp4a((int)hi0,xb.x,s); s=__dp4a((int)hi1,xb.y,s); s=__dp4a((int)hi2,xb.z,s); s=__dp4a((int)hi3,xb.w,s);
            acc[u] += d * xs[(size_t)v * nb + b] * (float)(s - 8 * xsum[(size_t)v * nb + b]);
        }
    }
#pragma unroll
    for (int u = 0; u < VPW; ++u) {
        float a = acc[u];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[(size_t)(v0 + u) * ystride + row] = a;
    }
}

// Same mapping without shared-memory staging: sibling vector warps reload the row from
// the normal read-only path. This checks whether the block-level shared copy was the cost.
template <int NB, int VPW>
__global__ void k_warp_per_vector_global(const uint8_t* w, long K, long rows, const int8_t* xq,
                                         const float* xs, const int* xsum, float* y, long ystride) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int v0 = (tid >> 5) * VPW;
    const long row = blockIdx.x;
    const long nb = K / 32;
    const size_t rowbytes = (size_t)nb * 18;
    const uint16_t* rowp = (const uint16_t*)(w + (size_t)row * rowbytes);
    float acc[VPW];
#pragma unroll
    for (int u = 0; u < VPW; ++u) acc[u] = 0.f;
    for (long b = lane; b < nb; b += 32) {
        const uint16_t* blk = rowp + (size_t)b * 9;
        const float d = __half2float(__ushort_as_half(blk[0]));
        uint32_t q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
        const uint32_t lo0=q[0]&0x0F0F0F0Fu,lo1=q[1]&0x0F0F0F0Fu,lo2=q[2]&0x0F0F0F0Fu,lo3=q[3]&0x0F0F0F0Fu;
        const uint32_t hi0=(q[0]>>4)&0x0F0F0F0Fu,hi1=(q[1]>>4)&0x0F0F0F0Fu,hi2=(q[2]>>4)&0x0F0F0F0Fu,hi3=(q[3]>>4)&0x0F0F0F0Fu;
#pragma unroll
        for (int u = 0; u < VPW; ++u) {
            const int v = v0 + u;
            const int4* xp = (const int4*)(xq + (size_t)v * K + b * 32);
            const int4 xa = xp[0], xb = xp[1];
            int s=0;
            s=__dp4a((int)lo0,xa.x,s);s=__dp4a((int)lo1,xa.y,s);s=__dp4a((int)lo2,xa.z,s);s=__dp4a((int)lo3,xa.w,s);
            s=__dp4a((int)hi0,xb.x,s);s=__dp4a((int)hi1,xb.y,s);s=__dp4a((int)hi2,xb.z,s);s=__dp4a((int)hi3,xb.w,s);
            acc[u] += d * xs[(size_t)v * nb + b] * (float)(s - 8 * xsum[(size_t)v * nb + b]);
        }
    }
#pragma unroll
    for (int u = 0; u < VPW; ++u) {
        float a=acc[u];for(int o=16;o>0;o>>=1)a+=__shfl_down_sync(0xffffffffu,a,o);
        if(lane==0)y[(size_t)(v0+u)*ystride+row]=a;
    }
}

int main() {
    constexpr long K = 5120, rows = 11652;
    constexpr int NB = 16, iters = 40, trials = 3;
    const long nb = K / 32;
    const size_t weight_bytes = (size_t)rows * nb * 18;
    std::vector<uint8_t> hw(weight_bytes);
    std::vector<int8_t> hx((size_t)NB * K);
    std::vector<float> hxs((size_t)NB * nb);
    std::vector<int> hsum((size_t)NB * nb);
    uint32_t seed = 1234567;
    auto rng = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    for (size_t i = 0; i < weight_bytes; ++i) hw[i] = (uint8_t)(rng() >> 24);
    for (long r = 0; r < rows; ++r) for (long b = 0; b < nb; ++b)
        *(uint16_t*)(hw.data() + ((size_t)r * nb + b) * 18) = host_half_bits(0.001f * (float)((r + b) % 97 + 1));
    for (auto& q : hx) q = (int8_t)((rng() % 31) - 15);
    for (size_t i = 0; i < hxs.size(); ++i) { hxs[i] = 0.01f; hsum[i] = (int)(rng() % 7) - 3; }

    uint8_t *dw, *dact; int8_t *dx; float *dxs, *dy0, *dy1; int* dsum;
    const size_t qbytes=hx.size(), scale_bytes=hxs.size()*sizeof(float), sum_bytes=hsum.size()*sizeof(int);
    const size_t scale_off=(qbytes+255)&~(size_t)255, sum_off=scale_off+scale_bytes, act_bytes=sum_off+sum_bytes;
    CHECK(cudaMalloc(&dw, weight_bytes)); CHECK(cudaMalloc(&dact, act_bytes));
    dx=(int8_t*)dact;dxs=(float*)(dact+scale_off);dsum=(int*)(dact+sum_off);
    CHECK(cudaMalloc(&dy0, (size_t)NB * rows * sizeof(float))); CHECK(cudaMalloc(&dy1, (size_t)NB * rows * sizeof(float)));
    CHECK(cudaMemcpy(dw, hw.data(), weight_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hx.data(), qbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dxs, hxs.data(), scale_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dsum, hsum.data(), sum_bytes, cudaMemcpyHostToDevice));

    const unsigned gref = (unsigned)((rows + 3) / 4), gw = (unsigned)rows;
    const size_t smem = (size_t)nb * 18;
    k_ref<NB><<<gref,128>>>(dw,K,rows,dx,dxs,dsum,dy0,rows);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    std::vector<float> hy0((size_t)NB * rows), hy1((size_t)NB * rows);
    CHECK(cudaMemcpy(hy0.data(),dy0,hy0.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hy1.data(),dy1,hy1.size()*sizeof(float),cudaMemcpyDeviceToHost));
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    std::vector<float> ref_ms;
    for (int t=0;t<trials;++t) {
        for (int i=0;i<5;++i) k_ref<NB><<<gref,128>>>(dw,K,rows,dx,dxs,dsum,dy0,rows);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(a)); for(int i=0;i<iters;++i) k_ref<NB><<<gref,128>>>(dw,K,rows,dx,dxs,dsum,dy0,rows); CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        float ms; CHECK(cudaEventElapsedTime(&ms,a,b)); ref_ms.push_back(ms/iters);
    }
    auto med=[](std::vector<float> x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    const float ref=med(ref_ms);
    printf("E052,shape=%ldx%ld,weight_MiB=%.2f,reference_ms=%.4f\n",rows,K,weight_bytes/1048576.0,ref);
    for (int i=0;i<trials;++i) printf("reference_trial,%d,ms=%.4f\n",i,ref_ms[i]);
    bool all_exact = true;
    for (int warps : {1,2,4,8,16}) {
        const unsigned block=(unsigned)(warps*32),grid_rows=(unsigned)((rows+warps-1)/warps);
        k_ref<NB><<<grid_rows,block>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hy1.data(),dy1,hy1.size()*sizeof(float),cudaMemcpyDeviceToHost));
        const bool exact=memcmp(hy0.data(),hy1.data(),hy0.size()*sizeof(float))==0;all_exact &= exact;
        std::vector<float> mslist;
        for(int t=0;t<trials;++t){
            CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_ref<NB><<<grid_rows,block>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));mslist.push_back(ms/iters);
        }
        const float cand=med(mslist);
        printf("block_config,warps=%d,threads=%u,ms=%.4f,speedup=%.3f,bitexact=%s\n",warps,block,cand,ref/cand,exact?"YES":"NO");
        for(int t=0;t<trials;++t)printf("block_trial,warps=%d,trial=%d,ms=%.4f\n",warps,t,mslist[t]);
    }
    auto run_cfg = [&](int vpw) {
        const int warps = NB / vpw;
        const unsigned block = (unsigned)(warps * 32);
        bool exact = true;
        if (vpw==1) k_warp_per_vector<NB,1><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
        else if (vpw==2) k_warp_per_vector<NB,2><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
        else if (vpw==4) k_warp_per_vector<NB,4><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
        else if (vpw==8) k_warp_per_vector<NB,8><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
        else k_warp_per_vector<NB,16><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hy1.data(),dy1,hy1.size()*sizeof(float),cudaMemcpyDeviceToHost));
        exact = memcmp(hy0.data(),hy1.data(),hy0.size()*sizeof(float)) == 0;
        if (!exact) for(size_t i=0;i<hy0.size();++i) if(hy0[i]!=hy1[i]) { printf("first_diff,vpw=%d,index=%zu,ref=%a,candidate=%a\n",vpw,i,hy0[i],hy1[i]); break; }
        std::vector<float> mslist;
        for(int t=0;t<trials;++t) {
            CHECK(cudaEventRecord(a));
            for(int i=0;i<iters;++i) {
                if (vpw==1) k_warp_per_vector<NB,1><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
                else if (vpw==2) k_warp_per_vector<NB,2><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
                else if (vpw==4) k_warp_per_vector<NB,4><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
                else if (vpw==8) k_warp_per_vector<NB,8><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
                else k_warp_per_vector<NB,16><<<gw,block,smem>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
            }
            CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
            float ms; CHECK(cudaEventElapsedTime(&ms,a,b)); mslist.push_back(ms/iters);
        }
        const float cand=med(mslist); all_exact &= exact;
        printf("config,vpw=%d,warps=%d,ms=%.4f,speedup=%.3f,bitexact=%s\n",vpw,warps,cand,ref/cand,exact?"YES":"NO");
        for(int t=0;t<trials;++t) printf("candidate_trial,vpw=%d,trial=%d,ms=%.4f\n",vpw,t,mslist[t]);
    };
    for (int vpw : {1,2,4,8,16}) run_cfg(vpw);

    for (int vpw : {1,2,4,8,16}) {
        const int warps=NB/vpw;const unsigned block=(unsigned)(warps*32);
        auto launch_global=[&](){
            if(vpw==1)k_warp_per_vector_global<NB,1><<<gw,block>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
            else if(vpw==2)k_warp_per_vector_global<NB,2><<<gw,block>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
            else if(vpw==4)k_warp_per_vector_global<NB,4><<<gw,block>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
            else if(vpw==8)k_warp_per_vector_global<NB,8><<<gw,block>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
            else k_warp_per_vector_global<NB,16><<<gw,block>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);
        };
        launch_global();CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hy1.data(),dy1,hy1.size()*sizeof(float),cudaMemcpyDeviceToHost));
        const bool exact=memcmp(hy0.data(),hy1.data(),hy0.size()*sizeof(float))==0;all_exact &= exact;
        if(!exact)for(size_t i=0;i<hy0.size();++i)if(hy0[i]!=hy1[i]){printf("global_first_diff,vpw=%d,index=%zu,ref=%a,candidate=%a\n",vpw,i,hy0[i],hy1[i]);break;}
        std::vector<float> mslist;
        for(int t=0;t<trials;++t){
            CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)launch_global();CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));mslist.push_back(ms/iters);
        }
        const float cand=med(mslist);
        printf("global_config,vpw=%d,warps=%d,ms=%.4f,speedup=%.3f,bitexact=%s\n",vpw,warps,cand,ref/cand,exact?"YES":"NO");
        for(int t=0;t<trials;++t)printf("global_trial,vpw=%d,trial=%d,ms=%.4f\n",vpw,t,mslist[t]);
    }
    cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
    if(prop.persistingL2CacheMaxSize>0){
        CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,(size_t)prop.persistingL2CacheMaxSize));
        cudaStreamAttrValue apw{};apw.accessPolicyWindow.base_ptr=dact;apw.accessPolicyWindow.num_bytes=act_bytes;
        apw.accessPolicyWindow.hitRatio=1.0f;apw.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;apw.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
        CHECK(cudaStreamSetAttribute((cudaStream_t)0,cudaStreamAttributeAccessPolicyWindow,&apw));
        std::vector<float> cache_ms;
        for(int t=0;t<trials;++t){
            CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_ref<NB><<<gref,128>>>(dw,K,rows,dx,dxs,dsum,dy1,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));cache_ms.push_back(ms/iters);
        }
        const float cached=med(cache_ms);
        CHECK(cudaMemcpy(hy1.data(),dy1,hy1.size()*sizeof(float),cudaMemcpyDeviceToHost));
        const bool exact=memcmp(hy0.data(),hy1.data(),hy0.size()*sizeof(float))==0;all_exact &= exact;
        printf("l2_policy,cache_max_MiB=%.3f,window_KiB=%.1f,ms=%.4f,speedup=%.3f,bitexact=%s\n",prop.persistingL2CacheMaxSize/1048576.0,act_bytes/1024.0,cached,ref/cached,exact?"YES":"NO");
        for(int t=0;t<trials;++t)printf("l2_trial,%d,ms=%.4f\n",t,cache_ms[t]);
        cudaStreamAttrValue clear{};CHECK(cudaStreamSetAttribute((cudaStream_t)0,cudaStreamAttributeAccessPolicyWindow,&clear));
    }
    return all_exact ? 0 : 2;
}
