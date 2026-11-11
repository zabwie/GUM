// E053: test bypassing L1 for the one-pass Q4 weight stream. The NB=16 activation
// vectors are repeatedly reused while each output row consumes a unique weight row;
// default cacheable weight traffic may evict the 100 KiB activation working set.
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
    uint16_t bits; memcpy(&bits,&h,sizeof(bits)); return bits;
}

__device__ __forceinline__ uint16_t load_cg_u16(const uint16_t* p) {
    uint16_t v; asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
template <bool CG_WEIGHT, bool LDG_ACT>
__global__ void k_mat(const uint8_t* w,long K,long rows,const int8_t* xq,const float* xs,const int* xsum,float* y,long ys) {
    const int lane=threadIdx.x&31;
    const long row=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
    if(row>=rows)return;
    const long nb=K/32; const size_t rowbytes=(size_t)nb*18;
    const uint8_t* r=w+(size_t)row*rowbytes;
    float acc[16];
#pragma unroll
    for(int v=0;v<16;++v)acc[v]=0.f;
    for(long b=lane;b<nb;b+=32){
        const uint16_t* blk=(const uint16_t*)(r+(size_t)b*18);
        const float d=CG_WEIGHT?__half2float(__ushort_as_half(load_cg_u16(blk))):__half2float(__ushort_as_half(blk[0]));
        uint32_t q[4];
        if(CG_WEIGHT){
#pragma unroll
            for(int j=0;j<4;++j)q[j]=(uint32_t)load_cg_u16(blk+1+2*j)|((uint32_t)load_cg_u16(blk+2+2*j)<<16);
        } else {
#pragma unroll
            for(int j=0;j<4;++j)q[j]=(uint32_t)blk[1+2*j]|((uint32_t)blk[2+2*j]<<16);
        }
        const uint32_t lo0=q[0]&0x0F0F0F0Fu,lo1=q[1]&0x0F0F0F0Fu,lo2=q[2]&0x0F0F0F0Fu,lo3=q[3]&0x0F0F0F0Fu;
        const uint32_t hi0=(q[0]>>4)&0x0F0F0F0Fu,hi1=(q[1]>>4)&0x0F0F0F0Fu,hi2=(q[2]>>4)&0x0F0F0F0Fu,hi3=(q[3]>>4)&0x0F0F0F0Fu;
#pragma unroll
        for(int v=0;v<16;++v){
            const int4* xp=(const int4*)(xq+(size_t)v*K+b*32);
            const int4 xa=LDG_ACT?__ldg(xp):xp[0], xb=LDG_ACT?__ldg(xp+1):xp[1];
            int s=0;
            s=__dp4a((int)lo0,xa.x,s);s=__dp4a((int)lo1,xa.y,s);s=__dp4a((int)lo2,xa.z,s);s=__dp4a((int)lo3,xa.w,s);
            s=__dp4a((int)hi0,xb.x,s);s=__dp4a((int)hi1,xb.y,s);s=__dp4a((int)hi2,xb.z,s);s=__dp4a((int)hi3,xb.w,s);
            const size_t ai=(size_t)v*nb+b;
            const float xscale=LDG_ACT?__ldg(xs+ai):xs[ai];
            const int xsumv=LDG_ACT?__ldg(xsum+ai):xsum[ai];
            acc[v]+=d*xscale*(float)(s-8*xsumv);
        }
    }
#pragma unroll
    for(int v=0;v<16;++v){float a=acc[v];for(int o=16;o>0;o>>=1)a+=__shfl_down_sync(0xffffffffu,a,o);if(lane==0)y[(size_t)v*ys+row]=a;}
}

int main(){
    constexpr long K=5120,rows=11652;constexpr int NB=16,iters=40,trials=3;
    const long nb=K/32;const size_t wb=(size_t)rows*nb*18;
    std::vector<uint8_t> hw(wb);std::vector<int8_t> hx((size_t)NB*K);std::vector<float> hxs((size_t)NB*nb);std::vector<int> hs((size_t)NB*nb);
    uint32_t seed=987654321;auto rng=[&](){seed=seed*1664525u+1013904223u;return seed;};
    for(auto&x:hw)x=(uint8_t)(rng()>>24);
    for(long r=0;r<rows;++r)for(long b=0;b<nb;++b)*(uint16_t*)(hw.data()+((size_t)r*nb+b)*18)=host_half_bits(0.001f*(float)((r+b)%97+1));
    for(auto&x:hx)x=(int8_t)((int)(rng()%31)-15);
    for(size_t i=0;i<hxs.size();++i){hxs[i]=0.01f;hs[i]=(int)(rng()%7)-3;}
    uint8_t*dw;int8_t*dx;float*dxs,*y0,*y1;int*ds;CHECK(cudaMalloc(&dw,wb));CHECK(cudaMalloc(&dx,hx.size()));CHECK(cudaMalloc(&dxs,hxs.size()*4));CHECK(cudaMalloc(&ds,hs.size()*4));CHECK(cudaMalloc(&y0,(size_t)NB*rows*4));CHECK(cudaMalloc(&y1,(size_t)NB*rows*4));
    CHECK(cudaMemcpy(dw,hw.data(),wb,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dxs,hxs.data(),hxs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ds,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
    const unsigned grid=(unsigned)((rows+3)/4);std::vector<float> ref,cg;
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
    auto launch=[&](int cfg,float* out){
        if(cfg==0)k_mat<false,false><<<grid,128>>>(dw,K,rows,dx,dxs,ds,out,rows);
        else if(cfg==1)k_mat<false,true><<<grid,128>>>(dw,K,rows,dx,dxs,ds,out,rows);
        else if(cfg==2)k_mat<true,false><<<grid,128>>>(dw,K,rows,dx,dxs,ds,out,rows);
        else k_mat<true,true><<<grid,128>>>(dw,K,rows,dx,dxs,ds,out,rows);
    };
    std::vector<float> times[4];
    for(int t=0;t<trials;++t){
        launch(0,y0);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
        std::vector<float> h0((size_t)NB*rows),h1((size_t)NB*rows);CHECK(cudaMemcpy(h0.data(),y0,h0.size()*4,cudaMemcpyDeviceToHost));
        for(int cfg=1;cfg<4;++cfg){
            launch(cfg,y1);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(h1.data(),y1,h1.size()*4,cudaMemcpyDeviceToHost));
            const bool exact=memcmp(h0.data(),h1.data(),h0.size()*4)==0;
            if(!exact){for(size_t i=0;i<h0.size();++i)if(h0[i]!=h1[i]){printf("first_diff,cfg=%d,index=%zu,ref=%a,candidate=%a\n",cfg,i,h0[i],h1[i]);break;}return 2;}
        }
        for(int cfg=0;cfg<4;++cfg){
            CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)launch(cfg,cfg?y1:y0);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));times[cfg].push_back(ms/iters);
        }
        printf("trial,%d,default_ms=%.4f,ldg_activations_ms=%.4f,cg_weight_ms=%.4f,cg_weight_ldg_act_ms=%.4f,bitexact=YES\n",t,times[0].back(),times[1].back(),times[2].back(),times[3].back());
    }
    auto med=[](std::vector<float>x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    printf("E053,shape=%ldx%ld,weight_MiB=%.2f,default_median_ms=%.4f,ldg_activations_median_ms=%.4f,cg_weight_median_ms=%.4f,cg_weight_ldg_act_median_ms=%.4f,ldg_speedup=%.3f\n",rows,K,wb/1048576.0,med(times[0]),med(times[1]),med(times[2]),med(times[3]),med(times[0])/med(times[1]));
    return 0;
}
