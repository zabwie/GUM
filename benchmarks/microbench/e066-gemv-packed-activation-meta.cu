// E058: distribute the 16 activation vectors over warps while staging one row's
// Q4_0 weights once per CTA. Every lane keeps the original ascending K-block
// accumulator order and the original warp reduction; the output check is exact.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#define CHECK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

static uint16_t host_half_bits(float x){const __half h=__float2half_rn(x);uint16_t b;memcpy(&b,&h,2);return b;}

__global__ void k_ref(const uint8_t* w,long K,long rows,const int8_t*xq,const float*xs,const int*xsum,float*y,long ys){
    const int lane=threadIdx.x&31;const long row=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);if(row>=rows)return;
    const long nb=K/32;const size_t rb=(size_t)nb*18;const uint8_t*r=w+(size_t)row*rb;float acc[16];
#pragma unroll
    for(int v=0;v<16;++v)acc[v]=0.f;
    for(long b=lane;b<nb;b+=32){const uint16_t*blk=(const uint16_t*)(r+(size_t)b*18);const float d=__half2float(__ushort_as_half(blk[0]));uint32_t q[4];
#pragma unroll
        for(int j=0;j<4;++j)q[j]=(uint32_t)blk[1+2*j]|((uint32_t)blk[2+2*j]<<16);
        const uint32_t lo0=q[0]&0x0F0F0F0Fu,lo1=q[1]&0x0F0F0F0Fu,lo2=q[2]&0x0F0F0F0Fu,lo3=q[3]&0x0F0F0F0Fu;
        const uint32_t hi0=(q[0]>>4)&0x0F0F0F0Fu,hi1=(q[1]>>4)&0x0F0F0F0Fu,hi2=(q[2]>>4)&0x0F0F0F0Fu,hi3=(q[3]>>4)&0x0F0F0F0Fu;
#pragma unroll
        for(int v=0;v<16;++v){const int4*xp=(const int4*)(xq+(size_t)v*K+b*32);const int4 xa=xp[0],xb=xp[1];int s=0;
            s=__dp4a((int)lo0,xa.x,s);s=__dp4a((int)lo1,xa.y,s);s=__dp4a((int)lo2,xa.z,s);s=__dp4a((int)lo3,xa.w,s);
            s=__dp4a((int)hi0,xb.x,s);s=__dp4a((int)hi1,xb.y,s);s=__dp4a((int)hi2,xb.z,s);s=__dp4a((int)hi3,xb.w,s);
            acc[v]+=d*xs[(size_t)v*nb+b]*(float)(s-8*xsum[(size_t)v*nb+b]);}
    }
#pragma unroll
    for(int v=0;v<16;++v){float a=acc[v];for(int o=16;o>0;o>>=1)a+=__shfl_down_sync(0xffffffffu,a,o);if(lane==0)y[(size_t)v*ys+row]=a;}
}

// E066 candidate: pair each per-quanta scale and sum into one aligned int2 load.
// The FP expression and reduction order match k_ref exactly.
__global__ void k_meta(const uint8_t* w,long K,long rows,const int8_t*xq,const int2*xmeta,float*y,long ys){
    const int lane=threadIdx.x&31;const long row=(long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);if(row>=rows)return;
    const long nb=K/32;const size_t rb=(size_t)nb*18;const uint8_t*r=w+(size_t)row*rb;float acc[16];
#pragma unroll
    for(int v=0;v<16;++v)acc[v]=0.f;
    for(long b=lane;b<nb;b+=32){const uint16_t*blk=(const uint16_t*)(r+(size_t)b*18);const float d=__half2float(__ushort_as_half(blk[0]));uint32_t q[4];
#pragma unroll
        for(int j=0;j<4;++j)q[j]=(uint32_t)blk[1+2*j]|((uint32_t)blk[2+2*j]<<16);
        const uint32_t lo0=q[0]&0x0F0F0F0Fu,lo1=q[1]&0x0F0F0F0Fu,lo2=q[2]&0x0F0F0F0Fu,lo3=q[3]&0x0F0F0F0Fu;
        const uint32_t hi0=(q[0]>>4)&0x0F0F0F0Fu,hi1=(q[1]>>4)&0x0F0F0F0Fu,hi2=(q[2]>>4)&0x0F0F0F0Fu,hi3=(q[3]>>4)&0x0F0F0F0Fu;
#pragma unroll
        for(int v=0;v<16;++v){const int4*xp=(const int4*)(xq+(size_t)v*K+b*32);const int4 xa=xp[0],xb=xp[1];const int2 meta=xmeta[(size_t)v*nb+b];int s=0;
            s=__dp4a((int)lo0,xa.x,s);s=__dp4a((int)lo1,xa.y,s);s=__dp4a((int)lo2,xa.z,s);s=__dp4a((int)lo3,xa.w,s);
            s=__dp4a((int)hi0,xb.x,s);s=__dp4a((int)hi1,xb.y,s);s=__dp4a((int)hi2,xb.z,s);s=__dp4a((int)hi3,xb.w,s);
            acc[v]+=d*__int_as_float(meta.x)*(float)(s-8*meta.y);}
    }
#pragma unroll
    for(int v=0;v<16;++v){float a=acc[v];for(int o=16;o>0;o>>=1)a+=__shfl_down_sync(0xffffffffu,a,o);if(lane==0)y[(size_t)v*ys+row]=a;}
}

template<int VPW>
__global__ void k_vwarp(const uint8_t*w,long K,long rows,const int8_t*xq,const float*xs,const int*xsum,float*y,long ys){
    constexpr int WARPS=16/VPW;
    extern __shared__ uint8_t raw[];
    const long row=(long)blockIdx.x;const long nb=K/32;const size_t rb=(size_t)nb*18;
    float*sd=(float*)raw;uint32_t*sq=(uint32_t*)(sd+nb);
    const uint8_t*wr=w+(size_t)row*rb;
    for(long b=threadIdx.x;b<nb;b+=blockDim.x){const uint16_t*blk=(const uint16_t*)(wr+(size_t)b*18);sd[b]=__half2float(__ushort_as_half(blk[0]));
#pragma unroll
        for(int j=0;j<4;++j)sq[(size_t)j*nb+b]=(uint32_t)blk[1+2*j]|((uint32_t)blk[2+2*j]<<16);}
    __syncthreads();
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;const int v0=warp*VPW;float acc[VPW];
#pragma unroll
    for(int u=0;u<VPW;++u)acc[u]=0.f;
    for(long b=lane;b<nb;b+=32){const float d=sd[b];const uint32_t q0=sq[b],q1=sq[nb+b],q2=sq[2*nb+b],q3=sq[3*nb+b];
        const uint32_t lo0=q0&0x0F0F0F0Fu,lo1=q1&0x0F0F0F0Fu,lo2=q2&0x0F0F0F0Fu,lo3=q3&0x0F0F0F0Fu;
        const uint32_t hi0=(q0>>4)&0x0F0F0F0Fu,hi1=(q1>>4)&0x0F0F0F0Fu,hi2=(q2>>4)&0x0F0F0F0Fu,hi3=(q3>>4)&0x0F0F0F0Fu;
#pragma unroll
        for(int u=0;u<VPW;++u){const int v=v0+u;const int4*xp=(const int4*)(xq+(size_t)v*K+b*32);const int4 xa=xp[0],xb=xp[1];int s=0;
            s=__dp4a((int)lo0,xa.x,s);s=__dp4a((int)lo1,xa.y,s);s=__dp4a((int)lo2,xa.z,s);s=__dp4a((int)lo3,xa.w,s);
            s=__dp4a((int)hi0,xb.x,s);s=__dp4a((int)hi1,xb.y,s);s=__dp4a((int)hi2,xb.z,s);s=__dp4a((int)hi3,xb.w,s);
            acc[u]+=d*xs[(size_t)v*nb+b]*(float)(s-8*xsum[(size_t)v*nb+b]);}}
#pragma unroll
    for(int u=0;u<VPW;++u){float a=acc[u];for(int o=16;o>0;o>>=1)a+=__shfl_down_sync(0xffffffffu,a,o);if(lane==0)y[(size_t)(v0+u)*ys+row]=a;}
}

template<int VPW> static void run_variant(const uint8_t*dw,long K,long rows,const int8_t*dx,const float*dxs,const int*ds,float*y,size_t smem,int iters,int trials,std::vector<float>&times){
    const unsigned threads=(16/VPW)*32;cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
    for(int t=0;t<trials;++t){CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_vwarp<VPW><<<(unsigned)rows,threads,smem>>>(dw,K,rows,dx,dxs,ds,y,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms/iters);}
    CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));
}

int main(int argc,char**argv){
    const long K=argc>1?atol(argv[1]):5120,rows=argc>2?atol(argv[2]):11652;constexpr int NB=16,iters=20,trials=5;const long nb=K/32;const size_t wb=(size_t)rows*nb*18;
    std::vector<uint8_t>hw(wb);std::vector<int8_t>hx((size_t)NB*K);std::vector<float>hxs((size_t)NB*nb);std::vector<int>hs((size_t)NB*nb);std::vector<int2>hm((size_t)NB*nb);
    uint32_t seed=1234567;auto rng=[&](){seed=seed*1664525u+1013904223u;return seed;};for(auto&x:hw)x=(uint8_t)(rng()>>24);
    for(long r=0;r<rows;++r)for(long b=0;b<nb;++b)*(uint16_t*)(hw.data()+((size_t)r*nb+b)*18)=host_half_bits(0.001f*(float)((r+b)%97+1));
    for(auto&x:hx)x=(int8_t)((int)(rng()%31)-15);
    for(int v=0;v<NB;++v)for(long b=0;b<nb;++b){hxs[(size_t)v*nb+b]=0.01f;int s=0;for(int k=0;k<32;++k)s+=hx[(size_t)v*K+b*32+k];hs[(size_t)v*nb+b]=s;memcpy(&hm[(size_t)v*nb+b].x,&hxs[(size_t)v*nb+b],sizeof(float));hm[(size_t)v*nb+b].y=s;}
    uint8_t*dw;int8_t*dx;float*dxs,*y0,*y1;int*ds;int2*dm;CHECK(cudaMalloc(&dw,wb));CHECK(cudaMalloc(&dx,hx.size()));CHECK(cudaMalloc(&dxs,hxs.size()*4));CHECK(cudaMalloc(&ds,hs.size()*4));CHECK(cudaMalloc(&dm,hm.size()*sizeof(int2)));CHECK(cudaMalloc(&y0,(size_t)NB*rows*4));CHECK(cudaMalloc(&y1,(size_t)NB*rows*4));
    CHECK(cudaMemcpy(dw,hw.data(),wb,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dxs,hxs.data(),hxs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ds,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dm,hm.data(),hm.size()*sizeof(int2),cudaMemcpyHostToDevice));
    const unsigned gr=(unsigned)((rows+3)/4);k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);CHECK(cudaDeviceSynchronize());
    std::vector<float>h0((size_t)NB*rows),h1(h0.size());CHECK(cudaMemcpy(h0.data(),y0,h0.size()*4,cudaMemcpyDeviceToHost));
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));std::vector<float>ref;
    for(int t=0;t<trials;++t){CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));ref.push_back(ms/iters);}
    std::vector<float>tmeta; k_meta<<<gr,128>>>(dw,K,rows,dx,dm,y1,rows);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h1.data(),y1,h1.size()*4,cudaMemcpyDeviceToHost));double meta_maxabs=0;size_t meta_different=0;for(size_t i=0;i<h0.size();++i){if(h0[i]!=h1[i])++meta_different;meta_maxabs=fmax(meta_maxabs,fabs((double)h0[i]-h1[i]));}
    for(int t=0;t<trials;++t){CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_meta<<<gr,128>>>(dw,K,rows,dx,dm,y1,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));tmeta.push_back(ms/iters);}
    auto med=[](std::vector<float>x){std::sort(x.begin(),x.end());return x[x.size()/2];};const size_t smem=(size_t)nb*(sizeof(float)+4*sizeof(uint32_t));
    std::vector<float>t16,t8,t4,t2,t1;run_variant<16>(dw,K,rows,dx,dxs,ds,y1,smem,iters,trials,t16);run_variant<8>(dw,K,rows,dx,dxs,ds,y1,smem,iters,trials,t8);run_variant<4>(dw,K,rows,dx,dxs,ds,y1,smem,iters,trials,t4);run_variant<2>(dw,K,rows,dx,dxs,ds,y1,smem,iters,trials,t2);run_variant<1>(dw,K,rows,dx,dxs,ds,y1,smem,iters,trials,t1);
    CHECK(cudaMemcpy(h1.data(),y1,h1.size()*4,cudaMemcpyDeviceToHost));
    double maxabs=0,maxrel=0;size_t different=0;for(size_t i=0;i<h0.size();++i){double d=fabs((double)h0[i]-h1[i]);maxabs=fmax(maxabs,d);maxrel=fmax(maxrel,d/(fabs((double)h0[i])+1e-12));if(h0[i]!=h1[i])++different;}
    // Last kernel was VPW=1; all variants share the same exact arithmetic path.
    printf("E066,shape=%ldx%ld,weight_MiB=%.2f,ref_ms=%.4f,meta_ms=%.4f,speedup=%.3f,bitexact_different=%zu,max_abs=%.7g\n",rows,K,wb/1048576.0,med(ref),med(tmeta),med(ref)/med(tmeta),meta_different,meta_maxabs);
    printf("E058,shape=%ldx%ld,weight_MiB=%.2f,ref_ms=%.4f,smem_KiB=%.2f,bitexact_different=%zu,max_abs=%.7g,max_rel=%.7g\n",rows,K,wb/1048576.0,med(ref),smem/1024.0,different,maxabs,maxrel);
    printf("variant,vpw=16,warps=1,threads=32,ms=%.4f,speedup=%.3f\n",med(t16),med(ref)/med(t16));printf("variant,vpw=8,warps=2,threads=64,ms=%.4f,speedup=%.3f\n",med(t8),med(ref)/med(t8));printf("variant,vpw=4,warps=4,threads=128,ms=%.4f,speedup=%.3f\n",med(t4),med(ref)/med(t4));printf("variant,vpw=2,warps=8,threads=256,ms=%.4f,speedup=%.3f\n",med(t2),med(ref)/med(t2));printf("variant,vpw=1,warps=16,threads=512,ms=%.4f,speedup=%.3f\n",med(t1),med(ref)/med(t1));
    return 0;
}
