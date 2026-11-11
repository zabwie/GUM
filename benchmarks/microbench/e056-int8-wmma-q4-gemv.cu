// E056 probe: use INT8 WMMA/Tensor Cores for the NB=16 Q4_0 GEMV.
// For each 32-value Q4 block, unpack signed q-8 weights and multiply them by the
// already-quantized activation block. INT32 block dots are exact; FP32 block scaling
// accumulates in a fixed K-block order. This changes the old dp4a reduction order, so
// the production token gate remains the acceptance criterion before integration.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#define CHECK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

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
            acc[v]+=d*xs[(size_t)v*nb+b]*(float)(s-8*xsum[(size_t)v*nb+b]);
        }
    }
#pragma unroll
    for(int v=0;v<16;++v){float a=acc[v];for(int o=16;o>0;o>>=1)a+=__shfl_down_sync(0xffffffffu,a,o);if(lane==0)y[(size_t)v*ys+row]=a;}
}

__global__ void k_tc(const uint8_t*w,long K,long rows,const int8_t*xq,const float*xs,float*y,long ys){
    extern __shared__ __align__(32) uint8_t sm[];
    uint8_t*raw=sm;int8_t*A=(int8_t*)(sm+320);int8_t*B=A+512;int32_t*C=(int32_t*)(B+512);float*wd=(float*)(C+256);float*xd=wd+16;
    const int lane=threadIdx.x&31;const long row0=(long)blockIdx.x*16;const long nb=K/32;const size_t rb=(size_t)nb*18;
    float out_acc[8];
#pragma unroll
    for(int u=0;u<8;++u)out_acc[u]=0.f;
    using namespace nvcuda;
    wmma::fragment<wmma::accumulator,16,16,16,int> cf;
    wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> a0,a1;
    wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::col_major> b0,b1;
    for(long b=0;b<nb;++b){
        for(int i=lane;i<16*18;i+=32){const int r=i/18,byte=i%18;const long gr=row0+r;raw[i]=(gr<rows)?w[(size_t)gr*rb+(size_t)b*18+byte]:0;}
        for(int i=lane;i<16*32;i+=32){const int r=i/32,k=i%32;const long gr=row0+r;const uint8_t q=gr<rows?raw[r*18+2+(k&15)]:0;const int nib=(k<16)?(q&15):(q>>4);A[i]=(gr<rows)?(int8_t)(nib-8):0;}
        for(int i=lane;i<16*32;i+=32){const int v=i/32,k=i%32;B[i]=xq[(size_t)v*K+(size_t)b*32+k];}
        if(lane<16){const int r=lane,v=lane;wd[r]=row0+r<rows?__half2float(__ushort_as_half(*(const uint16_t*)(raw+r*18))):0.f;xd[v]=xs[(size_t)v*nb+b];}
        __syncwarp();
        wmma::fill_fragment(cf,0);
        wmma::load_matrix_sync(a0,A,32);wmma::load_matrix_sync(a1,A+16,32);
        wmma::load_matrix_sync(b0,B,32);wmma::load_matrix_sync(b1,B+16,32);
        wmma::mma_sync(cf,a0,b0,cf,false);wmma::mma_sync(cf,a1,b1,cf,false);
        wmma::store_matrix_sync(C,cf,16,wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for(int u=0;u<8;++u){const int i=lane+32*u;const int r=i/16,v=i%16;out_acc[u]+=(float)C[i]*wd[r]*xd[v];}
        __syncwarp();
    }
#pragma unroll
    for(int u=0;u<8;++u){const int i=lane+32*u;const long r=row0+i/16,v=i%16;if(r<rows)y[(size_t)v*ys+r]=out_acc[u];}
}

int main(){
    constexpr long K=5120,rows=11652;constexpr int NB=16,iters=30,trials=3;const long nb=K/32;const size_t wb=(size_t)rows*nb*18;
    std::vector<uint8_t>hw(wb);std::vector<int8_t>hx((size_t)NB*K);std::vector<float>hxs((size_t)NB*nb);std::vector<int>hs((size_t)NB*nb);
    uint32_t seed=1234567;auto rng=[&](){seed=seed*1664525u+1013904223u;return seed;};for(auto&x:hw)x=(uint8_t)(rng()>>24);
    for(long r=0;r<rows;++r)for(long b=0;b<nb;++b)*(uint16_t*)(hw.data()+((size_t)r*nb+b)*18)=host_half_bits(0.001f*(float)((r+b)%97+1));
    for(auto&x:hx)x=(int8_t)((int)(rng()%31)-15);
    for(int v=0;v<NB;++v)for(long qb=0;qb<nb;++qb){hxs[(size_t)v*nb+qb]=0.01f;int s=0;for(int k=0;k<32;++k)s+=hx[(size_t)v*K+qb*32+k];hs[(size_t)v*nb+qb]=s;}
    uint8_t*dw;int8_t*dx;float*dxs,*y0,*y1;int*ds;CHECK(cudaMalloc(&dw,wb));CHECK(cudaMalloc(&dx,hx.size()));CHECK(cudaMalloc(&dxs,hxs.size()*4));CHECK(cudaMalloc(&ds,hs.size()*4));CHECK(cudaMalloc(&y0,(size_t)NB*rows*4));CHECK(cudaMalloc(&y1,(size_t)NB*rows*4));
    CHECK(cudaMemcpy(dw,hw.data(),wb,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dxs,hxs.data(),hxs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ds,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
    const unsigned gr=(unsigned)((rows+3)/4),gt=(unsigned)((rows+15)/16);const size_t smem=2496;
    k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);k_tc<<<gt,32,smem>>>(dw,K,rows,dx,dxs,y1,rows);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<float>h0((size_t)NB*rows),h1(h0.size());CHECK(cudaMemcpy(h0.data(),y0,h0.size()*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(h1.data(),y1,h1.size()*4,cudaMemcpyDeviceToHost));
    double maxabs=0,maxrel=0;size_t different=0;for(size_t i=0;i<h0.size();++i){double d=fabs((double)h0[i]-h1[i]);if(d>maxabs)maxabs=d;double rel=d/(fabs((double)h0[i])+1e-12);if(rel>maxrel)maxrel=rel;if(h0[i]!=h1[i])++different;}
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));std::vector<float>ref,tc;
    for(int t=0;t<trials;++t){CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));ref.push_back(ms/iters);
        CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_tc<<<gt,32,smem>>>(dw,K,rows,dx,dxs,y1,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));CHECK(cudaEventElapsedTime(&ms,a,b));tc.push_back(ms/iters);}
    auto med=[](std::vector<float>x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    printf("E056,shape=%ldx%ld,weight_MiB=%.2f,ref_ms=%.4f,tc_ms=%.4f,speedup=%.3f,bitexact_different=%zu,max_abs=%.7g,max_rel=%.7g\n",rows,K,wb/1048576.0,med(ref),med(tc),med(ref)/med(tc),different,maxabs,maxrel);
    for(int i=0;i<trials;++i)printf("trial,%d,ref_ms=%.4f,tc_ms=%.4f\n",i,ref[i],tc[i]);
    return 0;
}
