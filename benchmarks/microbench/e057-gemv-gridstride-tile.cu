// E057: stage one half-K activation/scale tile, reuse it over grid-stride row groups.
// Partials are written in a fixed chunk order with no atomics. The changed FP
// reduction order is checked against the production kernel and the full decoder gate.
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

template<int WARPS,int ROW_GROUPS,int CB>
__global__ void k_tile(const uint8_t*w,long K,long rows,const int8_t*xq,const float*xs,const int*xsum,float*y,long ys){
    constexpr int NB=16,THREADS=WARPS*32,ROW_TILE=WARPS*ROW_GROUPS;
    extern __shared__ uint8_t smem_raw[];
    uint32_t*tq=(uint32_t*)smem_raw;                 // [NB][8 words][CB], b is contiguous
    float*txs=(float*)(tq+(size_t)NB*8*CB);          // [NB][CB]
    int16_t*txsum=(int16_t*)(txs+(size_t)NB*CB);     // [NB][CB], sums fit signed 16-bit
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const long nb=K/32;const size_t rb=(size_t)nb*18;
    for(long c0=0,chunk=0;c0<nb;c0+=CB,++chunk){
        const int cb=(int)min((long)CB,nb-c0);
        for(int i=tid;i<NB*cb;i+=THREADS){const int v=i/cb,b=i%cb;const int4*p=(const int4*)(xq+(size_t)v*K+(size_t)(c0+b)*32);const int4 a=p[0],z=p[1];
            tq[((size_t)v*8+0)*CB+b]=(uint32_t)a.x;tq[((size_t)v*8+1)*CB+b]=(uint32_t)a.y;tq[((size_t)v*8+2)*CB+b]=(uint32_t)a.z;tq[((size_t)v*8+3)*CB+b]=(uint32_t)a.w;
            tq[((size_t)v*8+4)*CB+b]=(uint32_t)z.x;tq[((size_t)v*8+5)*CB+b]=(uint32_t)z.y;tq[((size_t)v*8+6)*CB+b]=(uint32_t)z.z;tq[((size_t)v*8+7)*CB+b]=(uint32_t)z.w;
            txs[(size_t)v*CB+b]=xs[(size_t)v*nb+c0+b];txsum[(size_t)v*CB+b]=(int16_t)xsum[(size_t)v*nb+c0+b];}
        __syncthreads();
        for(long rowbase=(long)blockIdx.x*ROW_TILE;rowbase<rows;rowbase+=(long)gridDim.x*ROW_TILE){
#pragma unroll
            for(int rg=0;rg<ROW_GROUPS;++rg){const long row=rowbase+(long)warp*ROW_GROUPS+rg;if(row>=rows)continue;const uint8_t*r=w+(size_t)row*rb;float acc[NB];
#pragma unroll
                for(int v=0;v<NB;++v)acc[v]=0.f;
                for(int b=lane;b<cb;b+=32){const uint16_t*blk=(const uint16_t*)(r+(size_t)(c0+b)*18);const float d=__half2float(__ushort_as_half(blk[0]));uint32_t q[4];
#pragma unroll
                    for(int j=0;j<4;++j)q[j]=(uint32_t)blk[1+2*j]|((uint32_t)blk[2+2*j]<<16);
                    const uint32_t lo0=q[0]&0x0F0F0F0Fu,lo1=q[1]&0x0F0F0F0Fu,lo2=q[2]&0x0F0F0F0Fu,lo3=q[3]&0x0F0F0F0Fu;
                    const uint32_t hi0=(q[0]>>4)&0x0F0F0F0Fu,hi1=(q[1]>>4)&0x0F0F0F0Fu,hi2=(q[2]>>4)&0x0F0F0F0Fu,hi3=(q[3]>>4)&0x0F0F0F0Fu;
#pragma unroll
                    for(int v=0;v<NB;++v){const size_t o=(size_t)v*CB+b;int s=0;
                        s=__dp4a((int)lo0,(int)tq[((size_t)v*8+0)*CB+b],s);s=__dp4a((int)lo1,(int)tq[((size_t)v*8+1)*CB+b],s);s=__dp4a((int)lo2,(int)tq[((size_t)v*8+2)*CB+b],s);s=__dp4a((int)lo3,(int)tq[((size_t)v*8+3)*CB+b],s);
                        s=__dp4a((int)hi0,(int)tq[((size_t)v*8+4)*CB+b],s);s=__dp4a((int)hi1,(int)tq[((size_t)v*8+5)*CB+b],s);s=__dp4a((int)hi2,(int)tq[((size_t)v*8+6)*CB+b],s);s=__dp4a((int)hi3,(int)tq[((size_t)v*8+7)*CB+b],s);
                        acc[v]+=d*txs[o]*(float)(s-8*(int)txsum[o]);}
                }
#pragma unroll
                for(int v=0;v<NB;++v){float a=acc[v];for(int o=16;o>0;o>>=1)a+=__shfl_down_sync(0xffffffffu,a,o);if(lane==0){float*dst=y+(size_t)v*ys+row;if(chunk==0)*dst=a;else *dst+=a;}}
            }
        }
        __syncthreads();
    }
}

int main(){
    constexpr long K=5120,rows=11652;constexpr int NB=16,iters=30,trials=5;constexpr int WARPS=8,ROW_GROUPS=4,CB=80;
    const long nb=K/32;const size_t wb=(size_t)rows*nb*18;std::vector<uint8_t>hw(wb);std::vector<int8_t>hx((size_t)NB*K);std::vector<float>hxs((size_t)NB*nb);std::vector<int>hs((size_t)NB*nb);
    uint32_t seed=1234567;auto rng=[&](){seed=seed*1664525u+1013904223u;return seed;};for(auto&x:hw)x=(uint8_t)(rng()>>24);
    for(long r=0;r<rows;++r)for(long b=0;b<nb;++b)*(uint16_t*)(hw.data()+((size_t)r*nb+b)*18)=host_half_bits(0.001f*(float)((r+b)%97+1));
    for(auto&x:hx)x=(int8_t)((int)(rng()%31)-15);
    for(int v=0;v<NB;++v)for(long b=0;b<nb;++b){hxs[(size_t)v*nb+b]=0.01f;int s=0;for(int k=0;k<32;++k)s+=hx[(size_t)v*K+b*32+k];hs[(size_t)v*nb+b]=s;}
    uint8_t*dw;int8_t*dx;float*dxs,*y0,*y1;int*ds;CHECK(cudaMalloc(&dw,wb));CHECK(cudaMalloc(&dx,hx.size()));CHECK(cudaMalloc(&dxs,hxs.size()*4));CHECK(cudaMalloc(&ds,hs.size()*4));CHECK(cudaMalloc(&y0,(size_t)NB*rows*4));CHECK(cudaMalloc(&y1,(size_t)NB*rows*4));
    CHECK(cudaMemcpy(dw,hw.data(),wb,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dxs,hxs.data(),hxs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ds,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
    const unsigned gr=(unsigned)((rows+3)/4),gt=(unsigned)((rows+WARPS*ROW_GROUPS-1)/(WARPS*ROW_GROUPS));const size_t smem=(size_t)NB*8*CB*4+(size_t)NB*CB*4+(size_t)NB*CB*2;
    CHECK(cudaFuncSetAttribute(k_tile<WARPS,ROW_GROUPS,CB>,cudaFuncAttributeMaxDynamicSharedMemorySize,99*1024));
    k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);k_tile<WARPS,ROW_GROUPS,CB><<<gt,WARPS*32,smem>>>(dw,K,rows,dx,dxs,ds,y1,rows);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<float>h0((size_t)NB*rows),h1(h0.size());CHECK(cudaMemcpy(h0.data(),y0,h0.size()*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(h1.data(),y1,h1.size()*4,cudaMemcpyDeviceToHost));
    double maxabs=0,maxrel=0;size_t different=0;for(size_t i=0;i<h0.size();++i){double d=fabs((double)h0[i]-h1[i]);maxabs=fmax(maxabs,d);maxrel=fmax(maxrel,d/(fabs((double)h0[i])+1e-12));if(h0[i]!=h1[i])++different;}
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));std::vector<float>ref,tile;
    for(int t=0;t<trials;++t){CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));ref.push_back(ms/iters);
        CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_tile<WARPS,ROW_GROUPS,CB><<<gt,WARPS*32,smem>>>(dw,K,rows,dx,dxs,ds,y1,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));CHECK(cudaEventElapsedTime(&ms,a,b));tile.push_back(ms/iters);}
    auto med=[](std::vector<float>x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    printf("E057,shape=%ldx%ld,weight_MiB=%.2f,warps=%d,row_groups=%d,chunk_blocks=%d,smem_KiB=%.1f,ref_ms=%.4f,tile_ms=%.4f,speedup=%.3f,bitexact_different=%zu,max_abs=%.7g,max_rel=%.7g\n",rows,K,wb/1048576.0,WARPS,ROW_GROUPS,CB,smem/1024.0,med(ref),med(tile),med(ref)/med(tile),different,maxabs,maxrel);
    for(int i=0;i<trials;++i)printf("trial,%d,ref_ms=%.4f,tile_ms=%.4f\n",i,ref[i],tile[i]);
    return 0;
}
