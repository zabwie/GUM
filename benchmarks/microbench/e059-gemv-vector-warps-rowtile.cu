// E059: group four output rows in one CTA; four warps per row each compute four
// vectors, while one cooperative load stages the row weights once. Reuses the
// exact baseline comparator and fixture from E058.
#define main e058_unused_main
#include "e058-gemv-vector-warps.cu"
#undef main

template<int VPW,int ROW_TILE>
__global__ void k_vwarp_rowtile(const uint8_t*w,long K,long rows,const int8_t*xq,const float*xs,const int*xsum,float*y,long ys){
    constexpr int WARPS=16/VPW,THREADS=WARPS*ROW_TILE*32;
    extern __shared__ uint8_t raw[];
    const long nb=K/32;const size_t rb=(size_t)nb*18;float*sd=(float*)raw;uint32_t*sq=(uint32_t*)(sd+(size_t)ROW_TILE*nb);
    for(long i=threadIdx.x;i<(long)ROW_TILE*nb;i+=THREADS){const int rr=(int)(i/nb);const long b=i%nb;const long row=(long)blockIdx.x*ROW_TILE+rr;if(row>=rows){sd[i]=0.f;
#pragma unroll
            for(int j=0;j<4;++j)sq[(size_t)(rr*4+j)*nb+b]=0;continue;}
        const uint16_t*blk=(const uint16_t*)(w+(size_t)row*rb+(size_t)b*18);sd[i]=__half2float(__ushort_as_half(blk[0]));
#pragma unroll
        for(int j=0;j<4;++j)sq[(size_t)(rr*4+j)*nb+b]=(uint32_t)blk[1+2*j]|((uint32_t)blk[2+2*j]<<16);}
    __syncthreads();
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;const int rr=warp/WARPS;const long row=(long)blockIdx.x*ROW_TILE+rr;const int v0=(warp%WARPS)*VPW;
    if(row<rows){float acc[VPW];
#pragma unroll
        for(int u=0;u<VPW;++u)acc[u]=0.f;
        for(long b=lane;b<nb;b+=32){const float d=sd[(size_t)rr*nb+b];const size_t qo=(size_t)rr*4*nb;const uint32_t q0=sq[qo+b],q1=sq[qo+nb+b],q2=sq[qo+2*nb+b],q3=sq[qo+3*nb+b];
            const uint32_t lo0=q0&0x0F0F0F0Fu,lo1=q1&0x0F0F0F0Fu,lo2=q2&0x0F0F0F0Fu,lo3=q3&0x0F0F0F0Fu;const uint32_t hi0=(q0>>4)&0x0F0F0F0Fu,hi1=(q1>>4)&0x0F0F0F0Fu,hi2=(q2>>4)&0x0F0F0F0Fu,hi3=(q3>>4)&0x0F0F0F0Fu;
#pragma unroll
            for(int u=0;u<VPW;++u){const int v=v0+u;const int4*xp=(const int4*)(xq+(size_t)v*K+b*32);const int4 xa=xp[0],xb=xp[1];int s=0;
                s=__dp4a((int)lo0,xa.x,s);s=__dp4a((int)lo1,xa.y,s);s=__dp4a((int)lo2,xa.z,s);s=__dp4a((int)lo3,xa.w,s);s=__dp4a((int)hi0,xb.x,s);s=__dp4a((int)hi1,xb.y,s);s=__dp4a((int)hi2,xb.z,s);s=__dp4a((int)hi3,xb.w,s);acc[u]+=d*xs[(size_t)v*nb+b]*(float)(s-8*xsum[(size_t)v*nb+b]);}}
#pragma unroll
        for(int u=0;u<VPW;++u){float a=acc[u];for(int o=16;o>0;o>>=1)a+=__shfl_down_sync(0xffffffffu,a,o);if(lane==0)y[(size_t)(v0+u)*ys+row]=a;}}
}

int main(){
    constexpr long K=5120,rows=11652;constexpr int NB=16,iters=30,trials=5,VPW=4,ROW_TILE=4;const long nb=K/32;const size_t wb=(size_t)rows*nb*18;
    std::vector<uint8_t>hw(wb);std::vector<int8_t>hx((size_t)NB*K);std::vector<float>hxs((size_t)NB*nb);std::vector<int>hs((size_t)NB*nb);uint32_t seed=1234567;auto rng=[&](){seed=seed*1664525u+1013904223u;return seed;};for(auto&x:hw)x=(uint8_t)(rng()>>24);
    for(long r=0;r<rows;++r)for(long b=0;b<nb;++b)*(uint16_t*)(hw.data()+((size_t)r*nb+b)*18)=host_half_bits(0.001f*(float)((r+b)%97+1));for(auto&x:hx)x=(int8_t)((int)(rng()%31)-15);
    for(int v=0;v<NB;++v)for(long b=0;b<nb;++b){hxs[(size_t)v*nb+b]=0.01f;int s=0;for(int k=0;k<32;++k)s+=hx[(size_t)v*K+b*32+k];hs[(size_t)v*nb+b]=s;}
    uint8_t*dw;int8_t*dx;float*dxs,*y0,*y1;int*ds;CHECK(cudaMalloc(&dw,wb));CHECK(cudaMalloc(&dx,hx.size()));CHECK(cudaMalloc(&dxs,hxs.size()*4));CHECK(cudaMalloc(&ds,hs.size()*4));CHECK(cudaMalloc(&y0,(size_t)NB*rows*4));CHECK(cudaMalloc(&y1,(size_t)NB*rows*4));
    CHECK(cudaMemcpy(dw,hw.data(),wb,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dxs,hxs.data(),hxs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ds,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
    const unsigned gr=(unsigned)((rows+3)/4),gt=(unsigned)((rows+ROW_TILE-1)/ROW_TILE);const size_t smem=(size_t)ROW_TILE*nb*(sizeof(float)+4*sizeof(uint32_t));k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);k_vwarp_rowtile<VPW,ROW_TILE><<<gt,16/VPW*ROW_TILE*32,smem>>>(dw,K,rows,dx,dxs,ds,y1,rows);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<float>h0((size_t)NB*rows),h1(h0.size());CHECK(cudaMemcpy(h0.data(),y0,h0.size()*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(h1.data(),y1,h1.size()*4,cudaMemcpyDeviceToHost));double maxabs=0,maxrel=0;size_t different=0;for(size_t i=0;i<h0.size();++i){double d=fabs((double)h0[i]-h1[i]);maxabs=fmax(maxabs,d);maxrel=fmax(maxrel,d/(fabs((double)h0[i])+1e-12));if(h0[i]!=h1[i])++different;}
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));std::vector<float>ref,tile;for(int t=0;t<trials;++t){CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));ref.push_back(ms/iters);CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_vwarp_rowtile<VPW,ROW_TILE><<<gt,16/VPW*ROW_TILE*32,smem>>>(dw,K,rows,dx,dxs,ds,y1,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));CHECK(cudaEventElapsedTime(&ms,a,b));tile.push_back(ms/iters);}
    auto med=[](std::vector<float>x){std::sort(x.begin(),x.end());return x[x.size()/2];};printf("E059,shape=%ldx%ld,weight_MiB=%.2f,row_tile=%d,vpw=%d,threads=%d,smem_KiB=%.2f,ref_ms=%.4f,tile_ms=%.4f,speedup=%.3f,bitexact_different=%zu,max_abs=%.7g,max_rel=%.7g\n",rows,K,wb/1048576.0,ROW_TILE,VPW,16/VPW*ROW_TILE*32,smem/1024.0,med(ref),med(tile),med(ref)/med(tile),different,maxabs,maxrel);
    for(int i=0;i<trials;++i)printf("trial,%d,ref_ms=%.4f,tile_ms=%.4f\n",i,ref[i],tile[i]);return 0;
}
