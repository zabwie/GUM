// E060: shared activation/scale tile plus four vectors per warp and row-grid
// reuse. Partial outputs are accumulated in ascending K-chunk order, with no
// atomics. Compare against the production kernel and then the decoder token gate.
#define main e057_unused_main
#include "e057-gemv-gridstride-tile.cu"
#undef main

template<int ROW_GROUPS,int CB>
__global__ void k_staged_vector16(const uint8_t*w,long K,long rows,const int8_t*xq,const float*xs,const int*xsum,float*y,long ys){
    constexpr int NB=16,VPW=4,WARPS=NB/VPW,THREADS=WARPS*4*32,ROW_TILE=WARPS*ROW_GROUPS;
    extern __shared__ uint8_t raw[];
    uint32_t*txq=(uint32_t*)raw;float*txs=(float*)(txq+(size_t)NB*8*CB);int16_t*txsum=(int16_t*)(txs+(size_t)NB*CB);
    float*twscale=(float*)(txsum+(size_t)NB*CB);uint32_t*twq=(uint32_t*)(twscale+(size_t)WARPS*CB);
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,row_in_group=warp/WARPS,vgroup=warp%WARPS;const long nb=K/32;
    for(long c0=0,chunk=0;c0<nb;c0+=CB,++chunk){const int cb=(int)min((long)CB,nb-c0);
        for(int i=tid;i<NB*cb;i+=THREADS){const int v=i/cb,b=i%cb;const int4*p=(const int4*)(xq+(size_t)v*K+(size_t)(c0+b)*32);const int4 a=p[0],z=p[1];
            txq[((size_t)v*8+0)*CB+b]=(uint32_t)a.x;txq[((size_t)v*8+1)*CB+b]=(uint32_t)a.y;txq[((size_t)v*8+2)*CB+b]=(uint32_t)a.z;txq[((size_t)v*8+3)*CB+b]=(uint32_t)a.w;
            txq[((size_t)v*8+4)*CB+b]=(uint32_t)z.x;txq[((size_t)v*8+5)*CB+b]=(uint32_t)z.y;txq[((size_t)v*8+6)*CB+b]=(uint32_t)z.z;txq[((size_t)v*8+7)*CB+b]=(uint32_t)z.w;
            txs[(size_t)v*CB+b]=xs[(size_t)v*nb+c0+b];txsum[(size_t)v*CB+b]=(int16_t)xsum[(size_t)v*nb+c0+b];}
        __syncthreads();
        for(long rowbase=(long)blockIdx.x*ROW_TILE;rowbase<rows;rowbase+=(long)gridDim.x*ROW_TILE){
            for(int rg=0;rg<ROW_GROUPS;++rg){const long row0=rowbase+(long)rg*WARPS;
                for(int i=tid;i<WARPS*cb;i+=THREADS){const int rr=i/cb,b=i%cb;const long row=row0+rr;if(row>=rows){twscale[(size_t)rr*CB+b]=0.f;
#pragma unroll
                        for(int j=0;j<4;++j)twq[((size_t)rr*4+j)*CB+b]=0;continue;}
                    const uint16_t*blk=(const uint16_t*)(w+(size_t)row*nb*18+(size_t)(c0+b)*18);twscale[(size_t)rr*CB+b]=__half2float(__ushort_as_half(blk[0]));
#pragma unroll
                    for(int j=0;j<4;++j)twq[((size_t)rr*4+j)*CB+b]=(uint32_t)blk[1+2*j]|((uint32_t)blk[2+2*j]<<16);}
                __syncthreads();
                const long row=row0+row_in_group;if(row<rows){float acc[VPW];
#pragma unroll
                    for(int u=0;u<VPW;++u)acc[u]=0.f;
                    for(int b=lane;b<cb;b+=32){const float d=twscale[(size_t)row_in_group*CB+b];const size_t qo=(size_t)row_in_group*4*CB+b;const uint32_t q0=twq[qo],q1=twq[qo+CB],q2=twq[qo+2*CB],q3=twq[qo+3*CB];
                        const uint32_t lo0=q0&0x0F0F0F0Fu,lo1=q1&0x0F0F0F0Fu,lo2=q2&0x0F0F0F0Fu,lo3=q3&0x0F0F0F0Fu;const uint32_t hi0=(q0>>4)&0x0F0F0F0Fu,hi1=(q1>>4)&0x0F0F0F0Fu,hi2=(q2>>4)&0x0F0F0F0Fu,hi3=(q3>>4)&0x0F0F0F0Fu;
#pragma unroll
                        for(int u=0;u<VPW;++u){const int v=vgroup*VPW+u;const size_t xo=(size_t)v*8*CB+b;int s=0;
                            s=__dp4a((int)lo0,(int)txq[xo],s);s=__dp4a((int)lo1,(int)txq[xo+CB],s);s=__dp4a((int)lo2,(int)txq[xo+2*CB],s);s=__dp4a((int)lo3,(int)txq[xo+3*CB],s);
                            s=__dp4a((int)hi0,(int)txq[xo+4*CB],s);s=__dp4a((int)hi1,(int)txq[xo+5*CB],s);s=__dp4a((int)hi2,(int)txq[xo+6*CB],s);s=__dp4a((int)hi3,(int)txq[xo+7*CB],s);
                            const size_t so=(size_t)v*CB+b;acc[u]+=d*txs[so]*(float)(s-8*(int)txsum[so]);}}
#pragma unroll
                    for(int u=0;u<VPW;++u){float a=acc[u];for(int o=16;o>0;o>>=1)a+=__shfl_down_sync(0xffffffffu,a,o);if(lane==0){float*dst=y+(size_t)(vgroup*VPW+u)*ys+row;if(chunk==0)*dst=a;else *dst+=a;}}}
                __syncthreads();
            }
        }
    }
}

template<int RG>static void run_rg(const uint8_t*dw,long K,long rows,const int8_t*dx,const float*dxs,const int*ds,float*y,size_t smem,int iters,int trials,std::vector<float>&times){const unsigned threads=(16/4)*4*32,grid=(unsigned)((rows+4*RG-1)/(4*RG));CHECK(cudaFuncSetAttribute(k_staged_vector16<RG,80>,cudaFuncAttributeMaxDynamicSharedMemorySize,99*1024));cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));for(int t=0;t<trials;++t){CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_staged_vector16<RG,80><<<grid,threads,smem>>>(dw,K,rows,dx,dxs,ds,y,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms/iters);}CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));}

int main(){
    constexpr long K=5120,rows=11652;constexpr int NB=16,iters=15,trials=5;const long nb=K/32;const size_t wb=(size_t)rows*nb*18;std::vector<uint8_t>hw(wb);std::vector<int8_t>hx((size_t)NB*K);std::vector<float>hxs((size_t)NB*nb);std::vector<int>hs((size_t)NB*nb);uint32_t seed=1234567;auto rng=[&](){seed=seed*1664525u+1013904223u;return seed;};for(auto&x:hw)x=(uint8_t)(rng()>>24);
    for(long r=0;r<rows;++r)for(long b=0;b<nb;++b)*(uint16_t*)(hw.data()+((size_t)r*nb+b)*18)=host_half_bits(0.001f*(float)((r+b)%97+1));for(auto&x:hx)x=(int8_t)((int)(rng()%31)-15);for(int v=0;v<NB;++v)for(long b=0;b<nb;++b){hxs[(size_t)v*nb+b]=0.01f;int s=0;for(int k=0;k<32;++k)s+=hx[(size_t)v*K+b*32+k];hs[(size_t)v*nb+b]=s;}
    uint8_t*dw;int8_t*dx;float*dxs,*y0,*y1;int*ds;CHECK(cudaMalloc(&dw,wb));CHECK(cudaMalloc(&dx,hx.size()));CHECK(cudaMalloc(&dxs,hxs.size()*4));CHECK(cudaMalloc(&ds,hs.size()*4));CHECK(cudaMalloc(&y0,(size_t)NB*rows*4));CHECK(cudaMalloc(&y1,(size_t)NB*rows*4));CHECK(cudaMemcpy(dw,hw.data(),wb,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dxs,hxs.data(),hxs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ds,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
    const unsigned gr=(unsigned)((rows+3)/4);k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);CHECK(cudaDeviceSynchronize());std::vector<float>h0((size_t)NB*rows),h1(h0.size());CHECK(cudaMemcpy(h0.data(),y0,h0.size()*4,cudaMemcpyDeviceToHost));
    const size_t smem=(size_t)NB*80*(8*4+4+2)+(size_t)4*80*(4+4*4);std::vector<float>r2,r4,r8;run_rg<2>(dw,K,rows,dx,dxs,ds,y1,smem,iters,trials,r2);run_rg<4>(dw,K,rows,dx,dxs,ds,y1,smem,iters,trials,r4);run_rg<8>(dw,K,rows,dx,dxs,ds,y1,smem,iters,trials,r8);CHECK(cudaMemcpy(h1.data(),y1,h1.size()*4,cudaMemcpyDeviceToHost));
    double maxabs=0,maxrel=0;size_t different=0;for(size_t i=0;i<h0.size();++i){double d=fabs((double)h0[i]-h1[i]);maxabs=fmax(maxabs,d);maxrel=fmax(maxrel,d/(fabs((double)h0[i])+1e-12));if(h0[i]!=h1[i])++different;}
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));std::vector<float>ref;for(int t=0;t<trials;++t){CHECK(cudaEventRecord(a));for(int i=0;i<iters;++i)k_ref<<<gr,128>>>(dw,K,rows,dx,dxs,ds,y0,rows);CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));ref.push_back(ms/iters);}auto med=[](std::vector<float>x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    printf("E060,shape=%ldx%ld,weight_MiB=%.2f,smem_KiB=%.2f,ref_ms=%.4f,bitexact_different=%zu,max_abs=%.7g,max_rel=%.7g\n",rows,K,wb/1048576.0,smem/1024.0,med(ref),different,maxabs,maxrel);printf("row_groups=2,rows_per_CTA=8,ms=%.4f,speedup=%.3f\n",med(r2),med(ref)/med(r2));printf("row_groups=4,rows_per_CTA=16,ms=%.4f,speedup=%.3f\n",med(r4),med(ref)/med(r4));printf("row_groups=8,rows_per_CTA=32,ms=%.4f,speedup=%.3f\n",med(r8),med(ref)/med(r8));return 0;
}
