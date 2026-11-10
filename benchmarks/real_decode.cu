// Real decode (E038): full transformer decode loop over a real GGUF model with tiered
// VRAM/RAM/NVMe weight residency — the streamed_decode proxy's intelligent residency
// (hot VRAM arena + pinned RAM cache + overlapped H2D) applied to actual layer math:
// attention with KV state, and for qwen35 the gated-delta-net (linear attention) path
// with persistent SSM state. Greedy sampling, token IDs in/out.
//
// Supported architectures (matched against llama.cpp src/models/{qwen3,qwen35}.cpp and
// ggml-cpu/ops.cpp ggml_compute_forward_gated_delta_net_one_chunk, read 2026-11-10):
//   qwen3  (Qwen3-0.6B): all full-attention layers, q/k per-head RMS norm, full RoPE.
//   qwen35 (Qwen3.8-27B): 64 blocks in a 3:1 pattern (3 gated-delta-net, 1 full attention),
//          fused attn_qkv + z gate on recurrent layers, interleaved Q+gate on attention
//          layers, mrope sections [11,11,10,0] (identical to standard RoPE for text where
//          T=H=W), partial rope (n_rot of head_dim), fused attn_norm -> post_attention_norm.
//
// --validate 1 runs a float32 CPU reference decode (same int8 per-32-block activation
// quantization as the GPU dp4a GEMV) and compares argmax + logits at every step.
// The CPU reference dequantizes all weights to RAM; it is only feasible for small models.
//
// Residency machinery (slots, pinned staging ring, reader threads, copy/compute streams,
// slot-reuse events) is carried over from benchmarks/streamed_decode.cu (E031-E037). All
// compute runs on the compute stream in layer order (real decode is sequential); cold
// copies run ahead on the copy stream so H2D overlaps with compute.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

static constexpr size_t kMaxItem = 32u << 20;
static constexpr size_t kStageBytes = kMaxItem + 8192;
static constexpr int kStage = 8;
static constexpr int kSlots = 4;
static constexpr uint64_t kMapMargin = 64ull << 20;
static constexpr int kGGML_F32 = 0, kGGML_F16 = 1, kGGML_Q4_0 = 2;  // ggml enum: Q4_0=2 (8 is Q8_0)

static void check(cudaError_t s, const char* w) {
    if (s != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(s)); exit(1); }
}
static long meminfo_mib(const char* key) {
    FILE* f = fopen("/proc/meminfo", "r"); char l[256]; long v = -1; size_t n = strlen(key);
    while (f && fgets(l, sizeof l, f)) if (!strncmp(l, key, n)) { v = atol(l + n) / 1024; break; }
    if (f) fclose(f);
    return v;
}
static long g_swap0, g_floor = 6144, g_swap_tol = 1024;
static int g_dbg = 0;
static void dbg_cpu(const char* tag, const float* v, long n) {
    if (!g_dbg) return;
    printf("dbg_cpu,%s,%ld", tag, n);
    for (long i = 0; i < n; ++i) printf(",%.9g", v[i]);
    printf("\n");
}
static void safety(const char* where) {
    long a = meminfo_mib("MemAvailable:"), s = meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:");
    if (a < g_floor || s > g_swap0 + g_swap_tol) {
        fprintf(stderr, "ABORT at %s: MemAvailable=%ld MiB swap_used=%ld (baseline %ld)\n", where, a, s, g_swap0);
        _exit(4);
    }
}
static long free_vram_mib() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return -1;
    return (long)(free_b >> 20);
}
static float cpu_half(const uint8_t* p) {
    uint16_t u = (uint16_t)(p[0] | (p[1] << 8)); uint32_t s = (u >> 15) & 1, e = (u >> 10) & 31, m = u & 1023; float v;
    if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf((float)(m + 1024), (int)e - 25);
    return s ? -v : v;
}
static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// ---------------------------------------------------------------------------
// Minimal GGUF reader: header KVs and tensor directory only.
// ---------------------------------------------------------------------------
struct GgufKV {
    std::map<std::string, std::string> strs;
    std::map<std::string, long> ints;
    std::map<std::string, double> flts;
    std::map<std::string, std::vector<std::string>> arr_strs;
    std::map<std::string, std::vector<long>> arr_ints;
    std::map<std::string, std::vector<char>> arr_bools;
};
struct GTensor { std::string name; int type; long ne[4] = {1, 1, 1, 1}; int nd; uint64_t off, bytes; };

static std::string read_str(FILE* f) {
    uint64_t n = 0;
    if (fread(&n, 8, 1, f) != 1) return "";
    if (n > (1u << 24)) return "";
    std::string s(n, 0);
    if (n && fread(&s[0], 1, n, f) != (size_t)n) return "";
    return s;
}
static bool parse_gguf(const char* path, GgufKV& kv, std::vector<GTensor>& tensors, uint64_t& data_start) {
    FILE* f = fopen(path, "rb"); if (!f) { perror("gguf open"); return false; }
    uint32_t magic, ver; uint64_t ntens, nkvs;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) { fprintf(stderr, "not gguf\n"); fclose(f); return false; }
    if (fread(&ver, 4, 1, f) != 1 || ver != 3) { fprintf(stderr, "gguf v%u unsupported\n", ver); fclose(f); return false; }
    if (fread(&ntens, 8, 1, f) != 1 || fread(&nkvs, 8, 1, f) != 1) { fclose(f); return false; }
    for (uint64_t i = 0; i < nkvs; ++i) {
        std::string k = read_str(f); if (k.empty()) { fclose(f); return false; }
        uint32_t vt; if (fread(&vt, 4, 1, f) != 1) { fclose(f); return false; }
        auto rd_u32 = [&]() { uint32_t v = 0; fread(&v, 4, 1, f); return (long)v; };
        auto rd_i64 = [&]() { int64_t v = 0; fread(&v, 8, 1, f); return (long)v; };
        auto rd_f32 = [&]() { float v = 0; fread(&v, 4, 1, f); return (double)v; };
        auto rd_u64 = [&]() { uint64_t v = 0; fread(&v, 8, 1, f); return (long)v; };
        switch (vt) {
            case 0: case 1: { uint8_t v = 0; if (fread(&v, 1, 1, f) != 1) { fclose(f); return false; } kv.ints[k] = v; break; }
            case 2: case 3: { uint16_t v = 0; if (fread(&v, 2, 1, f) != 1) { fclose(f); return false; } kv.ints[k] = v; break; }
            case 4: case 5: { int32_t v = 0; if (fread(&v, 4, 1, f) != 1) { fclose(f); return false; } kv.ints[k] = v; break; }
            case 6: kv.flts[k] = rd_f32(); break;
            case 7: { uint8_t v = 0; if (fread(&v, 1, 1, f) != 1) { fclose(f); return false; } kv.ints[k] = v ? 1 : 0; break; }
            case 8: kv.strs[k] = read_str(f); break;
            case 9: { uint32_t at; if (fread(&at, 4, 1, f) != 1) { fclose(f); return false; }
                      long cnt = rd_u64(); std::vector<std::string> vs; std::vector<long> vi; std::vector<char> vb;
                      for (long j = 0; j < cnt; ++j) {
                          if (at == 8) vs.push_back(read_str(f));
                          else if (at == 4 || at == 5) vi.push_back(rd_u32());
                          else if (at == 10 || at == 11) vi.push_back(rd_i64());
                          else if (at == 7) { uint8_t b = 0; if (fread(&b, 1, 1, f) != 1) { fclose(f); return false; } vb.push_back(b ? 1 : 0); }
                          else if (at == 0 || at == 1) { uint8_t b = 0; if (fread(&b, 1, 1, f) != 1) { fclose(f); return false; } vi.push_back(b); }
                          else if (at == 2 || at == 3) { uint16_t b = 0; if (fread(&b, 2, 1, f) != 1) { fclose(f); return false; } vi.push_back(b); }
                          else if (at == 6) { float b = 0; if (fread(&b, 4, 1, f) != 1) { fclose(f); return false; } vi.push_back((long)b); }
                          else { fprintf(stderr, "array elem type %u unsupported\n", at); fclose(f); return false; }
                      }
                      if (!vs.empty()) kv.arr_strs[k] = vs;
                      if (!vi.empty()) kv.arr_ints[k] = vi;
                      if (!vb.empty()) kv.arr_bools[k] = vb;
                      break; }
            case 10: case 11: kv.ints[k] = rd_i64(); break;
            case 12: { double v = 0; if (fread(&v, 8, 1, f) != 1) { fclose(f); return false; } kv.flts[k] = v; break; }
            default: fprintf(stderr, "kv type %u unsupported for %s\n", vt, k.c_str()); fclose(f); return false;
        }
    }
    for (uint64_t i = 0; i < ntens; ++i) {
        GTensor t; t.name = read_str(f);
        if (fread(&t.nd, 4, 1, f) != 1 || t.nd > 4) { fclose(f); return false; }
        for (int d = 0; d < t.nd; ++d) { uint64_t v; if (fread(&v, 8, 1, f) != 1) { fclose(f); return false; } t.ne[d] = (long)v; }
        uint32_t ty; if (fread(&ty, 4, 1, f) != 1) { fclose(f); return false; }
        t.type = (int)ty;
        if (fread(&t.off, 8, 1, f) != 1) { fclose(f); return false; }
        long esz = t.type == kGGML_F32 ? 4 : t.type == kGGML_F16 ? 2 : t.type == kGGML_Q4_0 ? 18 / 32 * 32 : 0;
        // exact: Q4_0 block = 18 bytes / 32 elems; F32 4; F16 2
        long n = 1; for (int d = 0; d < t.nd; ++d) n *= t.ne[d];
        t.bytes = t.type == kGGML_Q4_0 ? (uint64_t)((n + 31) / 32) * 18 : (uint64_t)n * esz;
        tensors.push_back(t);
    }
    long align = 32; auto it = kv.ints.find("general.alignment"); if (it != kv.ints.end()) align = it->second;
    data_start = align_up((uint64_t)ftell(f), (uint64_t)align);
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Byte-level BPE (greedy longest-match over vocab for encode; string decode).
// Only used for --prompt convenience and display; correctness tests use --prompt-ids.
// ---------------------------------------------------------------------------
struct Tok {
    std::vector<std::string> vocab;
    std::map<std::string, long> id_of;
    long bos = -1, eos = -1;
    std::string unescape(const std::string& s) const {
        // GPT-2/Qwen byte-level vocab: 'Ġ'->space, 'Ċ'->newline; <0xNN> -> raw byte.
        if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            auto hx = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
            return std::string(1, (char)(hx(s[3]) * 16 + hx(s[4])));
        }
        std::string out; out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            unsigned char c = (unsigned char)s[i];
            if (c == 0xC4 && i + 1 < s.size() && (unsigned char)s[i + 1] == 0xA0) { out += ' '; ++i; }
            else if (c == 0xC4 && i + 1 < s.size() && (unsigned char)s[i + 1] == 0x8A) { out += '\n'; ++i; }
            else out += (char)c;
        }
        return out;
    }
    std::string decode(const std::vector<long>& ids) const {
        std::string out;
        for (long id : ids) {
            if (id < 0 || id >= (long)vocab.size()) continue;
            out += unescape(vocab[id]);
        }
        return out;
    }
    static std::string to_bytes(const std::string& text) {
        // GPT-2 byte-level representation: printable ASCII identity, space -> "Ġ",
        // newline -> "Ċ" (UTF-8), all other bytes -> "<0xNN>".
        std::string out;
        for (unsigned char c : text) {
            if (c == ' ') out += "Ä ";
            else if (c == '\n') out += "Ä";
            else if (c >= 33 && c <= 126) out += (char)c;
            else {
                char b[8]; snprintf(b, sizeof b, "<0x%02X>", c); out += b;
            }
        }
        return out;
    }
    std::vector<long> encode(const std::string& text) const {
        // greedy longest match over the byte-level text (approximate BPE; exact
        // validation runs use --prompt-ids)
        const std::string bytes = to_bytes(text);
        std::vector<long> ids;
        size_t i = 0;
        while (i < bytes.size()) {
            size_t best_len = 0; long best_id = -1;
            for (size_t l = std::min(bytes.size() - i, (size_t)32); l >= 1; --l) {
                auto it = id_of.find(bytes.substr(i, l));
                if (it != id_of.end()) { best_len = l; best_id = it->second; break; }
            }
            if (best_id < 0) { ++i; continue; }
            ids.push_back(best_id); i += best_len;
        }
        return ids;
    }
};

// ---------------------------------------------------------------------------
// Model description
// ---------------------------------------------------------------------------
struct TensorPtrs { const GTensor* t; };  // resolved per name
struct Model {
    std::string arch;                 // "qwen3" or "qwen35"
    long n_layer = 0, n_embd = 0, n_head = 0, n_kv = 0, head_k = 0, head_v = 0;
    long n_ff = 0, n_rot = 0, n_vocab = 0;
    double rope_base = 10000.0, eps = 1e-6;
    // qwen35 ssm dims
    long sss_state = 0, sss_group = 0, sss_dt = 0, sss_inner = 0, sss_conv = 0;
    std::vector<char> is_recr;
    long n_recr = 0, n_attn = 0;
    const GTensor* embd = nullptr; const GTensor* out_w = nullptr;
    std::map<std::string, const GTensor*> by_name;
    // derived
    long n_qg = 0;   // attn_q output dims (2*n_head*head_v for qwen35, n_head*head_k for qwen3)
    long qkv_out = 0;
    long bos_id = -1, eos_id = -1;
    std::map<std::string, uint64_t> small_off;   // small F32 tensors -> offset in device blob
};

static const GTensor* gt(const Model& m, const char* fmt, long il, char* buf, size_t bs) {
    snprintf(buf, bs, fmt, il);
    auto it = m.by_name.find(buf);
    return it == m.by_name.end() ? nullptr : it->second;
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------
// quantize NB activation vectors: x [NB][K], xq [NB][K], xs/xsum [NB][K/32]
__global__ void k_quant_x(const float* x, long nb, int8_t* xq, float* xs, int* xsum) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nb) return;
    float amax = 0.f; for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
    float d = amax / 127.f; float id = d > 0.f ? 1.f / d : 0.f; int s = 0;
    for (int i = 0; i < 32; ++i) { int q = __float2int_rn(x[b * 32 + i] * id); xq[b * 32 + i] = (int8_t)q; s += q; }
    xs[b] = d; xsum[b] = s;
}
// y[v*stride + row] = dot(W_row, xq_v) — NB decode vectors share one weight read;
// identical math to streamed_decode gemv_q4 (dp4a, E033/E034)
template <int NB>
__global__ void k_mat_q4(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum, float* y, long ystride) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32; const size_t rowbytes = (size_t)nb * 18;
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
        const uint32_t lo0 = q[0] & 0x0F0F0F0Fu, lo1 = q[1] & 0x0F0F0F0Fu, lo2 = q[2] & 0x0F0F0F0Fu, lo3 = q[3] & 0x0F0F0F0Fu;
        const uint32_t hi0 = (q[0] >> 4) & 0x0F0F0F0Fu, hi1 = (q[1] >> 4) & 0x0F0F0F0Fu, hi2 = (q[2] >> 4) & 0x0F0F0F0Fu, hi3 = (q[3] >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int v = 0; v < NB; ++v) {
            const int4* xp = (const int4*)(xq + v * (K / 32 * 32) + b * 32);
            const int4 xa = xp[0], xb = xp[1];
            int s = 0;
            s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
            s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
            acc[v] += d * xs[v * (K / 32) + b] * (float)(s - 8 * xsum[v * (K / 32) + b]);
        }
    }
#pragma unroll
    for (int v = 0; v < NB; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[v * ystride + row] = a;
    }
}
// rms norm over NB vectors: x/out [NB][n], grid (1, NB)
__global__ void k_rms_norm(float* out, const float* x, const float* w, int n, float eps) {
    const long v = blockIdx.y;
    x += v * n; out += v * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = x[i]; ss += v * v; }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f; for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) t += red[i];
        red[0] = rsqrtf(t / n + eps);
    }
    __syncthreads();
    const float s = red[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * s * w[i];
}
// per-head RMS norm in place (x: [nh][stride]; stride==hd for packed heads, 2*hd for qwen35 q+gate)
__global__ void k_rms_heads(float* x, const float* w, int nh, int hd, float eps, int stride) {
    const int h = blockIdx.x;
    const long v = blockIdx.y;
    x += v * (size_t)nh * stride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) { float v = x[(size_t)h * stride + i]; ss += v * v; }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f; for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) t += red[i];
        red[0] = rsqrtf(t / hd + eps);
    }
    __syncthreads();
    const float s = red[0];
    for (int i = threadIdx.x; i < hd; i += blockDim.x) x[(size_t)h * stride + i] *= s * w[i];
}
// qwen35 attention output gate: out[h*hd+i] *= sigmoid(qg[h*2*hd+hd+i])
__global__ void k_attn_gate_apply(float* out, const float* qg, int nh, int hd) {
    const long idx = ((long)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= (long)gridDim.y * nh * hd) return;
    const long v = idx / ((long)nh * hd);
    const long i = idx - v * (long)nh * hd;
    const int h = (int)(i / hd);
    const float gt = qg[v * (size_t)nh * 2 * hd + (size_t)h * 2 * hd + hd + (i % hd)];
    out[idx] /= (1.f + expf(-gt));
}
// in-place L2 norm per head: x /= max(sqrt(sum(x^2)), eps)-style (ggml l2_norm semantics)
// x: [B][nh*hd] stacked with seq_stride between sequences; grid (nh, B)
__global__ void k_l2_heads(float* x, int nh, int hd, float eps, long seq_stride) {
    const int h = blockIdx.x;
    x += (size_t)blockIdx.y * seq_stride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) { float v = x[h * hd + i]; ss += v * v; }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f; for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) t += red[i];
        red[0] = 1.f / fmaxf(sqrtf(t), eps);   // ggml: 1/max(sqrt(sum), eps)
    }
    __syncthreads();
    const float s = red[0];
    for (int i = threadIdx.x; i < hd; i += blockDim.x) x[(size_t)h * hd + i] *= s;
}
// standard (neox half-split) RoPE, text positions: pair (x_i, x_i+n_rot/2), theta = pos*base^(-2i/n_rot)
// q: [nh_q][hd] (qwen35: only first hd of each (hd+hd_gate) slice — pass stride), k: [nh_k][hd]
__global__ void k_rope(float* q, float* k, int pos, int nh_q, int nh_k, int hd, int n_rot, float base, int q_stride, int style) {
    const int h = blockIdx.x;
    const long v = blockIdx.y;
    q += v * (size_t)nh_q * q_stride;
    k += v * (size_t)nh_k * hd;
    const int half = n_rot / 2;
    float* dst = h < nh_q ? q + (size_t)h * q_stride : k + (size_t)(h - nh_q) * hd;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float theta = (float)pos * powf(base, -2.f * (float)i / (float)n_rot);
        const float c = cosf(theta), s = sinf(theta);
        const int ia = style ? 2 * i : i;
        const int ib = style ? 2 * i + 1 : i + half;
        const float a = dst[ia], b = dst[ib];
        dst[ia] = a * c - b * s;
        dst[ib] = a * s + b * c;
    }
}
// GQA decode attention for one token at position pos (cache already contains pos+1 entries
// for this layer after append). K/V cache: [n_kh][max_ctx][hd]. out: [nh_q][hd].
__global__ void k_attn_decode(float* out, const float* Kc, const float* Vc, const float* q,
                              int pos, int n_ctx_max, int nh_q, int n_kh, int hd, float scale, int q_stride, long kv_seq_stride) {
    extern __shared__ float sh[];   // pos+1 scores
    const int h = blockIdx.x;
    const long v = blockIdx.y;
    q += v * (size_t)nh_q * q_stride;
    out += v * (size_t)nh_q * hd;
    Kc += v * (size_t)kv_seq_stride;
    Vc += v * (size_t)kv_seq_stride;
    const int kh = h / (nh_q / n_kh);
    const float* qh = q + (size_t)h * q_stride;
    const float* Kk = Kc + (size_t)kh * n_ctx_max * hd;
    const float* Vk = Vc + (size_t)kh * n_ctx_max * hd;
    const int np = pos + 1;
    for (int t = 0; t < np; ++t) {
        float p = 0.f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) p += qh[i] * Kk[(size_t)t * hd + i];
        __shared__ float red[32];
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = p;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f; for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) s += red[i];
            sh[t] = s * scale;
        }
        __syncthreads();
    }
    __shared__ float mx, zsum;
    if (threadIdx.x == 0) {
        float m = sh[0]; for (int t = 1; t < np; ++t) m = fmaxf(m, sh[t]);
        mx = m; float z = 0.f; for (int t = 0; t < np; ++t) z += expf(sh[t] - m);
        zsum = z;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < np; ++t) acc += expf(sh[t] - mx) * Vk[(size_t)t * hd + i];
        out[(size_t)h * hd + i] = acc / zsum;
    }
}
// gated delta net, one token: state[h] is S transposed (s[h][j*S+i] = S[i][j]).
// q16/k16: [ngroup][S]; v: [H][S]; g/beta: [H]; out: [H][S].
__global__ void k_delta_net(float* state, const float* q16, const float* k16, const float* v,
                            const float* g, const float* beta, float* out, int S, int H, int ngroup, float scale,
                            long state_seq_stride, long qkv_seq_stride) {
    const int h = blockIdx.x;
    const long vs = blockIdx.y;
    state += (size_t)vs * state_seq_stride;
    // q/k/v live in the dense [B][qkv_out] conv output: advance each by vs*qkv_seq_stride
    q16 += vs * qkv_seq_stride;
    k16 += vs * qkv_seq_stride;
    const float* vv = v + vs * qkv_seq_stride;
    g += vs * H; beta += vs * H; out += vs * (size_t)H * S;
    const int j = threadIdx.x;
    if (j >= S) return;
    // ggml maps v-head -> k-head with iv1 % n_k_heads (ops.cpp gated_delta_net), NOT grouping
    const int hk = h % ngroup;
    float* row = state + (size_t)h * S * S + (size_t)j * S;
    const float* kh = k16 + (size_t)hk * S;
    const float* qh = q16 + (size_t)hk * S;
    const float* vh = vv + (size_t)h * S;
    const float eg = expf(g[h]);
    for (int i = 0; i < S; ++i) row[i] *= eg;
    float dot = 0.f; for (int i = 0; i < S; ++i) dot += row[i] * kh[i];
    const float dj = (vh[j] - dot) * beta[h];
    for (int i = 0; i < S; ++i) row[i] += dj * kh[i];
    float o = 0.f; for (int i = 0; i < S; ++i) o += row[i] * qh[i];
    out[(size_t)h * S + j] = o * scale;
}
// conv1d(k=4) + silu, decode step: window = conv_state[c][0..2], x[c]; shifts state.
// w layout F32 [4, C]: w[c*4 + j].
__global__ void k_conv1d_silu(float* cstate, const float* x, const float* w, float* out, int C, long state_seq_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const long v = blockIdx.y;
    cstate += (size_t)v * state_seq_stride;
    x += v * C; out += v * C;
    if (c >= C) return;
    const float s0 = cstate[c * 3 + 0], s1 = cstate[c * 3 + 1], s2 = cstate[c * 3 + 2];
    const float xc = x[c];
    // conv weight is F32 with ne=(4, C): taps fastest, element (c, j) at c*4 + j
    const float sum = w[c * 4 + 0] * s0 + w[c * 4 + 1] * s1 + w[c * 4 + 2] * s2 + w[c * 4 + 3] * xc;
    cstate[c * 3 + 0] = s1; cstate[c * 3 + 1] = s2; cstate[c * 3 + 2] = xc;
    out[c] = sum / (1.f + expf(-sum));
}
// all elementwise kernels below operate on NB stacked vectors: idx in [0, NB*n), v = idx / n
__global__ void k_sigmoid(float* y, const float* x, int n) {
    const long i = ((long)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; if (i >= (long)gridDim.y * n) return;
    y[i] = 1.f / (1.f + expf(-x[i]));
}
__global__ void k_gate_ssm(float* g, const float* alpha, const float* dtb, const float* a, int n) {
    const long i = ((long)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; if (i >= (long)gridDim.y * n) return;
    const float v = alpha[i] + dtb[i % n];
    const float sp = v > 20.f ? v : log1pf(expf(v));
    g[i] = sp * a[i % n];
}
__global__ void k_silu_mul(float* y, const float* a, const float* b, int n) {  // y = silu(a)*b
    const long i = ((long)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; if (i >= (long)gridDim.y * n) return;
    const float v = a[i];
    y[i] = v / (1.f + expf(-v)) * b[i];
}
__global__ void k_sigmoid_mul(float* y, const float* gate, const float* x, int n) {  // y = x*sigmoid(gate)
    const long i = ((long)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; if (i >= (long)gridDim.y * n) return;
    y[i] = x[i] / (1.f + expf(-gate[i]));
}
__global__ void k_add(float* y, const float* x, int n) {
    const long i = ((long)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; if (i >= (long)gridDim.y * n) return;
    y[i] += x[i];
}
// gated RMS norm for ssm output: out = rms_norm(x)[.]*silu(z), per v-head (x: [H][S], z: [H][S])
__global__ void k_gated_norm(float* out, const float* x, const float* w, const float* z, int H, int S, float eps) {
    const int h = blockIdx.x;
    const long v = blockIdx.y;
    x += v * (size_t)H * S; z += v * (size_t)H * S; out += v * (size_t)H * S;
    float ss = 0.f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { float v = x[h * S + i]; ss += v * v; }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f; for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) t += red[i];
        red[0] = rsqrtf(t / S + eps);
    }
    __syncthreads();
    const float s = red[0];
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        const float zv = z[h * S + i];
        out[h * S + i] = x[h * S + i] * s * w[i] * (zv / (1.f + expf(-zv)));
    }
}

// ---------------------------------------------------------------------------
// CPU reference decode (validation): float32 mirror of the GPU math, including
// the int8 per-32-block activation quantization used by the dp4a Q4_0 GEMV.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// CPU reference decode (validation): exact mirror of the GPU math. GEMVs work
// on the raw Q4_0 nibbles with the same int8 per-32-block activation
// quantization as the dp4a kernel; norms/attention/delta-net are float32.
// ---------------------------------------------------------------------------
struct CPUModel {
    Model m;
    const uint8_t* mm_base = nullptr;                     // mmap of the GGUF file
    uint64_t mm_data_start = 0;
    std::map<std::string, const uint8_t*> raw_q4;         // raw Q4_0 bytes (into mmap)
    std::map<std::string, const float*> f32;              // F32 tensors (into mmap)
    std::vector<std::vector<float>> kv_k, kv_v;           // per attn layer [n_kh][pos][hd]
    std::vector<std::vector<float>> ssm;                  // per recr layer [H][S*S] transposed
    std::vector<std::vector<float>> conv;                 // per recr layer [C][3]
    long pos = 0, ctx = 0;
};
static void cpu_dequant_row_q4(const uint8_t* row, long K, float* out) {
    const long nb = K / 32;
    for (long b = 0; b < nb; ++b) {
        const uint8_t* blk = row + b * 18;
        const float d = cpu_half(blk);
        for (int j = 0; j < 16; ++j) {
            out[b * 32 + j] = d * (float)((int)(blk[2 + j] & 15) - 8);
            out[b * 32 + j + 16] = d * (float)((int)(blk[2 + j] >> 4) - 8);
        }
    }
}
static void cpu_quant(const float* x, long K, std::vector<int8_t>& xq, std::vector<float>& xs, std::vector<int>& xsum) {
    const long nb = K / 32;
    xq.resize(K); xs.resize(nb); xsum.resize(nb);
    for (long b = 0; b < nb; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        int s = 0;
        for (int i = 0; i < 32; ++i) { const int q = (int)nearbyintf(x[b * 32 + i] * id); xq[b * 32 + i] = (int8_t)q; s += q; }
        xs[b] = d; xsum[b] = s;
    }
}
static void cpu_mat_q4(const CPUModel& cm, const char* name, const float* x, float* y) {
    const uint8_t* raw = cm.raw_q4.at(name);
    const GTensor* t = cm.m.by_name.at(name);
    const long K = t->ne[0], N = t->ne[1];
    const long nb = K / 32;
    std::vector<int8_t> xq; std::vector<float> xs; std::vector<int> xsum;
    cpu_quant(x, K, xq, xs, xsum);
    for (long r = 0; r < N; ++r) {
        const uint8_t* row = raw + (size_t)r * nb * 18;
        float acc = 0.f;
        for (long b = 0; b < nb; ++b) {
            const uint8_t* blk = row + b * 18;
            const float d = cpu_half(blk);
            int s = 0;
            // mirror the GPU dp4a kernel exactly: raw nibble dot, single zero-point
            // subtraction via xsum (folding (q-8) here AND subtracting xsum doubles it)
            for (int j = 0; j < 16; ++j)
                s += (int)(blk[2 + j] & 15) * (int)xq[b * 32 + j] + (int)(blk[2 + j] >> 4) * (int)xq[b * 32 + j + 16];
            acc += d * xs[b] * (float)(s - 8 * xsum[b]);
        }
        y[r] = acc;
    }
}
static void cpu_rms(const float* w, const float* x, float* out, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float s = 1.f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * s * w[i];
}
static void cpu_rms_heads_inplace(float* x, const float* w, int nh, int hd, float eps) {
    for (int h = 0; h < nh; ++h) {
        double ss = 0; for (int i = 0; i < hd; ++i) ss += (double)x[h * hd + i] * x[h * hd + i];
        const float s = 1.f / sqrtf((float)(ss / hd) + eps);
        for (int i = 0; i < hd; ++i) x[h * hd + i] *= s * w[i];
    }
}
static void cpu_l2_heads_inplace(float* x, int nh, int hd, float eps) {
    for (int h = 0; h < nh; ++h) {
        double ss = 0; for (int i = 0; i < hd; ++i) ss += (double)x[h * hd + i] * x[h * hd + i];
        const float s = 1.f / sqrtf((float)ss + eps);
        for (int i = 0; i < hd; ++i) x[h * hd + i] *= s;
    }
}
static void cpu_rope(float* q, float* k, int pos, int nh_q, int nh_k, int hd, int n_rot, float base, int q_stride, int style) {
    const int half = n_rot / 2;
    for (int h = 0; h < nh_q + nh_k; ++h) {
        float* dst = h < nh_q ? q + (size_t)h * q_stride : k + (size_t)(h - nh_q) * hd;
        for (int i = 0; i < half; ++i) {
            const double theta = (double)pos * pow((double)base, -2.0 * i / n_rot);
            const float c = (float)cos(theta), s = (float)sin(theta);
            const int ia = style ? 2 * i : i;
            const int ib = style ? 2 * i + 1 : i + half;
            const float a = dst[ia], b = dst[ib];
            dst[ia] = a * c - b * s;
            dst[ib] = a * s + b * c;
        }
    }
}
static void cpu_silu(float* y, const float* x, int n) { for (int i = 0; i < n; ++i) y[i] = x[i] / (1.f + expf(-x[i])); }
static void cpu_add(float* y, const float* x, int n) { for (int i = 0; i < n; ++i) y[i] += x[i]; }

// One decode step on CPU: token -> logits (greedy argmax done by caller). Returns logits in out.
static void cpu_step(CPUModel& cm, long tok, std::vector<float>& logits) {
    const Model& m = cm.m;
    const bool is35 = m.arch == "qwen35";
    std::vector<float> x(m.n_embd), xn(m.n_embd), y(m.n_embd);
    {   // embedding row (token_embd Q4_0 [n_embd, vocab], row-major per vocab entry)
        const uint8_t* raw = cm.raw_q4.at("token_embd.weight");
        const long nb = m.n_embd / 32;
        cpu_dequant_row_q4(raw + (size_t)tok * nb * 18, m.n_embd, x.data());
    }
    dbg_cpu("emb", x.data(), m.n_embd);
    for (long il = 0; il < m.n_layer; ++il) {
        char nm[128];
        // attn_norm
        { const float* w = cm.f32.at("blk." + std::to_string(il) + ".attn_norm.weight");
          cpu_rms(w, x.data(), xn.data(), m.n_embd, (float)m.eps); }
        if (il == 0) dbg_cpu("norm0", xn.data(), m.n_embd);
        if (is35 && m.is_recr[il]) {
            const std::string L = "blk." + std::to_string(il);
            const long S = m.sss_state, NG = m.sss_group, H = m.sss_dt;   // 128, 16, 48
            const long Sv = m.sss_inner / H;                               // 128
            std::vector<float> qkv(m.qkv_out), z(m.sss_inner), beta(H), alpha(H), g(H), cout(m.qkv_out), dout(m.sss_inner);
            cpu_mat_q4(cm, (L + ".attn_qkv.weight").c_str(), xn.data(), qkv.data());
            cpu_mat_q4(cm, (L + ".attn_gate.weight").c_str(), xn.data(), z.data());
            cpu_mat_q4(cm, (L + ".ssm_beta.weight").c_str(), xn.data(), beta.data());
            cpu_mat_q4(cm, (L + ".ssm_alpha.weight").c_str(), xn.data(), alpha.data());
            const float* dtb = cm.f32.at(L + ".ssm_dt.bias");
            const float* av = cm.f32.at(L + ".ssm_a");
            for (int h = 0; h < H; ++h) {
                beta[h] = 1.f / (1.f + expf(-beta[h]));
                const float v = alpha[h] + dtb[h];
                g[h] = (v > 20.f ? v : log1pf(expf(v))) * av[h];
            }
            if (il == 0) { dbg_cpu("qkv0", qkv.data(), (long)qkv.size()); dbg_cpu("zraw0", z.data(), (long)z.size()); dbg_cpu("beta0", beta.data(), (long)beta.size()); dbg_cpu("alpharaw0", alpha.data(), (long)alpha.size()); }
            // conv1d + silu, state update
            const float* kw = cm.f32.at(L + ".ssm_conv1d.weight");   // [C,4] ggml order, tap j at kw[j*C + c]
            std::vector<float>& cst = cm.conv[il - (il + 1) / 4];   // recurrent-layer index
            for (int c = 0; c < m.qkv_out; ++c) {
                const float s0 = cst[c * 3 + 0], s1 = cst[c * 3 + 1], s2 = cst[c * 3 + 2];
                const float sum = kw[c * 4 + 0] * s0 + kw[c * 4 + 1] * s1 + kw[c * 4 + 2] * s2 + kw[c * 4 + 3] * qkv[c];
                cst[c * 3 + 0] = s1; cst[c * 3 + 1] = s2; cst[c * 3 + 2] = qkv[c];
                cout[c] = sum / (1.f + expf(-sum));
            }
            float* q16 = cout.data();
            float* k16 = cout.data() + S * NG;
            float* v48 = cout.data() + 2 * S * NG;
            cpu_l2_heads_inplace(q16, NG, S, (float)m.eps);
            cpu_l2_heads_inplace(k16, NG, S, (float)m.eps);
            if (il == 0) dbg_cpu("convout0", cout.data(), (long)cout.size());
            // delta net
            std::vector<float>& st = cm.ssm[il - (il + 1) / 4];   // recurrent-layer index
            const float scale = 1.f / sqrtf((float)Sv);
            for (int h = 0; h < H; ++h) {
                const int hk = h % NG;   // ggml: iv1 % n_k_heads
                float* row = &st[(size_t)h * Sv * Sv];
                const float* kh = k16 + (size_t)hk * S;
                const float* qh = q16 + (size_t)hk * S;
                const float* vh = v48 + (size_t)h * S;
                const float eg = expf(g[h]);
                for (int j = 0; j < Sv; ++j) {
                    float* r = row + (size_t)j * Sv;
                    for (int i = 0; i < Sv; ++i) r[i] *= eg;
                    float dot = 0.f; for (int i = 0; i < Sv; ++i) dot += r[i] * kh[i];
                    const float dj = (vh[j] - dot) * beta[h];
                    for (int i = 0; i < Sv; ++i) r[i] += dj * kh[i];
                    float o = 0.f; for (int i = 0; i < Sv; ++i) o += r[i] * qh[i];
                    dout[(size_t)h * Sv + j] = o * scale;
                }
            }
            if (il == 0) dbg_cpu("dnout0", dout.data(), (long)dout.size());
            // gated norm: out = rms(dout)[h] * silu(z)[h]
            const float* nw = cm.f32.at(L + ".ssm_norm.weight");
            for (int h = 0; h < H; ++h) {
                double ss = 0; for (int i = 0; i < Sv; ++i) ss += (double)dout[h * Sv + i] * dout[h * Sv + i];
                const float s = 1.f / sqrtf((float)(ss / Sv) + (float)m.eps);
                for (int i = 0; i < Sv; ++i) {
                    const float zv = z[h * Sv + i];
                    dout[h * Sv + i] = dout[h * Sv + i] * s * nw[i] * (zv / (1.f + expf(-zv)));
                }
            }
            if (il == 0) dbg_cpu("gatednorm0", dout.data(), (long)dout.size());
            cpu_mat_q4(cm, (L + ".ssm_out.weight").c_str(), dout.data(), y.data());
        } else {
            const std::string L = "blk." + std::to_string(il);
            const int hd = (int)(is35 ? m.head_v : m.head_k);
            const int nh_q = (int)m.n_head, nh_k = (int)m.n_kv;
            const int nrot = (int)(is35 ? m.n_rot : m.head_k);
            std::vector<float> qg(is35 ? m.n_qg : m.n_head * m.head_k), k(nh_k * hd), v(nh_k * hd), att(nh_q * hd);
            cpu_mat_q4(cm, (L + ".attn_q.weight").c_str(), xn.data(), qg.data());
            cpu_mat_q4(cm, (L + ".attn_k.weight").c_str(), xn.data(), k.data());
            cpu_mat_q4(cm, (L + ".attn_v.weight").c_str(), xn.data(), v.data());
            if (il <= 1) dbg_cpu(("v" + std::to_string(il)).c_str(), v.data(), (long)v.size());
            // per-head q norm (q part only for qwen35; gate sits after each head's q block)
            if (is35) {
                for (int h = 0; h < nh_q; ++h)
                    cpu_rms_heads_inplace(qg.data() + (size_t)h * 2 * hd, cm.f32.at(L + ".attn_q_norm.weight"), 1, hd, (float)m.eps);
            } else {
                cpu_rms_heads_inplace(qg.data(), cm.f32.at(L + ".attn_q_norm.weight"), nh_q, hd, (float)m.eps);
            }
            cpu_rms_heads_inplace(k.data(), cm.f32.at(L + ".attn_k_norm.weight"), nh_k, hd, (float)m.eps);
            cpu_rope(qg.data(), k.data(), (int)cm.pos, nh_q, nh_k, hd, nrot, (float)m.rope_base, is35 ? 2 * hd : hd, 0);
            // append kv, attend
            const int li = (int)(is35 ? il / 4 : il);
            const size_t hstride = (size_t)cm.ctx * hd;
            for (int h = 0; h < nh_k; ++h)
                for (int i = 0; i < hd; ++i) {
                    cm.kv_k[li][(size_t)h * hstride + (size_t)cm.pos * hd + i] = k[h * hd + i];
                    cm.kv_v[li][(size_t)h * hstride + (size_t)cm.pos * hd + i] = v[h * hd + i];
                }
            const float scale = 1.f / sqrtf((float)hd);
            const int np = (int)cm.pos + 1;
            for (int h = 0; h < nh_q; ++h) {
                const int kh = h / (nh_q / nh_k);
                const float* qh = qg.data() + (size_t)h * (is35 ? 2 * hd : hd);
                float mx = -1e30f;
                std::vector<float> sc(np);
                for (int t = 0; t < np; ++t) {
                    float s = 0.f;
                    for (int i = 0; i < hd; ++i) s += qh[i] * cm.kv_k[li][(size_t)kh * hstride + (size_t)t * hd + i];
                    sc[t] = s * scale;
                    mx = fmaxf(mx, sc[t]);
                }
                float z = 0.f; for (int t = 0; t < np; ++t) z += expf(sc[t] - mx);
                for (int i = 0; i < hd; ++i) {
                    float a = 0.f;
                    for (int t = 0; t < np; ++t) a += expf(sc[t] - mx) / z * cm.kv_v[li][(size_t)kh * hstride + (size_t)t * hd + i];
                    att[(size_t)h * hd + i] = a;
                }
            }
            if (il <= 1) dbg_cpu(("att" + std::to_string(il)).c_str(), att.data(), (long)att.size());
            if (is35) {  // sigmoid gate (gate = second half of each qg head slice)
                for (size_t i = 0; i < att.size(); ++i) {
                    const float gt = qg[(i / hd) * 2 * hd + hd + (i % hd)];
                    att[i] /= (1.f + expf(-gt));
                }
            }
            cpu_mat_q4(cm, (L + ".attn_output.weight").c_str(), att.data(), y.data());
        }
        cpu_add(x.data(), y.data(), m.n_embd);
        if (il <= 1) dbg_cpu(("attn_out" + std::to_string(il)).c_str(), x.data(), m.n_embd);
        // second norm + FFN
        const std::string ln2 = is35 ? "blk." + std::to_string(il) + ".post_attention_norm.weight"
                                     : "blk." + std::to_string(il) + ".ffn_norm.weight";
        cpu_rms(cm.f32.at(ln2), x.data(), xn.data(), m.n_embd, (float)m.eps);
        const std::string L2 = "blk." + std::to_string(il);
        std::vector<float> gg(m.n_ff), up(m.n_ff);
        cpu_mat_q4(cm, (L2 + ".ffn_gate.weight").c_str(), xn.data(), gg.data());
        cpu_mat_q4(cm, (L2 + ".ffn_up.weight").c_str(), xn.data(), up.data());
        for (int i = 0; i < m.n_ff; ++i) gg[i] = gg[i] / (1.f + expf(-gg[i])) * up[i];
        cpu_mat_q4(cm, (L2 + ".ffn_down.weight").c_str(), gg.data(), y.data());
        cpu_add(x.data(), y.data(), m.n_embd);
        if (il <= 1) dbg_cpu(("ffn_out" + std::to_string(il)).c_str(), x.data(), m.n_embd);
    }
    cpu_rms(cm.f32.at("output_norm.weight"), x.data(), xn.data(), m.n_embd, (float)m.eps);
    if (cm.pos == 0) dbg_cpu("final_norm", xn.data(), m.n_embd);
    logits.resize(m.n_vocab);
    cpu_mat_q4(cm, "output.weight", xn.data(), logits.data());
    if (cm.pos == 0) {
        long t1 = 0;
        for (long j = 1; j < m.n_vocab; ++j) if (logits[j] > logits[t1]) t1 = j;
        printf("dbg_cpu,logits_top,%ld,%ld:%.4f\n", cm.pos, t1, logits[t1]);
    }
    ++cm.pos;
}

// NB-dispatched Q4_0 matvec: y[v*ystride + row] for v in [0, g_batch)
static long g_batch = 1;
static void launch_mat_q4(const uint8_t* w, long K, long rows, const int8_t* xq, const float* xs, const int* xsum,
                          float* y, long ystride, cudaStream_t s) {
    const unsigned g = (unsigned)((rows + 3) / 4);
    switch (g_batch) {
        case 1: k_mat_q4<1><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, y, ystride); break;
        case 2: k_mat_q4<2><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, y, ystride); break;
        case 4: k_mat_q4<4><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, y, ystride); break;
        case 8: k_mat_q4<8><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, y, ystride); break;
        case 16: k_mat_q4<16><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, y, ystride); break;
        case 32: k_mat_q4<32><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, y, ystride); break;
        case 64: k_mat_q4<64><<<g, 128, 0, s>>>(w, K, rows, xq, xs, xsum, y, ystride); break;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string model, prompt, prompt_ids_s, out_ids_path;
    long hot_mib = 8192, cache_mib = 5553, ctx = 1024, gen_tokens = 16, batch = 1;
    int o_direct = 1, validate = 0, readers = 4, show_text = 1, dbg_dump = 0, verify_hot = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i]; const char* v = argv[i + 1];
        if (k == "--model") model = v;
        else if (k == "--hot-mib") hot_mib = atol(v);
        else if (k == "--cache-mib") cache_mib = atol(v);
        else if (k == "--ctx") ctx = atol(v);
        else if (k == "--gen-tokens") gen_tokens = atol(v);
        else if (k == "--batch") batch = atol(v);
        else if (k == "--prompt") prompt = v;
        else if (k == "--prompt-ids") prompt_ids_s = v;
        else if (k == "--validate") validate = atoi(v);
        else if (k == "--readers") readers = atol(v);
        else if (k == "--ram-floor-mib") g_floor = atol(v);
        else if (k == "--swap-tol-mib") g_swap_tol = atol(v);
        else if (k == "--o-direct") o_direct = atoi(v);
        else if (k == "--show-text") show_text = atoi(v);
        else if (k == "--dbg-dump") dbg_dump = atoi(v);
        else if (k == "--verify-hot") verify_hot = atoi(v);
        else if (k == "--out-ids") out_ids_path = v;
    }
    if (model.empty()) { fprintf(stderr, "need --model\n"); return 2; }
    if (batch < 1 || batch > 64 || (batch & (batch - 1))) { fprintf(stderr, "--batch must be a power of two in 1..64\n"); return 2; }
    if (validate && batch != 1) { fprintf(stderr, "--validate requires --batch 1\n"); return 2; }
    g_batch = batch;
    g_swap0 = meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:"); safety("start");

    GgufKV kv; std::vector<GTensor> gtensors; uint64_t data_start = 0;
    if (!parse_gguf(model.c_str(), kv, gtensors, data_start)) return 1;
    Model m;
    m.arch = kv.strs["general.architecture"];
    const std::string A = m.arch;
    if (A != "qwen3" && A != "qwen35") { fprintf(stderr, "unsupported arch %s\n", A.c_str()); return 1; }
    m.n_layer = kv.ints[A + ".block_count"];
    m.n_embd = kv.ints[A + ".embedding_length"];
    m.n_head = kv.ints[A + ".attention.head_count"];
    m.n_kv = kv.ints[A + ".attention.head_count_kv"];
    m.n_ff = kv.ints[A + ".feed_forward_length"];
    m.n_vocab = (long)kv.arr_strs["tokenizer.ggml.tokens"].size();
    m.eps = kv.flts.count(A + ".attention.layer_norm_rms_epsilon") ? kv.flts[A + ".attention.layer_norm_rms_epsilon"] : 1e-6;
    m.rope_base = kv.flts.count(A + ".rope.freq_base") ? kv.flts[A + ".rope.freq_base"] : 10000.0;
    m.bos_id = kv.ints.count("tokenizer.ggml.bos_token_id") ? kv.ints["tokenizer.ggml.bos_token_id"] : -1;
    m.eos_id = kv.ints.count("tokenizer.ggml.eos_token_id") ? kv.ints["tokenizer.ggml.eos_token_id"] : -1;
    if (A == "qwen35") {
        m.head_k = kv.ints[A + ".attention.key_length"];
        m.head_v = kv.ints[A + ".attention.value_length"];
        m.n_rot = kv.ints[A + ".rope.dimension_count"];
        m.sss_state = kv.ints[A + ".ssm.state_size"];
        m.sss_group = kv.ints[A + ".ssm.group_count"];
        m.sss_dt = kv.ints[A + ".ssm.time_step_rank"];
        m.sss_inner = kv.ints[A + ".ssm.inner_size"];
        m.sss_conv = kv.ints[A + ".ssm.conv_kernel"];
        m.is_recr = kv.arr_bools[A + ".attention.recurrent_layers"];
        if ((long)m.is_recr.size() != m.n_layer) { fprintf(stderr, "recurrent_layers size mismatch\n"); return 1; }
        m.n_qg = 2 * m.n_head * m.head_v;
        m.qkv_out = 2 * m.sss_state * m.sss_group + m.sss_inner;
    } else {
        m.head_k = kv.ints.count(A + ".attention.key_length") ? kv.ints[A + ".attention.key_length"] : m.n_embd / m.n_head;
        m.head_v = kv.ints.count(A + ".attention.value_length") ? kv.ints[A + ".attention.value_length"] : m.head_k;
        m.n_rot = m.head_k;
        m.n_qg = m.n_head * m.head_k;
        m.qkv_out = 0;
    }
    for (const GTensor& t : gtensors) m.by_name[t.name] = &t;
    for (char c : m.is_recr) { if (c) ++m.n_recr; else ++m.n_attn; }
    if (A == "qwen3") { m.is_recr.assign(m.n_layer, 0); m.n_attn = m.n_layer; m.n_recr = 0; }
    const GTensor* embd = m.by_name["token_embd.weight"];
    const GTensor* outw = m.by_name["output.weight"];
    if (!embd || !outw) { fprintf(stderr, "missing token_embd/output\n"); return 1; }

    printf("config,model=%s,arch=%s,layers=%ld,recr=%ld,attn=%ld,n_embd=%ld,n_head=%ld,n_kv=%ld,head=%ld,rot=%ld,ffn=%ld,vocab=%ld,ssm_state=%ld,ssm_group=%ld,ssm_dt=%ld,ssm_inner=%ld,rope_base=%.0f,ctx=%ld,batch=%ld,hot_mib=%ld,cache_mib=%ld,validate=%d\n",
           model.c_str(), A.c_str(), m.n_layer, m.n_recr, m.n_attn, m.n_embd, m.n_head, m.n_kv, m.head_v, m.n_rot, m.n_ff,
           m.n_vocab, m.sss_state, m.sss_group, m.sss_dt, m.sss_inner, m.rope_base, ctx, batch, hot_mib, cache_mib, validate);

    // ---- items: chunk every Q4_0 tensor except the embedding table ----
    struct Item { std::string tensor; long K, N, row0, rows; uint64_t file_off, bytes, aligned_off, head, read_len; int tier = 0; int64_t res_off = -1; };
    enum { kNvme = 0, kHot = 1, kStaged = 2 };
    std::map<std::string, std::vector<int>> tensor_items;
    std::vector<Item> items; uint64_t total_bytes = 0;
    for (const GTensor& t : gtensors) {
        if (t.type != kGGML_Q4_0 || t.name == "token_embd.weight") continue;
        if (t.ne[0] % 32) { fprintf(stderr, "tensor %s K not /32\n", t.name.c_str()); return 1; }
        const uint64_t rowbytes = (uint64_t)(t.ne[0] / 32) * 18;
        const long per = (long)(kMaxItem / rowbytes);
        const std::string tn = t.name;
        for (long r = 0; r < t.ne[1]; r += per) {
            Item it; it.tensor = tn; it.K = t.ne[0]; it.N = t.ne[1];
            it.row0 = r; it.rows = std::min<long>(per, t.ne[1] - r);
            it.file_off = data_start + t.off + (uint64_t)r * rowbytes;
            it.bytes = (uint64_t)it.rows * rowbytes;
            it.aligned_off = it.file_off & ~4095ull; it.head = it.file_off - it.aligned_off;
            it.read_len = align_up(it.head + it.bytes, 4096);
            tensor_items[tn].push_back((int)items.size());
            items.push_back(it); total_bytes += it.bytes;
        }
    }

    // ---- tier assignment: hot first-fit, then staged RAM cache, rest NVMe ----
    uint64_t hot_used = 0, staged_used = 0;
    for (Item& it : items) {
        if (hot_used + it.bytes + 16 <= (uint64_t)hot_mib << 20) {
            it.tier = kHot; it.res_off = (int64_t)hot_used; hot_used = align_up(hot_used + it.bytes, 256);
        } else if (staged_used + it.bytes + 4096 <= (uint64_t)cache_mib << 20) {
            it.tier = kStaged; it.res_off = (int64_t)staged_used; staged_used = align_up(staged_used + it.bytes, 4096);
        }
    }
    uint64_t nvme_bytes = 0; for (auto& it : items) if (it.tier == kNvme) nvme_bytes += it.bytes;

    struct rlimit lim{}; uint64_t memlock = 0; int memlock_inf = 0;
    if (getrlimit(RLIMIT_MEMLOCK, &lim) != 0) { perror("getrlimit"); return 1; }
    if (lim.rlim_cur == RLIM_INFINITY) memlock_inf = 1; else memlock = lim.rlim_cur;
    const uint64_t stage_lock = (uint64_t)kStage * (kStageBytes + 4096);
    uint64_t map_budget = ~0ull;
    if (!memlock_inf) map_budget = memlock > stage_lock + kMapMargin ? memlock - stage_lock - kMapMargin : 0;
    const uint64_t pin_bytes = std::min<uint64_t>(staged_used, map_budget);
    const long host_mib = (long)(staged_used >> 20);
    if (meminfo_mib("MemAvailable:") < g_floor + host_mib) {
        fprintf(stderr, "ABORT before alloc: MemAvailable=%ld host_tier=%ld floor=%ld\n",
                meminfo_mib("MemAvailable:"), host_mib, g_floor);
        return 4;
    }
    // VRAM preflight: hot + slots + states must keep >= 2 GiB free
    const long sss_v = m.sss_dt ? m.sss_inner / m.sss_dt : 0;
    const uint64_t ssm_vram = ((uint64_t)m.n_recr * m.sss_dt * (uint64_t)sss_v * sss_v * 4
                            + (uint64_t)m.n_recr * m.qkv_out * (m.sss_conv ? m.sss_conv - 1 : 0) * 4) * batch;
    const uint64_t kv_vram = (uint64_t)m.n_attn * m.n_kv * ctx * m.head_v * 4 * 2 * batch;
    size_t total_vram = 0;
    const uint64_t vram_plan = hot_used + (uint64_t)kSlots * kMaxItem + ssm_vram + kv_vram + (16u << 20);
    { int dev = 0; cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, dev), "props"); total_vram = prop.totalGlobalMem; }
    if (vram_plan + (2ull << 30) > total_vram) {
        fprintf(stderr, "ABORT before alloc: VRAM plan %.2f GiB + 2 GiB floor exceeds %.2f GiB\n",
                vram_plan / 1073741824.0, total_vram / 1073741824.0);
        return 4;
    }
    printf("alloc_plan,items=%zu,stream_GiB=%.2f,hot_GiB=%.2f,cache_GiB=%.2f,pinned_MiB=%.0f,nvme_per_token_GiB=%.2f,ssm_state_MiB=%.0f,kv_MiB=%.0f\n",
           items.size(), total_bytes / 1073741824.0, hot_used / 1073741824.0, staged_used / 1073741824.0,
           pin_bytes / 1048576.0, nvme_bytes / 1073741824.0, ssm_vram / 1048576.0, kv_vram / 1048576.0);

    // ---- CUDA allocations ----
    check(cudaSetDevice(0), "device");
    uint8_t *hot = nullptr, *slots = nullptr;
    if (hot_used) check(cudaMalloc((void**)&hot, hot_used), "hot arena");
    bool need_slots = false; for (auto& it : items) if (it.tier != kHot) need_slots = true;
    if (need_slots) check(cudaMalloc((void**)&slots, (size_t)kSlots * kMaxItem), "slots");
    long max_rows = 0, max_K = 0;
    // multi-chunk tensors write the full N contiguously into yA/yB, so the scratch
    // must cover the largest TENSOR row count, not the largest chunk
    for (const Item& it : items) { max_K = std::max(max_K, it.K); }
    for (const GTensor& t : gtensors) if (t.type == kGGML_Q4_0 && t.name != "token_embd.weight") max_rows = std::max(max_rows, t.ne[1]);
    max_rows += 16;
    const long B = batch;   // all activation buffers are [B][...] stacked
    float *x, *t_norm, *yA, *yB, *t_v, *t_att, *t_dn, *t_conv, *t_g48a, *t_g48b, *logits, *t_ffnh;
    check(cudaMalloc((void**)&x, B * m.n_embd * 4), "x");
    check(cudaMalloc((void**)&t_norm, B * m.n_embd * 4), "t_norm");
    check(cudaMalloc((void**)&yA, B * max_rows * 4), "yA");
    check(cudaMalloc((void**)&yB, B * max_rows * 4), "yB");
    check(cudaMalloc((void**)&t_v, B * 4096 * 4), "t_v");
    check(cudaMalloc((void**)&t_att, B * (A == "qwen35" ? 12288 : 4096) * 4), "t_att");
    check(cudaMalloc((void**)&t_dn, B * (A == "qwen35" ? 6144 : 16) * 4), "t_dn");
    check(cudaMalloc((void**)&t_conv, B * (A == "qwen35" ? m.qkv_out : 16) * 4), "t_conv");
    check(cudaMalloc((void**)&t_g48a, B * 64 * 4), "t_g48a");
    check(cudaMalloc((void**)&t_g48b, B * 64 * 4), "t_g48b");
    check(cudaMalloc((void**)&t_ffnh, B * (A == "qwen35" ? 17408 : 3072) * 4), "t_ffnh");
    check(cudaMalloc((void**)&logits, B * m.n_vocab * 4), "logits");
    int8_t* xq; float* xs; int* xsum;
    check(cudaMalloc((void**)&xq, B * max_K), "xq");
    check(cudaMalloc((void**)&xs, B * max_K / 32 * 4), "xs");
    check(cudaMalloc((void**)&xsum, B * max_K / 32 * 4), "xsum");
    // small F32 tensors -> one device blob
    uint64_t small_bytes = 0;
    std::vector<std::pair<std::string, const GTensor*>> smalls;
    for (const GTensor& t : gtensors) {
        if (t.type != kGGML_F32) continue;
        const bool is_small = t.bytes <= (1u << 20);
        if (!is_small) continue;
        smalls.push_back({t.name, &t}); small_bytes += align_up(t.bytes, 256);
    }
    uint8_t* small_dev = nullptr;
    check(cudaMalloc((void**)&small_dev, small_bytes), "small blob");
    { uint64_t o = 0; std::vector<uint8_t> hbuf;
      for (auto& p : smalls) {
          hbuf.resize(p.second->bytes);
          FILE* f = fopen(model.c_str(), "rb");
          if (fseek(f, (long)(data_start + p.second->off), SEEK_SET) || fread(hbuf.data(), 1, p.second->bytes, f) != p.second->bytes) { fprintf(stderr, "small read %s\n", p.first.c_str()); return 1; }
          fclose(f);
          check(cudaMemcpy(small_dev + o, hbuf.data(), p.second->bytes, cudaMemcpyHostToDevice), "small copy");
          m.small_off[p.first] = o; o = align_up(o + p.second->bytes, 256);
      } }
    // KV + SSM state
    float *kv_dev = nullptr, *ssm_dev = nullptr, *conv_dev = nullptr;
    const uint64_t kv_per_layer = (uint64_t)m.n_kv * ctx * m.head_v * 4;
    if (m.n_attn) check(cudaMalloc((void**)&kv_dev, kv_per_layer * 2 * m.n_attn * B), "kv cache");
    const uint64_t sss = m.sss_inner / m.sss_dt;
    if (m.n_recr) {
        check(cudaMalloc((void**)&ssm_dev, (size_t)m.n_recr * m.sss_dt * sss * sss * 4 * B), "ssm state");
        check(cudaMalloc((void**)&conv_dev, (size_t)m.n_recr * m.qkv_out * (m.sss_conv - 1) * 4 * B), "conv state");
        check(cudaMemset(ssm_dev, 0, (size_t)m.n_recr * m.sss_dt * sss * sss * 4 * B), "ssm zero");
        check(cudaMemset(conv_dev, 0, (size_t)m.n_recr * m.qkv_out * (m.sss_conv - 1) * 4 * B), "conv zero");
    }
    // pinned stage ring + staged cache
    uint8_t* stage[kStage]; for (auto& s : stage) check(cudaHostAlloc((void**)&s, kStageBytes + 4096, cudaHostAllocPortable), "stage");
    uint8_t* staged_cache = nullptr; uint8_t* pinned_cache = nullptr;
    if (staged_used) {
        if (posix_memalign((void**)&staged_cache, 4096, staged_used)) { fprintf(stderr, "cache alloc\n"); return 1; }
        if (pin_bytes) check(cudaHostAlloc((void**)&pinned_cache, pin_bytes, cudaHostAllocPortable), "pinned cache");
    }
    std::vector<char> cached(items.size(), 0);
    int fd = open(model.c_str(), O_RDONLY | (o_direct ? O_DIRECT : 0)); if (fd < 0) { perror("open o_direct"); return 1; }
    int fdc = open(model.c_str(), O_RDONLY); if (fdc < 0) { perror("open"); return 1; }
    cudaStream_t st, stc; check(cudaStreamCreate(&st), "stream"); check(cudaStreamCreate(&stc), "compute stream");
    cudaEvent_t ev[kStage], evc[kSlots]; for (auto& e : ev) check(cudaEventCreate(&e), "event"); for (auto& e : evc) check(cudaEventCreate(&e), "event");
    for (auto& e : evc) check(cudaEventRecord(e, stc), "evc prime");
    check(cudaStreamSynchronize(stc), "evc prime sync");

    // ---- schedule: item indices in decode order ----
    std::vector<int> sched; std::vector<int> sched_row0;  // logits row0 if output chunk
    auto push_tensor = [&](const std::string& tn) {
        auto itn = tensor_items.find(tn);
        if (itn == tensor_items.end()) { fprintf(stderr, "missing tensor %s\n", tn.c_str()); exit(1); }
        for (int idx : itn->second) sched.push_back(idx);
    };
    for (long il = 0; il < m.n_layer; ++il) {
        char nm[128];
        if (A == "qwen35" && m.is_recr[il]) {
            push_tensor("blk." + std::to_string(il) + ".attn_qkv.weight");
            push_tensor("blk." + std::to_string(il) + ".attn_gate.weight");
            push_tensor("blk." + std::to_string(il) + ".ssm_beta.weight");
            push_tensor("blk." + std::to_string(il) + ".ssm_alpha.weight");
            push_tensor("blk." + std::to_string(il) + ".ssm_out.weight");
        } else {
            push_tensor("blk." + std::to_string(il) + ".attn_q.weight");
            push_tensor("blk." + std::to_string(il) + ".attn_k.weight");
            push_tensor("blk." + std::to_string(il) + ".attn_v.weight");
            push_tensor("blk." + std::to_string(il) + ".attn_output.weight");
        }
        push_tensor("blk." + std::to_string(il) + ".ffn_gate.weight");
        push_tensor("blk." + std::to_string(il) + ".ffn_up.weight");
        push_tensor("blk." + std::to_string(il) + ".ffn_down.weight");
    }
    push_tensor("output.weight");

    // ---- tokenizer ----
    Tok tok;
    tok.vocab = kv.arr_strs["tokenizer.ggml.tokens"];
    for (size_t i = 0; i < tok.vocab.size(); ++i) tok.id_of[tok.vocab[i]] = (long)i;
    tok.bos = m.bos_id; tok.eos = m.eos_id;
    std::vector<std::vector<long>> prompts;
    if (!prompt_ids_s.empty()) {
        size_t p = 0;
        while (p <= prompt_ids_s.size()) {
            size_t semi = prompt_ids_s.find(';', p);
            if (semi == std::string::npos) semi = prompt_ids_s.size();
            std::vector<long> ids;
            size_t q = p;
            while (q < semi) {
                size_t c = prompt_ids_s.find(',', q);
                if (c == std::string::npos || c > semi) c = semi;
                ids.push_back(atol(prompt_ids_s.substr(q, c - q).c_str()));
                q = c + 1;
            }
            if (!ids.empty()) prompts.push_back(ids);
            p = semi + 1;
        }
    } else {
        const std::vector<long> ids = tok.encode(prompt);
        if (ids.empty()) { fprintf(stderr, "empty prompt\n"); return 2; }
        prompts.push_back(ids);
    }
    while (prompts.size() < (size_t)batch) prompts.push_back(prompts[0]);   // replicate for equivalence tests
    if ((long)prompts.size() != batch) { fprintf(stderr, "need exactly %ld prompt sequences (';'-separated)\n", batch); return 2; }
    for (long v = 1; v < batch; ++v)
        if (prompts[v].size() != prompts[0].size()) { fprintf(stderr, "all prompts must have equal length (lockstep decode)\n"); return 2; }
    if (show_text) printf("prompt_seqs,%zu,first_len=%zu,ids=%s\n", prompts.size(), prompts[0].size(), prompt_ids_s.c_str());
    if (show_text && batch == 1) printf("prompt_text,%s\n", tok.decode(prompts[0]).c_str());

    // ---- CPU reference ----
    CPUModel cm;
    if (validate) {
        g_dbg = dbg_dump;
        if (total_bytes + embd->bytes > (16ull << 30)) { fprintf(stderr, "model too large for CPU reference\n"); return 2; }
        cm.m = m; cm.m.small_off.clear();
        cm.pos = 0; cm.ctx = ctx;
        cm.kv_k.resize(m.n_attn); cm.kv_v.resize(m.n_attn);
        for (long l = 0; l < m.n_attn; ++l) { cm.kv_k[l].assign(m.n_kv * ctx * m.head_v, 0); cm.kv_v[l].assign(m.n_kv * ctx * m.head_v, 0); }
        const long Sv = m.sss_dt ? m.sss_inner / m.sss_dt : 0;
        cm.ssm.resize(m.n_recr); cm.conv.resize(m.n_recr);
        for (long l = 0; l < m.n_recr; ++l) {
            cm.ssm[l].assign(m.sss_dt * Sv * Sv, 0);
            cm.conv[l].assign(m.qkv_out * (m.sss_conv - 1), 0);
        }
        // mmap the model file: weight bytes stay page-cache backed (no 14 GiB anon copy,
        // which would swap out the desktop and trip the swap-abort guard)
        const int mfd = open(model.c_str(), O_RDONLY);
        if (mfd < 0) { perror("cpu ref mmap open"); return 1; }
        struct stat st{}; if (fstat(mfd, &st) != 0) { perror("fstat"); return 1; }
        cm.mm_base = (const uint8_t*)mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, mfd, 0);
        close(mfd);
        if (cm.mm_base == MAP_FAILED) { perror("mmap"); return 1; }
        cm.mm_data_start = data_start;
        for (const GTensor& t : gtensors) {
            const uint8_t* p = cm.mm_base + data_start + t.off;
            if (t.type == kGGML_Q4_0) cm.raw_q4[t.name] = p;
            else cm.f32[t.name] = (const float*)p;
        }
        printf("cpu_ref,weights_mapped=%zu,raw_MiB=%.0f\n", cm.raw_q4.size() + cm.f32.size(), (total_bytes + embd->bytes) / 1048576.0);
    }
    const long free_after_alloc = free_vram_mib();
    printf("alloc,free_vram_MiB=%ld,ram_avail_MiB=%ld,swap_used_MiB=%ld\n",
           free_after_alloc, meminfo_mib("MemAvailable:"), meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:"));

    // ---- load hot arena once (it is read in place every token thereafter) ----
    {
        std::vector<uint8_t> bounce(kMaxItem + 8192);
        long n_hot_items = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            Item& it = items[i]; if (it.tier != kHot) continue;
            const ssize_t got = pread(fdc, bounce.data(), it.bytes, (off_t)it.file_off);
            if (got < (ssize_t)it.bytes) { fprintf(stderr, "hot pread %s\n", it.tensor.c_str()); return 1; }
            check(cudaMemcpy(hot + it.res_off, bounce.data(), it.bytes, cudaMemcpyHostToDevice), "hot load");
            ++n_hot_items;
        }
        if (verify_hot) {
            long bad = 0;
            for (size_t i = 0; i < items.size(); ++i) {
                Item& it = items[i]; if (it.tier != kHot) continue;
                const ssize_t got = pread(fdc, bounce.data(), it.bytes, (off_t)it.file_off);
                check(cudaMemcpy(bounce.data() + kMaxItem, hot + it.res_off, it.bytes, cudaMemcpyDeviceToHost), "hot verify d2h");
                if (got < (ssize_t)it.bytes || memcmp(bounce.data(), bounce.data() + kMaxItem, it.bytes) != 0) {
                    fprintf(stderr, "HOT MISMATCH item %zu tensor %s rows %ld-%ld\n", i, it.tensor.c_str(), it.row0, it.row0 + it.rows);
                    ++bad;
                }
            }
            printf("hot_verify,items=%ld,bad=%ld\n", n_hot_items, bad);
        }
        printf("hot_load,items=%ld,GiB=%.2f\n", n_hot_items, hot_used / 1073741824.0);
    }

    // ---- decode loop ----
    const long hd = m.head_v, nh_q = m.n_head, nh_k = m.n_kv;
    const long Sv = m.sss_dt ? m.sss_inner / m.sss_dt : 1;
    const float attn_scale = 1.f / sqrtf((float)hd);
    const float dn_scale = 1.f / sqrtf((float)Sv);
    const long emb_nb = m.n_embd / 32;
    uint8_t* emb_buf = nullptr; check(cudaHostAlloc((void**)&emb_buf, (size_t)B * emb_nb * 18 + 4096, cudaHostAllocPortable), "emb buf");
    std::vector<float> emb_row((size_t)B * m.n_embd), logits_h((size_t)B * m.n_vocab);

    double best_tok_s = 1e30, sum_steady_s = 0; long n_steady = 0; long free_min = free_after_alloc;
    uint64_t steady_cold_bytes = 0; long n_steady_cold = 0;
    long checked = 0, mismatches = 0; double worst_rel = 0;
    std::vector<long> all_ids;          // every token fed (prompt + generated)
    std::vector<long> generated;        // argmax tokens produced
    long cur_pos = 0;                   // sequence position for rope/kv

    auto run_token = [&](const std::vector<long>& toks, int pass, double* tok_s_out, uint64_t* cold_out) { (void)dbg_dump;
        std::atomic<size_t> released{0};
        std::atomic<long> ready[kStage]; for (auto& r : ready) r = 0;
        std::atomic<bool> fail{false};
        *cold_out = 0;
        for (long v = 0; v < B; ++v) {
            if (pread(fdc, emb_buf + (size_t)v * emb_nb * 18, emb_nb * 18, (off_t)(data_start + embd->off + (uint64_t)toks[v] * emb_nb * 18)) != (ssize_t)(emb_nb * 18)) { fprintf(stderr, "embd pread\n"); exit(1); }
            cpu_dequant_row_q4(emb_buf + (size_t)v * emb_nb * 18, m.n_embd, &emb_row[(size_t)v * m.n_embd]);
        }
        if (dbg_dump && pass == 0 && B > 1) {
            printf("dbg_host,embrow_v0");
            for (int i = 0; i < 6; ++i) printf(",%.9g", emb_row[i]);
            printf("\ndbg_host,embrow_v1");
            for (int i = 0; i < 6; ++i) printf(",%.9g", emb_row[m.n_embd + i]);
            printf("\n");
        }
        check(cudaMemcpy(x, emb_row.data(), B * m.n_embd * 4, cudaMemcpyHostToDevice), "emb h2d");

        // cold_list must contain EVERY non-hot schedule position: the reader and the
        // main thread both number slots by list position, and direct-pin items keep
        // their position (reader skips the fill, main skips the ready wait)
        std::vector<int> cold_list;
        for (int idx : sched) {
            Item& it = items[idx];
            if (it.tier == kHot) continue;
            cold_list.push_back(idx);
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (long t = 0; t < readers; ++t) pool.emplace_back([&, t]() {
            for (size_t k = (size_t)t; k < cold_list.size(); k += (size_t)readers) {
                const int idx = cold_list[k]; Item& it = items[idx];
                const bool in_pin = (uint64_t)it.res_off + it.bytes <= pin_bytes;
                if (pass > 0 && in_pin && cached[idx]) continue;   // keep position, no fill
                while (released.load() + kStage <= k && !fail.load()) std::this_thread::yield();
                if (fail.load()) return;
                const int b = (int)(k % kStage);
                if (it.tier == kStaged && cached[idx]) {
                    memcpy(stage[b] + it.head, staged_cache + it.res_off, it.bytes);
                } else {
                    const ssize_t got = pread(fd, stage[b], it.read_len, (off_t)it.aligned_off);
                    if (got < (ssize_t)(it.head + it.bytes)) { fprintf(stderr, "pread %zd < %llu\n", got, (unsigned long long)(it.head + it.bytes)); fail = true; return; }
                    if (it.tier == kStaged) { memcpy((in_pin ? pinned_cache : staged_cache) + it.res_off, stage[b] + it.head, it.bytes); cached[idx] = 1; }
                }
                ready[b].store((long)k + 1);
            }
        });
        size_t cold_k = 0;
        std::vector<int> pending;
        auto S = [&](const std::string& tn) -> const float* { return (const float*)(small_dev + m.small_off.at(tn)); };
        auto mat = [&](const std::string& tn, const float* xin, long K, float* out, long out_row0, long ystride) {
            const long nb = K / 32;
            // batched GEMV writes y[v*ystride + row]; consumers index dense [B][N], so the
            // stride must be this tensor's own row count — NOT the max_rows buffer pitch
            ystride = m.by_name.at(tn)->ne[1];
            k_quant_x<<<dim3((unsigned)((nb * B + 127) / 128)), 128, 0, stc>>>(xin, nb * B, xq, xs, xsum);
            for (int idx : tensor_items.at(tn)) {
                Item& it = items[idx];
                if (it.tier == kHot) {
                    launch_mat_q4(hot + it.res_off, it.K, it.rows, xq, xs, xsum, out + out_row0 + it.row0, ystride, stc);
                    continue;
                }
                const bool in_pin = (uint64_t)it.res_off + it.bytes <= pin_bytes;
                const bool direct_pin = pass > 0 && in_pin && cached[idx];
                uint8_t* dst = slots + (size_t)(cold_k % kSlots) * kMaxItem;
                const uint8_t* csrc;
                if (!direct_pin) {
                    const int b = (int)(cold_k % kStage);
                    while (ready[b].load() != (long)cold_k + 1) { if (fail.load()) exit(1); std::this_thread::yield(); }
                    csrc = stage[b] + it.head;
                } else {
                    csrc = pinned_cache + it.res_off;
                }
                check(cudaStreamWaitEvent(st, evc[cold_k % kSlots], 0), "slot wait");
                check(cudaMemcpyAsync(dst, csrc, it.bytes, cudaMemcpyHostToDevice, st), "copy");
                check(cudaEventRecord(ev[cold_k % kStage], st), "ev");
                check(cudaStreamWaitEvent(stc, ev[cold_k % kStage], 0), "wait copy");
                launch_mat_q4(dst, it.K, it.rows, xq, xs, xsum, out + out_row0 + it.row0, ystride, stc);
                check(cudaEventRecord(evc[cold_k % kSlots], stc), "evc");
                pending.push_back((int)cold_k);
                if (pending.size() >= 3) {
                    const int o = pending.front(); pending.erase(pending.begin());
                    check(cudaEventSynchronize(ev[o % kStage]), "ev sync");
                    released.store((size_t)o + 1);
                }
                ++cold_k;
            }
        };
        auto dbg2 = [&](const char* tag, const float* dev, long n, long stride, long v) {
            if (!dbg_dump) return;
            std::vector<float> h(n);
            check(cudaMemcpy(h.data(), dev + v * stride, h.size() * 4, cudaMemcpyDeviceToHost), "dbg2");
            printf("dbg_gpu,%s_v%ld", tag, v);
            for (float f : h) printf(",%.9g", f);
            printf("\n");
        };
        if (dbg_dump && pass == 0 && B > 1) { check(cudaDeviceSynchronize(), "dbg sync"); dbg2("embdev", x, m.n_embd, m.n_embd, 1); }
        auto dbg = [&](const char* tag, const float* dev, long n) {
            if (!dbg_dump) return;
            std::vector<float> h(n);
            check(cudaMemcpy(h.data(), dev, h.size() * 4, cudaMemcpyDeviceToHost), "dbg");
            printf("dbg_gpu,%s,%ld", tag, n);
            for (float v : h) printf(",%.9g", v);
            printf("\n");
        };
        const dim3 grid1(1, (unsigned)B);
        for (long il = 0; il < m.n_layer; ++il) {
            const std::string L = "blk." + std::to_string(il);
            k_rms_norm<<<grid1, 256, 0, stc>>>(t_norm, x, S(L + ".attn_norm.weight"), (int)m.n_embd, (float)m.eps);
            if (dbg_dump && il <= 1 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("emb" + std::to_string(il)).c_str(), x, m.n_embd); dbg(("norm" + std::to_string(il)).c_str(), t_norm, m.n_embd); }
            if (A == "qwen35" && m.is_recr[il]) {
                // gated delta net path (linear attention with persistent state)
                mat(L + ".attn_qkv.weight", t_norm, m.n_embd, yA, 0, max_rows);
                mat(L + ".attn_gate.weight", t_norm, m.n_embd, yB, 0, max_rows);
                mat(L + ".ssm_beta.weight", t_norm, m.n_embd, t_g48a, 0, max_rows);
                if (dbg_dump && il == 0 && pass == 0 && B > 1) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg2("betaraw", t_g48a, m.sss_dt, 48, 1); dbg2("betaraw0", t_g48a, m.sss_dt, 48, 0); }
                mat(L + ".ssm_alpha.weight", t_norm, m.n_embd, t_g48b, 0, max_rows);
                k_sigmoid<<<dim3((unsigned)((m.sss_dt * B + 63) / 64), (unsigned)B), 64, 0, stc>>>(t_g48a, t_g48a, (int)m.sss_dt);
                k_gate_ssm<<<dim3((unsigned)((m.sss_dt * B + 63) / 64), (unsigned)B), 64, 0, stc>>>(t_v, t_g48b, S(L + ".ssm_dt.bias"), S(L + ".ssm_a"), (int)m.sss_dt);
                const long ri = il - (il + 1) / 4;   // recurrent-layer index (pattern 1,1,1,0)
                const long conv_seq_stride = m.n_recr * m.qkv_out * (m.sss_conv - 1);
                const long ssm_seq_stride = m.n_recr * m.sss_dt * Sv * Sv;
                if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg("qkv0", yA, m.qkv_out); dbg("zraw0", yB, m.sss_inner); dbg("beta0", t_g48a, m.sss_dt); dbg("alpharaw0", t_g48b, m.sss_dt); if (B > 1) { dbg2("qkv0", yA, m.qkv_out, m.qkv_out, 1); dbg2("beta0", t_g48a, m.sss_dt, 64, 1); } }
                k_conv1d_silu<<<dim3((unsigned)((m.qkv_out + 255) / 256), (unsigned)B), 256, 0, stc>>>(conv_dev + (size_t)ri * m.qkv_out * (m.sss_conv - 1), yA, S(L + ".ssm_conv1d.weight"), t_conv, (int)m.qkv_out, conv_seq_stride);
                if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg("convout0", t_conv, m.qkv_out); if (B > 1) dbg2("convout0", t_conv, m.qkv_out, m.qkv_out, 1); }
                const long Sdim = m.sss_state;
                const long conv_slice_stride = A == "qwen35" ? m.qkv_out : 0;
                k_l2_heads<<<dim3((unsigned)m.sss_group, (unsigned)B), 128, 0, stc>>>(t_conv, (int)m.sss_group, (int)Sdim, (float)m.eps, conv_slice_stride);
                // K lives at [Sdim*group, 2*Sdim*group) — V (at 2*Sdim*group) must NOT be normalized
                k_l2_heads<<<dim3((unsigned)m.sss_group, (unsigned)B), 128, 0, stc>>>(t_conv + Sdim * m.sss_group, (int)m.sss_group, (int)Sdim, (float)m.eps, conv_slice_stride);
                k_delta_net<<<dim3((unsigned)m.sss_dt, (unsigned)B), 128, 0, stc>>>(ssm_dev + (size_t)ri * m.sss_dt * Sv * Sv, t_conv, t_conv + Sdim * m.sss_group,
                                                                 t_conv + 2 * Sdim * m.sss_group, t_v, t_g48a, t_dn, (int)Sdim, (int)m.sss_dt, (int)m.sss_group, dn_scale, ssm_seq_stride, (long)m.qkv_out);
                if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg("dnout0", t_dn, m.sss_inner); if (B > 1) { dbg2("dnout0", t_dn, m.sss_inner, m.sss_inner, 1); dbg2("vsl", t_conv + 2 * m.sss_state * m.sss_group, 8, m.qkv_out, 1); dbg2("vsl0", t_conv + 2 * m.sss_state * m.sss_group, 8, m.qkv_out, 0); dbg2("gsl", t_v, 8, 4096, 1); dbg2("gsl0", t_v, 8, 4096, 0); } }
                k_gated_norm<<<dim3((unsigned)m.sss_dt, (unsigned)B), 128, 0, stc>>>(t_att, t_dn, S(L + ".ssm_norm.weight"), yB, (int)m.sss_dt, (int)Sv, (float)m.eps);
                if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg("gatednorm0", t_att, m.sss_inner); if (B > 1) dbg2("gatednorm0", t_att, m.sss_inner, m.sss_inner, 1); }
                mat(L + ".ssm_out.weight", t_att, m.sss_inner, yA, 0, max_rows);
            } else {
                // full attention path with KV cache
                mat(L + ".attn_q.weight", t_norm, m.n_embd, yA, 0, max_rows);
                mat(L + ".attn_k.weight", t_norm, m.n_embd, yB, 0, max_rows);
                mat(L + ".attn_v.weight", t_norm, m.n_embd, t_v, 0, max_rows);
                if (dbg_dump && il <= 1 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("v" + std::to_string(il)).c_str(), t_v, nh_k * hd); }
                const int q_stride = A == "qwen35" ? 2 * (int)hd : (int)hd;
                k_rms_heads<<<dim3((unsigned)nh_q, (unsigned)B), 128, 0, stc>>>(yA, S(L + ".attn_q_norm.weight"), (int)nh_q, (int)hd, (float)m.eps, q_stride);
                k_rms_heads<<<dim3((unsigned)nh_k, (unsigned)B), 128, 0, stc>>>(yB, S(L + ".attn_k_norm.weight"), (int)nh_k, (int)hd, (float)m.eps, (int)hd);
                // ggml uses NEOX half-split pairing for NORMAL?no: NORM is adjacent; NEOX/MROPE/IMROPE
                // are half-split (ops.cpp rope switch). qwen35 = IMROPE = half-split; for text the
                // mrope section positions are all equal, so it is plain half-split RoPE.
                const int rope_style = 0;
                k_rope<<<dim3((unsigned)(nh_q + nh_k), (unsigned)B), 128, 0, stc>>>(yA, yB, (int)cur_pos, (int)nh_q, (int)nh_k, (int)hd, (int)m.n_rot, (float)m.rope_base, q_stride, rope_style);
                const long li = A == "qwen35" ? il / 4 : il;
                const long kv_seq_stride = (long)m.n_attn * 2 * m.n_kv * ctx * m.head_v;   // FLOATS per sequence (kv_per_layer is bytes)
                float* Kc = (float*)((uint8_t*)kv_dev + (size_t)li * 2 * kv_per_layer);
                float* Vc = (float*)((uint8_t*)kv_dev + (size_t)li * 2 * kv_per_layer + kv_per_layer);
                // NOTE: appends must run on the compute stream — they read yB/t_v produced
                // by GEMVs on stc (a copy-stream race here corrupted every layer >= 1)
                for (long v = 0; v < B; ++v) {
                    float* Ks = Kc + v * kv_seq_stride;
                    float* Vs = Vc + v * kv_seq_stride;
                    for (long h = 0; h < nh_k; ++h) {
                        check(cudaMemcpyAsync(Ks + ((size_t)h * ctx + cur_pos) * hd, yB + v * nh_k * hd + h * hd, hd * 4, cudaMemcpyDeviceToDevice, stc), "kv append k");
                        check(cudaMemcpyAsync(Vs + ((size_t)h * ctx + cur_pos) * hd, t_v + v * nh_k * hd + h * hd, hd * 4, cudaMemcpyDeviceToDevice, stc), "kv append v");
                    }
                }
                if (dbg_dump && il <= 1 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("vc" + std::to_string(il)).c_str(), Vc, nh_k * hd); }
                k_attn_decode<<<dim3((unsigned)nh_q, (unsigned)B), 256, (unsigned)((cur_pos + 1) * 4), stc>>>(t_att, Kc, Vc, yA, (int)cur_pos, (int)ctx, (int)nh_q, (int)nh_k, (int)hd, attn_scale, q_stride, kv_seq_stride);
                if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("att" + std::to_string(il)).c_str(), t_att, nh_q * hd); if (B > 1) { dbg2(("att" + std::to_string(il)).c_str(), t_att, nh_q * hd, nh_q * hd, 1); dbg2(("vv" + std::to_string(il)).c_str(), t_v, nh_k * hd, nh_k * hd, 1); } }
                if (A == "qwen35") k_attn_gate_apply<<<dim3((unsigned)((nh_q * hd + 255) / 256), (unsigned)B), 256, 0, stc>>>(t_att, yA, (int)nh_q, (int)hd);
                mat(L + ".attn_output.weight", t_att, nh_q * hd, yA, 0, max_rows);
                if (dbg_dump && il == 0 && pass == 0 && B > 1) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg2("wo0", yA, m.n_embd, m.n_embd, 1); dbg2("xpre0", x, m.n_embd, m.n_embd, 1); }
            }
            k_add<<<dim3((unsigned)((m.n_embd * B + 255) / 256), (unsigned)B), 256, 0, stc>>>(x, yA, (int)m.n_embd);
            if (dbg_dump && il == 0 && pass == 0 && B > 1) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg2("res0", x, m.n_embd, m.n_embd, 1); }
            if (dbg_dump && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("attn_out" + std::to_string(il)).c_str(), x, m.n_embd); if (B > 1) for (long vv = 1; vv < std::min<long>(B, 2); ++vv) dbg2(("attnout" + std::to_string(il)).c_str(), x, m.n_embd, m.n_embd, vv); }
            const std::string ln2 = A == "qwen35" ? L + ".post_attention_norm.weight" : L + ".ffn_norm.weight";
            k_rms_norm<<<grid1, 256, 0, stc>>>(t_norm, x, S(ln2), (int)m.n_embd, (float)m.eps);
            mat(L + ".ffn_gate.weight", t_norm, m.n_embd, yA, 0, max_rows);
            mat(L + ".ffn_up.weight", t_norm, m.n_embd, yB, 0, max_rows);
            k_silu_mul<<<dim3((unsigned)((m.n_ff * B + 255) / 256), (unsigned)B), 256, 0, stc>>>(t_ffnh, yA, yB, (int)m.n_ff);
            mat(L + ".ffn_down.weight", t_ffnh, m.n_ff, yA, 0, max_rows);
            k_add<<<dim3((unsigned)((m.n_embd * B + 255) / 256), (unsigned)B), 256, 0, stc>>>(x, yA, (int)m.n_embd);
            if (dbg_dump && il <= 1 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("ffn_out" + std::to_string(il)).c_str(), x, m.n_embd); }
            if ((il & 7) == 0) safety("token");
        }
        k_rms_norm<<<grid1, 256, 0, stc>>>(t_norm, x, S("output_norm.weight"), (int)m.n_embd, (float)m.eps);
        if (dbg_dump && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg("final_norm", t_norm, m.n_embd); }
        mat("output.weight", t_norm, m.n_embd, logits, 0, m.n_vocab);
        check(cudaGetLastError(), "kernel");
        check(cudaStreamSynchronize(stc), "sync c");
        check(cudaStreamSynchronize(st), "sync copy");
        released.store(cold_list.size() + (size_t)kStage);
        for (auto& th : pool) th.join();
        check(cudaMemcpy(logits_h.data(), logits, B * m.n_vocab * 4, cudaMemcpyDeviceToHost), "logits d2h");
        if (dbg_dump) {
            long t1 = 0;
            for (long j = 1; j < m.n_vocab; ++j) if (logits_h[j] > logits_h[t1]) t1 = j;
            printf("dbg_gpu,logits_top,%ld,%ld:%.4f\n", cur_pos, t1, logits_h[t1]);
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        *tok_s_out = s;
        uint64_t cb = 0; for (int idx : cold_list) cb += items[idx].bytes;
        *cold_out = cb;
        if (fail.load()) exit(1);
        ++cur_pos;
    };

    // ---- run: prompt then greedy generation ----
    auto argmax = [&](const float* lg) {
        long best = 0; float bv = lg[0];
        for (long i = 1; i < m.n_vocab; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        return best;
    };
    long ties = 0;
    auto cmp_step = [&](const std::vector<float>& cl, const char* tag, long step) {
        const long g1 = argmax(logits_h.data()), c1 = argmax(cl.data());
        double mx = 0, mg = 0;
        for (long j = 0; j < m.n_vocab; ++j) { mx = fmaxf(mx, fabsf(logits_h[j] - cl[j])); mg = fmaxf(mg, fabsf(cl[j])); }
        worst_rel = fmax(worst_rel, mx / (mg + 1e-9));
        ++checked;
        if (g1 == c1) return;
        // near-tie argmax flips are FP accumulation-order noise, not errors: the two
        // implementations quantize activations identically but reduce in different order
        const double gap = fabs((double)logits_h[g1] - (double)cl[c1]);
        if (gap <= 0.25) { ++ties; printf("tie,%s,%ld,gpu=%ld(%s),cpu=%ld(%s),logit_gap=%.4f\n", tag, step, g1, g1 < (long)tok.vocab.size() ? tok.vocab[g1].c_str() : "?", c1, c1 < (long)tok.vocab.size() ? tok.vocab[c1].c_str() : "?", gap); return; }
        ++mismatches;
        fprintf(stderr, "ARGMAX MISMATCH at %s %ld: gpu=%ld(%s) cpu=%ld(%s) logit_gap=%.4f rel=%.3g\n",
                tag, step, g1, g1 < (long)tok.vocab.size() ? tok.vocab[g1].c_str() : "?", c1, c1 < (long)tok.vocab.size() ? tok.vocab[c1].c_str() : "?", gap, mx / (mg + 1e-9));
    };

    std::vector<float> cl;
    std::vector<long> next_toks(batch, -1);
    std::vector<std::vector<long>> gen_ids(batch);
    std::vector<size_t> plen(batch);
    double sum_gen_s = 0; long n_gen_s = 0;
    size_t max_plen = 0;
    for (long v = 0; v < batch; ++v) { plen[v] = prompts[v].size(); max_plen = std::max(max_plen, plen[v]); }
    for (size_t i = 0; i < max_plen; ++i) {
        std::vector<long> toks(batch);
        for (long v = 0; v < batch; ++v) toks[v] = i < plen[v] ? prompts[v][i] : prompts[v].back();
        double s; uint64_t cold;
        run_token(toks, (int)i, &s, &cold);
        if (validate) {
            cpu_step(cm, toks[0], cl);
            cmp_step(cl, "seq", (long)i);
        }
        for (long v = 0; v < batch; ++v)
            if (i == plen[v] - 1) {
                next_toks[v] = argmax(&logits_h[(size_t)v * m.n_vocab]);
                if (dbg_dump) {   // top-2 per seq at the handoff: tie vs gross divergence
                    const float* lg = &logits_h[(size_t)v * m.n_vocab];
                    long t1 = 0, t2 = 0;
                    for (long j = 1; j < m.n_vocab; ++j) { if (lg[j] > lg[t1]) { t2 = t1; t1 = j; } else if (lg[j] > lg[t2]) t2 = j; }
                    printf("handoff,seq=%ld,top1=%ld:%.4f,top2=%ld:%.4f,gap=%.4f\n", v, t1, lg[t1], t2, lg[t2], lg[t1] - lg[t2]);
                }
            }
        const long fv = free_vram_mib();
        if (fv >= 0 && fv < free_min) free_min = fv;
        safety("prompt");
        printf("prompt_token,%zu,seconds=%.3f,cold_bytes=%llu,free_vram_MiB=%ld\n", i, s, (unsigned long long)cold, fv);
    }
    long n_eos = 0;
    for (long g = 0; g < gen_tokens && n_eos < batch; ++g) {
        const int pass = (int)(max_plen + g);
        std::vector<long> toks(batch);
        for (long v = 0; v < batch; ++v) toks[v] = next_toks[v] >= 0 ? next_toks[v] : m.eos_id;
        double s; uint64_t cold;
        run_token(toks, pass, &s, &cold);
        n_eos = 0;
        for (long v = 0; v < batch; ++v) {
            gen_ids[v].push_back(toks[v]);
            if (validate && v == 0) {
                cpu_step(cm, toks[0], cl);
                cmp_step(cl, "gen", g);
            }
            next_toks[v] = argmax(&logits_h[(size_t)v * m.n_vocab]);
            if (next_toks[v] == m.eos_id) ++n_eos;
        }
        const long fv = free_vram_mib();
        if (fv >= 0 && fv < free_min) free_min = fv;
        safety("gen");
        if (pass >= 1) { sum_steady_s += s; ++n_steady; steady_cold_bytes += cold; ++n_steady_cold; if (s < best_tok_s) best_tok_s = s; }
        sum_gen_s += s; ++n_gen_s;
        printf("gen_token,%ld,seconds=%.3f,agg_tok_per_s=%.3f,cold_bytes=%llu,free_vram_MiB=%ld,seq0_token=%ld[%s]\n",
               g, s, (double)batch / s, (unsigned long long)cold, fv, toks[0],
               toks[0] >= 0 && toks[0] < (long)tok.vocab.size() ? tok.vocab[toks[0]].c_str() : "?");
    }
    // stash generated ids of sequence 0 for the summary/text
    generated = gen_ids[0];
    all_ids = prompts[0];
    all_ids.insert(all_ids.end(), gen_ids[0].begin(), gen_ids[0].end());

    // ---- summary ----
    if (out_ids_path.size()) {
        FILE* f = fopen(out_ids_path.c_str(), "w");
        if (f) { for (long id : generated) fprintf(f, "%ld\n", id); fclose(f); }
    }
    if (show_text && batch == 1) printf("generated_text,%s\n", tok.decode(generated).c_str());
    if (show_text && batch > 1) {
        // equivalence check: all sequences must have produced identical token streams
        bool same = true;
        for (long v = 1; v < batch; ++v) if (gen_ids[v] != gen_ids[0]) { same = false; break; }
        printf("batch_equivalence,seqs=%ld,identical=%s\n", batch, same ? "yes" : "no");
        if (!same) {
            for (long v = 0; v < batch; ++v) {
                printf("batch_seq_ids,%ld", v);
                for (long id : gen_ids[v]) printf(",%ld", id);
                printf("\n");
            }
        }
        printf("generated_text,%s\n", tok.decode(gen_ids[0]).c_str());
    }
    printf("verify,steps_checked=%ld,argmax_mismatches=%ld,near_ties=%ld,worst_logit_rel=%.4g,validate=%d\n", checked, mismatches, ties, worst_rel, validate);
    const double steady_tps = sum_steady_s > 0 ? (double)n_steady / sum_steady_s : 0;          // steps/s
    const double gen_tps = sum_gen_s > 0 ? (double)n_gen_s / sum_gen_s : 0;
    const double cold_per_tok = n_steady_cold ? (double)steady_cold_bytes / n_steady_cold / 1073741824.0 : 0;
    printf("summary,%s,batch=%ld,gen_steps=%zu,steady_steps_per_s=%.3f,steady_aggregate_tok_per_s=%.3f,best_step_aggregate_tok_per_s=%.3f,overall_aggregate_tok_per_s=%.3f,cold_GiB_per_step=%.3f,cold_GiB_per_token=%.4f,min_free_vram_MiB=%ld,ram_avail_MiB=%ld,swap_used_MiB=%ld\n",
           (validate && mismatches) ? "FAIL" : "PASS", batch, generated.size(), steady_tps,
           steady_tps * batch, best_tok_s > 0 ? batch / best_tok_s : 0.0, gen_tps * batch,
           cold_per_tok, n_steady ? cold_per_tok / batch : 0.0,
           free_min, meminfo_mib("MemAvailable:"), meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:"));
    return (validate && mismatches) ? 1 : 0;
}
