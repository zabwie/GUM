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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <immintrin.h>
#include <atomic>
#include <vector>

static constexpr size_t kMaxItem = 16u << 20;
static constexpr size_t kStageBytes = kMaxItem + 8192;
// E186 GUM-Fuse wavefront sweep knobs. In-flight bytes = kSlots * 16 MiB; the
// release-window back-pressure (see pending handling) must stay < kStage because
// the copy-event sync indexes o % kStage. E186 sweep (repeated twice, token-exact):
// (8,8,6)=6.98 tok/s, (16,16,12)=7.17 (+3.0%), (24,24,18)=6.66 — deeper queues
// DEGRADE past 16; do not raise further without a new hypothesis.
#ifndef GUM_KSTAGE
#define GUM_KSTAGE 16
#endif
#ifndef GUM_KSLOTS
#define GUM_KSLOTS 16
#endif
#ifndef GUM_WINDOW
#define GUM_WINDOW 12
#endif
static constexpr int kStage = GUM_KSTAGE;
static constexpr int kSlots = GUM_KSLOTS;   // E047: deeper slot ring
static constexpr int kWindow = GUM_WINDOW;
static constexpr uint64_t kMapMargin = 64ull << 20;
static constexpr int kGGML_F32 = 0, kGGML_F16 = 1, kGGML_Q4_0 = 2, kGGML_Q8_0 = 8;

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
        auto rd_u32 = [&]() { uint32_t v = 0; if (fread(&v, 4, 1, f) != 1) return 0L; return (long)v; };
        auto rd_i64 = [&]() { int64_t v = 0; if (fread(&v, 8, 1, f) != 1) return 0L; return (long)v; };
        auto rd_f32 = [&]() { float v = 0; if (fread(&v, 4, 1, f) != 1) return 0.0; return (double)v; };
        auto rd_u64 = [&]() { uint64_t v = 0; if (fread(&v, 8, 1, f) != 1) return 0L; return (long)v; };
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
        long n = 1;
        for (int d = 0; d < t.nd; ++d) n *= t.ne[d];
        switch (t.type) {
            case kGGML_F32: t.bytes = (uint64_t)n * 4; break;
            case kGGML_F16: t.bytes = (uint64_t)n * 2; break;
            case kGGML_Q4_0: t.bytes = (uint64_t)((n + 31) / 32) * 18; break;
            case kGGML_Q8_0: t.bytes = (uint64_t)((n + 31) / 32) * 34; break;
            default: t.bytes = 0; break;
        }
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

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------
// quantize NB activation vectors: x [NB][K], xq [NB][K], xmeta [NB][K/32]
// xmeta packs the quant scale bits and sum correction into one aligned 64-bit load.

// ---- E193: CPU row-split compute (GUM-Fuse second stream) --------------------
// Reads MMA-packed Q4_0 tiles straight from the pinned cache (pack_q4_tiles layout:
// per 16-row tile and block b, tile[r*16..+16) = the row's nibble bytes (q in 0..15)
// and tile[256+r*2..+2) = the row's fp16 scale). Activations use the engine scheme:
// q_x (s8) + per-block meta (d_x, s_x). One output row contribution per block:
//   d_w * d_x * (sum(q_w*q_x) - 8*s_x)
// dpbusd: unsigned slot carries q_w (0..15), signed slot carries q_x.
static double e193_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#define E193LOG(...) do { if (getenv("GUM_E193_DEBUG")) fprintf(stderr, "e193 %.6f ", e193_now()); if (getenv("GUM_E193_DEBUG")) fprintf(stderr, __VA_ARGS__); } while(0)
static inline float cpu_fp16_to_f32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}
static void cpu_q4_rows(const uint8_t* wpacked, int K, int rows, int B,
                        const int8_t* xq, const int2* xmeta, float* out, long stride) {
    const int blocks = K / 32;
    for (int t = 0; t < rows / 16; ++t) {
        const uint8_t* tile0 = wpacked + (size_t)t * blocks * 288;
        const long row0 = (long)t * 16;
        __m256 vacc[64][2];
        for (int v = 0; v < B; ++v) { vacc[v][0] = _mm256_setzero_ps(); vacc[v][1] = _mm256_setzero_ps(); }
        int32_t sv[64][16];
        for (int b = 0; b < blocks; ++b) {
            const uint8_t* tb = tile0 + (size_t)b * 288;
            const uint16_t* scales = (const uint16_t*)(tb + 256);
            for (int r = 0; r < 16; ++r) {
                const uint8_t* nib = tb + r * 16;
                uint8_t wu[32];
                // GGUF Q4_0 order (matches k_mat_q4_shared): weights 0-15 = LOW nibbles of bytes 0-15, weights 16-31 = HIGH nibbles
                for (int j = 0; j < 16; ++j) { wu[j] = nib[j] & 0xF; wu[16 + j] = nib[j] >> 4; }
                const __m128i wl = _mm_loadu_si128((const __m128i*)wu);
                const __m128i wh = _mm_loadu_si128((const __m128i*)(wu + 16));
                const int8_t* ax = xq + (size_t)b * 32;
                for (int v = 0; v < B; ++v) {
                    const __m128i a0 = _mm_loadu_si128((const __m128i*)(ax + (size_t)v * K));
                    const __m128i a1 = _mm_loadu_si128((const __m128i*)(ax + (size_t)v * K + 16));
                    __m128i acc = _mm_add_epi32(_mm_dpbusd_epi32(_mm_setzero_si128(), wl, a0),
                                                _mm_dpbusd_epi32(_mm_setzero_si128(), wh, a1));
                    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2)));
                    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1)));
                    sv[v][r] = _mm_cvtsi128_si32(acc);
                }
            }
            for (int v = 0; v < B; ++v) {
                const int2 mx = xmeta[(size_t)v * blocks + b];
                const float dx = *(const float*)&mx.x;
                const __m256 corr = _mm256_set1_ps(-8.f * (float)mx.y);
                const __m256 dxv = _mm256_set1_ps(dx);
                __m256 dw0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)scales));
                __m256 dw1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(scales + 8)));
                __m256 s0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)&sv[v][0]));
                __m256 s1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)&sv[v][8]));
                vacc[v][0] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_add_ps(s0, corr), dw0), dxv, vacc[v][0]);
                vacc[v][1] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_add_ps(s1, corr), dw1), dxv, vacc[v][1]);
            }
        }
        for (int v = 0; v < B; ++v) {
            _mm256_storeu_ps(out + (size_t)v * stride + row0, vacc[v][0]);
            _mm256_storeu_ps(out + (size_t)v * stride + row0 + 8, vacc[v][1]);
        }
    }
}

__global__ void k_quant_x(const float* x, long nb, int8_t* xq, int2* xmeta) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nb) return;
    float amax = 0.f; for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
    float d = amax / 127.f; float id = d > 0.f ? 1.f / d : 0.f; int s = 0;
    for (int i = 0; i < 32; ++i) { int q = __float2int_rn(x[b * 32 + i] * id); xq[b * 32 + i] = (int8_t)q; s += q; }
    xmeta[b] = make_int2(__float_as_int(d), s);
}

__device__ __forceinline__ void store_quantized_warp(float value, long block, int8_t* xq, int2* meta) {
    constexpr unsigned all = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    float peak = fabsf(value);
    for (int stride = 16; stride; stride /= 2) peak = fmaxf(peak, __shfl_xor_sync(all, peak, stride));
    const float scale = peak / 127.f;
    const float inverse = scale > 0.f ? 1.f / scale : 0.f;
    const int q = __float2int_rn(value * inverse);
    xq[block * 32 + lane] = (int8_t)q;
    int sum = q;
    for (int stride = 16; stride; stride /= 2) sum += __shfl_down_sync(all, sum, stride);
    if (!lane) meta[block] = make_int2(__float_as_int(scale), sum);
}

__global__ void k_rms_quant(int8_t* xq, int2* meta, const float* x, const float* w, int n, float eps) {
    const long row = blockIdx.x;
    x += row * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    __shared__ float reduction[8];
    for (int stride = 16; stride; stride /= 2) ss += __shfl_down_sync(0xffffffffu, ss, stride);
    if (!(threadIdx.x & 31)) reduction[threadIdx.x / 32] = ss;
    __syncthreads();
    if (!threadIdx.x) {
        float sum = 0.f;
        for (int i = 0; i < 8; ++i) sum += reduction[i];
        reduction[0] = rsqrtf(sum / n + eps);
    }
    __syncthreads();
    const float norm = reduction[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        store_quantized_warp(x[i] * norm * w[i], row * (n / 32) + i / 32, xq, meta);
}

__global__ void k_silu_quant(int8_t* xq, int2* meta, const float* a, const float* b, long count) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = a[i];
    store_quantized_warp(value / (1.f + expf(-value)) * b[i], i / 32, xq, meta);
}

// y[v*stride + row] = dot(W_row, xq_v) — NB decode vectors share one weight read;
// identical math to streamed_decode gemv_q4 (dp4a, E033/E034)
template <int NB>
__global__ void k_mat_q4(const uint8_t* w, long K, long rows, const int8_t* xq, const int2* xmeta, float* y, long ystride, const int* row_ids = nullptr) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long nb = K / 32; const size_t rowbytes = (size_t)nb * 18;
    const uint8_t* r = w + (size_t)(row_ids ? row_ids[row] : row) * rowbytes;
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
            const int2 meta = xmeta[v * (K / 32) + b];
            int s = 0;
            s = __dp4a((int)lo0, xa.x, s); s = __dp4a((int)lo1, xa.y, s); s = __dp4a((int)lo2, xa.z, s); s = __dp4a((int)lo3, xa.w, s);
            s = __dp4a((int)hi0, xb.x, s); s = __dp4a((int)hi1, xb.y, s); s = __dp4a((int)hi2, xb.z, s); s = __dp4a((int)hi3, xb.w, s);
            acc[v] += d * __int_as_float(meta.x) * (float)(s - 8 * meta.y);
        }
    }
#pragma unroll
    for (int v = 0; v < NB; ++v) {
        float a = acc[v];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) y[v * ystride + row] = a;
    }
}

// NB=16, K>=5120 path. A CTA stages one 64-quanta tile of all activation vectors
// and their scale/correction metadata. Sixteen row warps reuse the tile before
// advancing; 8-byte shared banks make the 40-byte per-quanta record conflict-free.
// Each lane visits K blocks in the same ascending order as k_mat_q4<16>.
template<int NB>
__global__ void k_mat_q4_shared(const uint8_t* w, long K, long rows, const int8_t* xq,
                                      const int2* xmeta, float* y, long ystride) {
    constexpr int CB = NB <= 16 ? 64 : 32, THREADS = 512;
    extern __shared__ uint32_t tile[]; // [NB][CB][8 Q4 words + 2 metadata words]
    const int tid = threadIdx.x, lane = tid & 31;
    const long nb = K / 32;
    const size_t rowbytes = (size_t)nb * 18;
    const long row = (long)blockIdx.x * 16 + (tid >> 5);
    float acc[NB];
#pragma unroll
    for (int v = 0; v < NB; ++v) acc[v] = 0.f;
    int2* smeta = (int2*)tile;

    for (long c0 = 0; c0 < nb; c0 += CB) {
        const int cb = (int)min((long)CB, nb - c0);
        // Four aligned uint2 copies cover the eight activation words per block.
        for (int i = tid; i < NB * cb * 4; i += THREADS) {
            const int v = i / (cb * 4), rem = i % (cb * 4), b = rem >> 2, pair = rem & 3;
            const uint2 value = *((const uint2*)(xq + (size_t)v * K + (size_t)(c0 + b) * 32 + pair * 8));
            uint32_t* dst = tile + ((size_t)v * CB + b) * 10 + pair * 2;
            ((uint2*)dst)[0] = value;
        }
        for (int i = tid; i < NB * cb; i += THREADS) {
            const int v = i / cb, b = i % cb;
            smeta[((size_t)v * CB + b) * 5 + 4] = xmeta[(size_t)v * nb + c0 + b];
        }
        __syncthreads();

        if (row < rows) {
            const uint8_t* r = w + (size_t)row * rowbytes;
            for (long b = lane; b < cb; b += 32) {
                const uint16_t* blk = (const uint16_t*)(r + (size_t)(c0 + b) * 18);
                const float d = __half2float(__ushort_as_half(blk[0]));
                uint32_t q[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) q[j] = (uint32_t)blk[1 + 2 * j] | ((uint32_t)blk[2 + 2 * j] << 16);
                const uint32_t lo0 = q[0] & 0x0F0F0F0Fu, lo1 = q[1] & 0x0F0F0F0Fu;
                const uint32_t lo2 = q[2] & 0x0F0F0F0Fu, lo3 = q[3] & 0x0F0F0F0Fu;
                const uint32_t hi0 = (q[0] >> 4) & 0x0F0F0F0Fu, hi1 = (q[1] >> 4) & 0x0F0F0F0Fu;
                const uint32_t hi2 = (q[2] >> 4) & 0x0F0F0F0Fu, hi3 = (q[3] >> 4) & 0x0F0F0F0Fu;
#pragma unroll
                for (int v = 0; v < NB; ++v) {
                    const uint32_t* qb = tile + ((size_t)v * CB + b) * 10;
                    const uint2 a0 = ((const uint2*)qb)[0], a1 = ((const uint2*)qb)[1];
                    const uint2 a2 = ((const uint2*)qb)[2], a3 = ((const uint2*)qb)[3];
                    const int2 meta = smeta[((size_t)v * CB + b) * 5 + 4];
                    int s = 0;
                    s = __dp4a((int)lo0, (int)a0.x, s); s = __dp4a((int)lo1, (int)a0.y, s);
                    s = __dp4a((int)lo2, (int)a1.x, s); s = __dp4a((int)lo3, (int)a1.y, s);
                    s = __dp4a((int)hi0, (int)a2.x, s); s = __dp4a((int)hi1, (int)a2.y, s);
                    s = __dp4a((int)hi2, (int)a3.x, s); s = __dp4a((int)hi3, (int)a3.y, s);
                    acc[v] += d * __int_as_float(meta.x) * (float)(s - 8 * meta.y);
                }
            }
        }
        __syncthreads();
    }

    if (row < rows) {
#pragma unroll
        for (int v = 0; v < NB; ++v) {
            float a = acc[v];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
            if (lane == 0) y[(size_t)v * ystride + row] = a;
        }
    }
}
// rms norm over NB vectors: x/out [NB][n], grid (1, NB)
__global__ void k_rms_norm(float* out, const float* x, const float* w, int n, float eps, int out_stride = 0) {
    const long v = blockIdx.y;
    x += v * n; out += v * (out_stride ? out_stride : n);
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
__global__ void k_rope(float* q, float* k, int pos, int nh_q, int nh_k, int hd, int n_rot, float base, int q_stride, int style, int row_pos, const int* tree_depth = nullptr) {
    const int h = blockIdx.x;
    const long v = blockIdx.y;
    q += v * (size_t)nh_q * q_stride;
    k += v * (size_t)nh_k * hd;
    const int half = n_rot / 2;
    float* dst = h < nh_q ? q + (size_t)h * q_stride : k + (size_t)(h - nh_q) * hd;
    // multi-position verify: row v sits at position pos+v; otherwise all rows share pos
    const int p = pos + (tree_depth ? tree_depth[v] : (row_pos ? (int)v : 0));
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float theta = (float)p * powf(base, -2.f * (float)i / (float)n_rot);
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
                              int pos, int n_ctx_max, int nh_q, int n_kh, int hd, float scale, int q_stride, long kv_seq_stride, int row_pos,
                              const int* tree_depth = nullptr, const int* tree_path = nullptr, int path_stride = 0,
                              int first_pos = 0) {
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
    // multi-position verify: row v is at position pos+v and attends cache[0..pos+v]
    const int np = pos + 1 + (tree_depth ? tree_depth[v] : (row_pos ? (int)v : 0)) - first_pos;
    const int* path = tree_path ? tree_path + v * path_stride : nullptr;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int t = warp; t < np; t += blockDim.x / 32) {
        const int at = first_pos + t;
        const int slot = path && at >= pos ? pos + path[at - pos] : at;
        const float* key = Kk + (size_t)(slot < n_ctx_max ? slot : slot % n_ctx_max) * hd;
        float score = 0.f;
        // Each warp scores a different key while retaining the previous order
        // of the 32-element reductions and the sum across groups.
        for (int g = 0; g < (int)blockDim.x / 32; ++g) {
            float p = 0.f;
            for (int i = g * 32 + lane; i < hd; i += blockDim.x) p += qh[i] * key[i];
            for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
            score += p;
        }
        if (lane == 0) sh[t] = score * scale;
    }
    __syncthreads();
    __shared__ float mx, zsum;
    if (threadIdx.x == 0) {
        float m = sh[0]; for (int t = 1; t < np; ++t) m = fmaxf(m, sh[t]);
        mx = m;
    }
    __syncthreads();
    for (int t = threadIdx.x; t < np; t += blockDim.x) sh[t] = expf(sh[t] - mx);
    __syncthreads();
    if (threadIdx.x == 0) {
        float z = 0.f; for (int t = 0; t < np; ++t) z += sh[t];
        zsum = z;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < np; ++t) {
            const int at = first_pos + t;
            const int slot = path && at >= pos ? pos + path[at - pos] : at;
            acc += sh[t] * Vk[(size_t)(slot < n_ctx_max ? slot : slot % n_ctx_max) * hd + i];
        }
        out[(size_t)h * hd + i] = acc / zsum;
    }
}

__global__ void k_append_kv(float* keys, float* values, const float* k, const float* v,
                            int pos, int ctx, int heads, int head_dim, long sequence_stride, bool positions) {
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * head_dim) return;
    const long from = (long)row * heads * head_dim + i;
    const int slot = pos + (positions ? row : 0);
    const long to = (long)row * sequence_stride +
                    ((long)(i / head_dim) * ctx + (slot < ctx ? slot : slot % ctx)) * head_dim + i % head_dim;
    keys[to] = k[from];
    values[to] = v[from];
}
#include "tiered_kv.cuh"
// gated delta net, one token: state[h] is [i][j], so adjacent threads j load adjacent state.
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
    float* col = state + (size_t)h * S * S + j;
    const float* kh = k16 + (size_t)hk * S;
    const float* qh = q16 + (size_t)hk * S;
    const float* vh = vv + (size_t)h * S;
    const float eg = expf(g[h]);
    for (int i = 0; i < S; ++i) col[(size_t)i * S] *= eg;
    float dot = 0.f; for (int i = 0; i < S; ++i) dot += col[(size_t)i * S] * kh[i];
    const float dj = (vh[j] - dot) * beta[h];
    for (int i = 0; i < S; ++i) col[(size_t)i * S] += dj * kh[i];
    float o = 0.f; for (int i = 0; i < S; ++i) o += col[(size_t)i * S] * qh[i];
    out[(size_t)h * S + j] = o * scale;
}
// multi-position verify: ONE sequence's conv over B positions, sequential state shifts.
// x/out: [B][C] rows are positions; grid (C/256).
__global__ void k_conv1d_silu_multi(float* cstate, const float* x, const float* w, float* out, int C, int B,
                                    bool update_state = true, float* trial_state = nullptr, int prompt_rows = 0,
                                    bool save_boundary = false) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = cstate[c * 3 + 0], s1 = cstate[c * 3 + 1], s2 = cstate[c * 3 + 2];
    for (int p = 0; p < B; ++p) {
        const float xc = x[(size_t)p * C + c];
        const float sum = w[c * 4 + 0] * s0 + w[c * 4 + 1] * s1 + w[c * 4 + 2] * s2 + w[c * 4 + 3] * xc;
        s0 = s1; s1 = s2; s2 = xc;
        out[(size_t)p * C + c] = sum / (1.f + expf(-sum));
        if (trial_state && p + 1 == prompt_rows) {
            float* dst = save_boundary ? trial_state : cstate;
            dst[c * 3 + 0] = s0; dst[c * 3 + 1] = s1; dst[c * 3 + 2] = s2;
        }
    }
    if (update_state) {
        float* dst = trial_state && !save_boundary ? trial_state : cstate;
        dst[c * 3 + 0] = s0; dst[c * 3 + 1] = s1; dst[c * 3 + 2] = s2;
    }
}
// Multi-position gated delta net for ONE sequence. State is [H][i][j], with lanes split
// over j so each serial i-loop touches coalesced state elements. One block covers 32 j
// values; the 2D grid exposes four blocks/head at S=128 without changing each j's FP order.
// q/k/v: [B][qkv_out] slices; g/beta: [B][H]; out: [B][H*S].
__global__ void k_delta_net_multi(float* state, const float* qkv, const float* g, const float* beta, float* out,
                                  int S, int H, int ngroup, float scale, long qkv_stride, long out_stride, int B) {
    const int h = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= S) return;
    const int hk = h % ngroup;   // ggml: iv1 % n_k_heads
    float* st = state + (size_t)h * S * S;
    for (int p = 0; p < B; ++p) {
        const float* qkvp = qkv + (size_t)p * qkv_stride;
        const float* kh = qkvp + (size_t)S * ngroup + (size_t)hk * S;
        const float* qh = qkvp + (size_t)hk * S;
        const float* vh = qkvp + 2 * (size_t)S * ngroup + (size_t)h * S;
        float* col = st + j;
        const float eg = expf(g[(size_t)p * H + h]);
        for (int i = 0; i < S; ++i) col[(size_t)i * S] *= eg;
        float dot = 0.f; for (int i = 0; i < S; ++i) dot += col[(size_t)i * S] * kh[i];
        const float dj = (vh[j] - dot) * beta[(size_t)p * H + h];
        for (int i = 0; i < S; ++i) col[(size_t)i * S] += dj * kh[i];
        if (out) {
            float o = 0.f; for (int i = 0; i < S; ++i) o += col[(size_t)i * S] * qh[i];
            out[(size_t)p * out_stride + (size_t)h * S + j] = o * scale;
        }
    }
}

__global__ void k_conv_commit(float* state, const float* before, const float* input, int channels, int accepted, const int* path = nullptr) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int i = 0; i < 3; ++i) {
        const int p = accepted + i - 3;
        state[3 * c + i] = p < 0 ? before[3 * c + p + 3] : input[(size_t)(path ? path[p] : p) * channels + c];
    }
}

template <bool output, bool tree = false>
__global__ void k_delta_net_128(float* state, const float* qkv, const float* g, const float* beta, float* out,
                                int H, int ngroup, float scale, long qkv_stride, long out_stride, int B,
                                const int* path = nullptr, const int* lengths = nullptr,
                                const int* owners = nullptr, int path_stride = 0, bool update_state = true,
                                float* trial_state = nullptr, int prompt_rows = 0, bool save_boundary = false) {
    constexpr int S = 128;
    const int leaf = blockIdx.z;
    if constexpr (tree) {
        path += leaf * path_stride;
        B = lengths[leaf];
    }
    const int h = blockIdx.x;
    const int j = blockIdx.y * 32 + threadIdx.x;
    const int hk = h % ngroup;
    float* col = state + (size_t)h * S * S + j;
    float local[S];
#pragma unroll
    for (int i = 0; i < S; ++i) local[i] = col[i * S];
    for (int p = 0; p < B; ++p) {
        const int row = path ? path[p] : p;
        const float* qkvp = qkv + (size_t)row * qkv_stride;
        const float* qh = qkvp + (size_t)hk * S;
        const float* kh = qh + S * ngroup;
        const float* vh = qkvp + 2 * S * ngroup + h * S;
        const float decay = expf(g[(size_t)row * H + h]);
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            local[i] = __fmul_rn(local[i], decay);
            dot = fmaf(local[i], kh[i], dot);
        }
        const float correction = (vh[j] - dot) * beta[(size_t)row * H + h];
#pragma unroll
        for (int i = 0; i < S; ++i) local[i] = fmaf(correction, kh[i], local[i]);
        if constexpr (output) {
          if (!tree || owners[row] == leaf) {
            float value = 0.f;
#pragma unroll
            for (int i = 0; i < S; ++i) value = fmaf(local[i], qh[i], value);
            out[(size_t)row * out_stride + h * S + j] = value * scale;
          }
        }
        if constexpr (!tree) {
            if (trial_state && p + 1 == prompt_rows) {
                float* dst = save_boundary ? trial_state + (size_t)h * S * S + j : col;
#pragma unroll
                for (int i = 0; i < S; ++i) dst[i * S] = local[i];
            }
        }
    }
    if constexpr (!tree) {
        if (update_state) {
            float* dst = trial_state && !save_boundary ? trial_state + (size_t)h * S * S + j : col;
#pragma unroll
            for (int i = 0; i < S; ++i) dst[i * S] = local[i];
        }
    }
}

static void launch_delta_multi(float* state, const float* qkv, const float* g, const float* beta, float* out,
                               int S, int H, int groups, float scale, long qkv_stride, long out_stride,
                               int rows, cudaStream_t stream, const int* path = nullptr, bool update_state = true,
                               float* trial_state = nullptr, int prompt_rows = 0, bool save_boundary = false) {
    const dim3 grid((unsigned)H, (unsigned)((S + 31) / 32));
    if (S != 128) {
        k_delta_net_multi<<<grid, 32, 0, stream>>>(state, qkv, g, beta, out, S, H, groups, scale, qkv_stride, out_stride, rows);
    } else if (out) {
        k_delta_net_128<true><<<grid, 32, 0, stream>>>(state, qkv, g, beta, out, H, groups, scale,
            qkv_stride, out_stride, rows, nullptr, nullptr, nullptr, 0, update_state, trial_state, prompt_rows, save_boundary);
    } else {
        k_delta_net_128<false><<<grid, 32, 0, stream>>>(state, qkv, g, beta, out, H, groups, scale, qkv_stride, out_stride, rows, path);
    }
}
__global__ void k_conv_tree(const float* state, const float* x, const float* w, float* out,
                             int channels, const int* depth, const int* paths, int stride) {
    const int row = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int* path = paths + row * stride;
    float taps[4];
    for (int i = 0; i < 4; ++i) {
        const int d = depth[row] + i - 3;
        taps[i] = d < 0 ? state[3 * c + d + 3] : x[(size_t)path[d] * channels + c];
    }
    const float sum = w[c * 4] * taps[0] + w[c * 4 + 1] * taps[1] +
                      w[c * 4 + 2] * taps[2] + w[c * 4 + 3] * taps[3];
    out[(size_t)row * channels + c] = sum / (1.f + expf(-sum));
}

__global__ void k_compact_kv(float* kv, int ctx, int head_dim, int columns,
                             int pos, int count, const int* path) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= columns) return;
    float* column = kv + (size_t)(i / head_dim) * ctx * head_dim + i % head_dim;
    // Row indices increase along a path, so compaction cannot overwrite a
    // source row that a later iteration needs.
    for (int d = 0; d < count; ++d) {
        if (path[d] != d) column[(size_t)(pos + d) * head_dim] = column[(size_t)(pos + path[d]) * head_dim];
    }
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
__global__ void k_gate_ssm(float* g, const float* alpha, const float* dtb, const float* a, int n, int rows) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * n) return;
    const float v = alpha[i] + dtb[i % n];
    const float sp = v > 20.f ? v : log1pf(expf(v));
    g[i] = sp * a[i % n];
}
__global__ void k_silu_mul(float* y, const float* a, const float* b, int n) {  // y = silu(a)*b
    const long i = ((long)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; if (i >= (long)gridDim.y * n) return;
    const float v = a[i];
    y[i] = v / (1.f + expf(-v)) * b[i];
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
                    float* col = row + j;
                    for (int i = 0; i < Sv; ++i) col[(size_t)i * Sv] *= eg;
                    float dot = 0.f; for (int i = 0; i < Sv; ++i) dot += col[(size_t)i * Sv] * kh[i];
                    const float dj = (vh[j] - dot) * beta[h];
                    for (int i = 0; i < Sv; ++i) col[(size_t)i * Sv] += dj * kh[i];
                    float o = 0.f; for (int i = 0; i < Sv; ++i) o += col[(size_t)i * Sv] * qh[i];
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
static bool g_packed_widths = false;
static constexpr long kPrefillChunk = 32;
static long mat_width(long rows) {
    if (g_packed_widths && rows > 8 && rows <= 12) return 12;
    if (g_packed_widths && rows > 16 && rows <= 24) return 24;
    long width = 1;
    while (width < rows) width *= 2;
    return width;
}
static void launch_mat_q4(const uint8_t* w, long K, long rows, const int8_t* xq, const int2* xmeta,
                          float* y, long ystride, cudaStream_t s) {
    const unsigned g = (unsigned)((rows + 3) / 4);
    switch (g_batch) {
        case 1: k_mat_q4<1><<<g, 128, 0, s>>>(w, K, rows, xq, xmeta, y, ystride); break;
        case 2: k_mat_q4<2><<<g, 128, 0, s>>>(w, K, rows, xq, xmeta, y, ystride); break;
        case 4: k_mat_q4<4><<<g, 128, 0, s>>>(w, K, rows, xq, xmeta, y, ystride); break;
        case 8: k_mat_q4<8><<<g, 128, 0, s>>>(w, K, rows, xq, xmeta, y, ystride); break;
        case 12:
            if (K >= 5120) {
                constexpr size_t smem = 12 * 64 * 40;
                k_mat_q4_shared<12><<<(unsigned)((rows + 15) / 16), 512, smem, s>>>(w, K, rows, xq, xmeta, y, ystride);
            } else k_mat_q4<12><<<g, 128, 0, s>>>(w, K, rows, xq, xmeta, y, ystride);
            break;
        case 16:
            if (K >= 5120) {
                const size_t smem = 16 * 64 * 40;
                k_mat_q4_shared<16><<<(unsigned)((rows + 15) / 16), 512, smem, s>>>(w, K, rows, xq, xmeta, y, ystride);
            } else k_mat_q4<16><<<g, 128, 0, s>>>(w, K, rows, xq, xmeta, y, ystride);
            break;
        case 24:
            if (K >= 5120) {
                constexpr size_t smem = 24 * 32 * 40;
                k_mat_q4_shared<24><<<(unsigned)((rows + 15) / 16), 512, smem, s>>>(w, K, rows, xq, xmeta, y, ystride);
            } else k_mat_q4<24><<<g, 128, 0, s>>>(w, K, rows, xq, xmeta, y, ystride);
            break;
        case 32:
            if (K >= 5120) {
                const size_t smem = 32 * 32 * 40;
                k_mat_q4_shared<32><<<(unsigned)((rows + 15) / 16), 512, smem, s>>>(w, K, rows, xq, xmeta, y, ystride);
            } else k_mat_q4<32><<<g, 128, 0, s>>>(w, K, rows, xq, xmeta, y, ystride);
            break;
        case 64: k_mat_q4<64><<<g, 128, 0, s>>>(w, K, rows, xq, xmeta, y, ystride); break;
    }
}

#include "packed_scale_codec.cuh"

template <int NB, int WARPS, int GROUPS = 1, bool ENCODED = false>
__global__ void q4_mma(const uint8_t* w, long K, long rows, const int8_t* xq, const int2* xmeta,
                       float* y, long ystride, int runtime_groups = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ float partial[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int group = lane >> 2, thread = lane & 3;
    const unsigned groups = GROUPS ? GROUPS : runtime_groups;
    const long tile_index = blockIdx.x / groups;
    const long group_index = blockIdx.x % groups;
    const long first_row = tile_index * 16;
    const long nb = K / 32;
    xq += group_index * NB * K;
    xmeta += group_index * NB * nb;
    y += group_index * NB * ystride;
    float sum[32 / WARPS][NB / 8][4] = {};
    for (long start = 0; start < nb; start += 32) {
        const uint8_t* panel = w + ((size_t)tile_index * nb + start) * kRawTile;
        bool compact = false;
        if constexpr (ENCODED) {
            const uint32_t offset = ((const uint32_t*)w)[(tile_index * nb + start) / kPanelTiles];
            panel = w + (offset & ~1u);
            compact = offset & 1;
        }
#pragma unroll
        for (int strip = 0; strip < 32 / WARPS; ++strip) {
            const int k = warp + strip * WARPS;
            const uint8_t* tile = panel + k * (compact ? kTileQuants : kRawTile);
            const uint32_t* quants = (const uint32_t*)tile;
            const uint16_t* scales = (const uint16_t*)(tile + 256);
            const uint32_t u = quants[group * 4 + thread];
            const uint32_t v = quants[(group + 8) * 4 + thread];
            const float da = __half2float(__ushort_as_half(compact ? unpack_scale(panel, k, group) : scales[group]));
            const float db = __half2float(__ushort_as_half(compact ? unpack_scale(panel, k, group + 8) : scales[group + 8]));
            constexpr uint32_t mask = 0x0f0f0f0fu;
#pragma unroll
            for (int n = 0; n < NB / 8; ++n) {
                const uint32_t* xp = (const uint32_t*)(xq + (n * 8 + group) * K + (start + k) * 32);
                const uint32_t xb0 = xp[thread], xb1 = xp[thread + 4];
                int c0, c1, c2, c3;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {0,0,0,0};"
                    : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                    : "r"(u & mask),"r"(v & mask),"r"((u>>4)&mask),"r"((v>>4)&mask),"r"(xb0),"r"(xb1));
                const int col = n * 8 + 2 * thread;
                const int2 m0 = xmeta[col * nb + start + k], m1 = xmeta[(col + 1) * nb + start + k];
                sum[strip][n][0] += da * __int_as_float(m0.x) * (float)(c0 - 8 * m0.y);
                sum[strip][n][1] += da * __int_as_float(m1.x) * (float)(c1 - 8 * m1.y);
                sum[strip][n][2] += db * __int_as_float(m0.x) * (float)(c2 - 8 * m0.y);
                sum[strip][n][3] += db * __int_as_float(m1.x) * (float)(c3 - 8 * m1.y);
            }
        }
    }
#pragma unroll
    for (int strip = 0; strip < 32 / WARPS; ++strip) {
        const int k = warp + strip * WARPS;
#pragma unroll
        for (int n = 0; n < NB / 8; ++n) {
            const int col = n * 8 + 2 * thread;
            // Rotate each output's 32 partials so the lanes of a writing
            // warp access different shared-memory banks. Preserve reduction order.
            const int slot = (k + 8 * thread + group) & 31;
            partial[(col * 16 + group) * 32 + slot] = sum[strip][n][0];
            partial[((col + 1) * 16 + group) * 32 + slot] = sum[strip][n][1];
            partial[(col * 16 + group + 8) * 32 + slot] = sum[strip][n][2];
            partial[((col + 1) * 16 + group + 8) * 32 + slot] = sum[strip][n][3];
        }
    }
    __syncthreads();
    for (int o = warp; o < NB * 16; o += WARPS) {
        const int rotation = 8 * (o / 32) + (o % 8);
        float total = partial[o * 32 + ((lane + rotation) & 31)];
        for (int delta = 16; delta; delta >>= 1) total += __shfl_down_sync(0xffffffffu, total, delta);
        const long row = first_row + o % 16;
        if (lane == 0 && row < rows) y[(o / 16) * ystride + row] = total;
    }
#endif
}

template <bool ENCODED = false>
static void launch_packed_q4(const uint8_t* w, long K, long rows, const int8_t* xq, const int2* meta,
                             float* y, long ystride, cudaStream_t stream, long active = 0, bool grouped_prefill = false) {
    const unsigned tiles = (rows + 15) / 16;
    constexpr size_t smem8 = 16 * 8 * 32 * sizeof(float);
    constexpr size_t smem16 = 16 * 16 * 32 * sizeof(float);
    if (grouped_prefill) {
        switch ((active + 15) / 16) {
            case 3: q4_mma<16,8,3,ENCODED><<<3*tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride); break;
            case 4: q4_mma<16,8,4,ENCODED><<<4*tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride); break;
            case 5: q4_mma<16,8,5,ENCODED><<<5*tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride); break;
            case 6: q4_mma<16,8,6,ENCODED><<<6*tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride); break;
            case 7: q4_mma<16,8,7,ENCODED><<<7*tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride); break;
            case 8: q4_mma<16,8,8,ENCODED><<<8*tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride); break;
            default: {
                const int groups = (active + 15) / 16;
                q4_mma<16,8,0,ENCODED><<<groups * tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride,groups);
                break;
            }
        }
    } else if (g_batch <= 8) {
        q4_mma<8,8,1,ENCODED><<<tiles,256,smem8,stream>>>(w,K,rows,xq,meta,y,ystride);
    } else if (g_batch <= 16) {
        q4_mma<16,8,1,ENCODED><<<tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride);
    } else if (g_batch <= 24) {
        if (K >= 17408) q4_mma<8,8,3,ENCODED><<<3*tiles,256,smem8,stream>>>(w,K,rows,xq,meta,y,ystride);
        else q4_mma<8,16,3,ENCODED><<<3*tiles,512,smem8,stream>>>(w,K,rows,xq,meta,y,ystride);
    } else if (g_batch <= 32) {
        if (K >= 17408) q4_mma<16,8,2,ENCODED><<<2*tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride);
        else q4_mma<8,16,4,ENCODED><<<4*tiles,512,smem8,stream>>>(w,K,rows,xq,meta,y,ystride);
    } else {
        // Deep verification keeps the E167 dispatch.
        if (K >= 17408)
            q4_mma<16,8,4,ENCODED><<<4*tiles,256,smem16,stream>>>(w,K,rows,xq,meta,y,ystride);
        else q4_mma<8,8,8,ENCODED><<<8*tiles,256,smem8,stream>>>(w,K,rows,xq,meta,y,ystride);
    }
}

static void pack_q4_tiles(uint8_t* destination, const uint8_t* source, long K, long rows) {
    const long blocks = K / 32;
    for (long first = 0; first < rows; first += 16) {
        for (long b = 0; b < blocks; ++b) {
            uint8_t* tile = destination + ((size_t)(first / 16) * blocks + b) * 288;
            for (long r = 0; r < 16; ++r) {
                const uint8_t* from = source + ((size_t)(first + r) * blocks + b) * 18;
                memcpy(tile + r * 16, from + 2, 16);
                memcpy(tile + 256 + r * 2, from, 2);
            }
        }
    }
}

__global__ void k_pack_q4_tiles(uint16_t* destination, const uint16_t* source, int blocks, int tiles) {
    const int tile = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (tile >= tiles) return;
    const int lane = threadIdx.x & 31;
    const int row = lane & 15;
    const int half = lane >> 4;
    const size_t from = ((size_t)(tile / blocks * 16 + row) * blocks + tile % blocks) * 9;
    uint16_t* to = destination + (size_t)tile * 144;
    for (int word = 0; word < 4; ++word) to[row * 8 + half * 4 + word] = source[from + 1 + half * 4 + word];
    if (!half) to[128 + row] = source[from];
}

struct LogitMax {
    float score;
    int token;
    unsigned flags;
};

__device__ __forceinline__ LogitMax combine_max(LogitMax a, LogitMax b) {
    const unsigned flags = a.flags | b.flags;
    if (b.score > a.score || (b.score == a.score && b.token < a.token)) a = b;
    a.flags = flags;
    return a;
}

__device__ __forceinline__ LogitMax reduce_max(LogitMax result) {
    __shared__ LogitMax warp_max[8];
    constexpr unsigned all = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    for (int step = 16; step > 0; step /= 2) {
        const LogitMax other{__shfl_down_sync(all, result.score, step),
            __shfl_down_sync(all, result.token, step), __shfl_down_sync(all, result.flags, step)};
        if (lane + step < 32) result = combine_max(result, other);
    }
    if (!lane) warp_max[warp] = result;
    __syncthreads();
    if (!warp) {
        result = lane < 8 ? warp_max[lane] : LogitMax{-INFINITY, 0x7fffffff, 0};
        for (int step = 4; step > 0; step /= 2) {
            const LogitMax other{__shfl_down_sync(all, result.score, step),
                __shfl_down_sync(all, result.token, step), __shfl_down_sync(all, result.flags, step)};
            if (lane + step < 8) result = combine_max(result, other);
        }
    }
    return result;
}

__global__ void k_logit_tile_max(const float* logits, int vocab, LogitMax* maxima) {
    constexpr int tile = 1024;
    const int row = blockIdx.y;
    const int start = blockIdx.x * tile;
    const float* input = logits + (size_t)row * vocab;
    LogitMax result{-INFINITY, 0x7fffffff, 0};
    for (int i = threadIdx.x; i < tile && start + i < vocab; i += blockDim.x) {
        const float value = input[start + i];
        const LogitMax candidate = isfinite(value) ? LogitMax{value, start + i, value != 0.f ? 1U : 0U}
            : LogitMax{-INFINITY, 0x7fffffff, 2U};
        result = combine_max(result, candidate);
    }
    result = reduce_max(result);
    if (!threadIdx.x) maxima[(size_t)row * gridDim.x + blockIdx.x] = result;
}

__global__ void k_logit_row_max(const LogitMax* tiles, int count, LogitMax* output) {
    const int row = blockIdx.x;
    tiles += (size_t)row * count;
    LogitMax result{-INFINITY, 0x7fffffff, 0};
    for (int i = threadIdx.x; i < count; i += blockDim.x) result = combine_max(result, tiles[i]);
    result = reduce_max(result);
    if (!threadIdx.x) output[row] = result;
}

__global__ void k_mtp_shift_hidden(const float* previous, const float* target, const int* path,
                                   float* output, int width, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * count) return;
    const int row = index / width;
    const int column = index % width;
    if (row == 0) output[index] = previous[column];
    else {
        const int source = path ? path[row - 1] : row - 1;
        output[index] = target[(size_t)source * width + column];
    }
}

#include "eagle4_draft.cuh"
#include "parallel_draft.cuh"

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string model, prompt, prompt_ids_s, out_ids_path, expect_ids_path, mtp_path, eagle_path;
    long hot_mib = 8192, cache_mib = 5553, ctx = 1024, gen_tokens = 16, batch = 1, multi = 0, speculate = 0;
    double p_min = 0.25;
    long draft_max = 7, draft_vocab = 32768;
    bool balance_hot = true, explicit_speculate = false;
    bool explicit_p_min = false, explicit_draft_max = false, explicit_packed = false;
    int mtp_tree_mode = 1, mtp_group = kMaxDraftGroup, packed_verify = 0;
    int o_direct = 1, validate = 0, readers = 4, show_text = 1, dbg_dump = 0, verify_hot = 0, pld_enabled = 2;
    int l2_win_flag = 0;   // E190/A6: per-tensor persisting window — measured NEGATIVE, off by default
    float cpu_rows_frac = 0.f;   // E193: fraction of each tensor's rows computed on the CPU (0 = off)
    int cpu_verify = 0;          // E193: GPU also computes CPU rows; compare on host (debug)
    long chain_min = 0;   // E168: minimum direct-chain length for the deep override (0 = mtp_limit)
    std::string feed_ids_s, oracle_ids_path, parallel_path;
    bool resident_output = false, compress_packed = false, session = false, session_tail = false;
    bool session_draft_hints = false, session_stream = false;
    int fuse_prefill_tail = -1;
    bool measure_h2d = false;
    std::string kv_mode = "fp32";
    long kv_recent = kTierRecent;
    long prefill_chunks = 1, session_prefix_tokens = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i]; const char* v = argv[i + 1];
        if (k == "--model") model = v;
        else if (k == "--hot-mib") hot_mib = atol(v);
        else if (k == "--cache-mib") cache_mib = atol(v);
        else if (k == "--ctx") ctx = atol(v);
        else if (k == "--gen-tokens") gen_tokens = atol(v);
        else if (k == "--batch") batch = atol(v);
        else if (k == "--multi") multi = atol(v);
        else if (k == "--speculate") { speculate = atol(v); explicit_speculate = true; }
        else if (k == "--feed-ids") feed_ids_s = v;
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
        else if (k == "--expect-ids") expect_ids_path = v;
        else if (k == "--oracle-ids") oracle_ids_path = v;
        else if (k == "--parallel-draft") parallel_path = v;
        else if (k == "--resident-output") resident_output = atoi(v) != 0;
        else if (k == "--compress-packed") compress_packed = atoi(v) != 0;
        else if (k == "--session") session = atoi(v) != 0;
        else if (k == "--session-tail") session_tail = atoi(v) != 0;
        else if (k == "--session-draft-hints") session_draft_hints = atoi(v) != 0;
        else if (k == "--session-stream") session_stream = atoi(v) != 0;
        else if (k == "--fuse-prefill-tail") fuse_prefill_tail = atoi(v);
        else if (k == "--measure-h2d") measure_h2d = atoi(v) != 0;
        else if (k == "--l2-win") l2_win_flag = atoi(v) != 0;
        else if (k == "--cpu-rows") cpu_rows_frac = (float)atof(v);
        else if (k == "--cpu-verify") cpu_verify = atoi(v) != 0;
        else if (k == "--kv-cache") kv_mode = v;
        else if (k == "--kv-recent") kv_recent = atol(v);
        else if (k == "--prefill-chunks") prefill_chunks = atol(v);
        else if (k == "--session-prefix-tokens") session_prefix_tokens = atol(v);
        else if (k == "--mtp") mtp_path = v;
        else if (k == "--mtp-tree") mtp_tree_mode = atoi(v);
        else if (k == "--mtp-group") mtp_group = atoi(v);
        else if (k == "--packed-verify") { packed_verify = atoi(v); explicit_packed = true; }
        else if (k == "--eagle4") eagle_path = v;
        else if (k == "--p-min") { p_min = atof(v); explicit_p_min = true; }
        else if (k == "--draft-max") { draft_max = atol(v); explicit_draft_max = true; }
        else if (k == "--draft-vocab") draft_vocab = atol(v);
        else if (k == "--balance-hot") balance_hot = atoi(v) != 0;
        else if (k == "--pld") pld_enabled = atoi(v);
        else if (k == "--chain-min") chain_min = atol(v);
    }
    const bool auto_packed = !explicit_packed && !explicit_speculate && !mtp_path.empty() && mtp_tree_mode == 1 &&
        batch == 1 && !multi && !validate && getenv("GUM_PACKED_DEFAULT");
    if (packed_verify == 1 || auto_packed) {
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0), "packed verification device");
        cudaFuncAttributes kernel{};
        if (device.major >= 8) check(cudaFuncGetAttributes(&kernel, q4_mma<8, 8>), "packed verification kernel");
        const bool available = device.major >= 8 && kernel.binaryVersion >= 80;
        if (explicit_packed && !available) {
            fprintf(stderr, "--packed-verify requires an Ampere or newer build and GPU\n");
            return 2;
        }
        packed_verify = available;
    }
    if (!eagle_path.empty() || (!mtp_path.empty() && mtp_tree_mode == 1)) {
        if (!explicit_p_min) p_min = 0.06;
        if (!explicit_draft_max) draft_max = packed_verify ? 23 : 15;
    }
    if (gen_tokens < 1 || ctx < 1 || hot_mib < 0 || cache_mib < 0 || readers < 1 || draft_max < 0 || draft_vocab < 0 ||
        pld_enabled < 0 || pld_enabled > 2 || mtp_tree_mode < 0 || mtp_tree_mode > 1 || mtp_group < 1 || mtp_group > kMaxDraftGroup || packed_verify < 0 || packed_verify > 1 || !(p_min >= 0.0 && p_min <= 1.0)) {
        fprintf(stderr, "invalid context, memory, or drafting setting\n"); return 2;
    }
    if (model.empty()) { fprintf(stderr, "need --model\n"); return 2; }
    if (batch < 1 || batch > 64 || (batch & (batch - 1))) { fprintf(stderr, "--batch must be a power of two in 1..64\n"); return 2; }
    if (validate && batch != 1) { fprintf(stderr, "--validate requires --batch 1\n"); return 2; }
    if (!explicit_speculate && (!mtp_path.empty() || !eagle_path.empty()) && batch == 1 && !multi && !validate) speculate = packed_verify ? 64 : 16;
    if ((multi > 0 || speculate > 0) && (batch != 1 || (multi > 0 && speculate > 0))) {
        fprintf(stderr, "use one sequence and one verification mode\n");
        return 2;
    }
    if (multi > 0 || speculate > 0) {
        // E040 verify / E041 speculative mode: one sequence, B rows share one cold-set
        // transfer; batch buffers are reused as position rows
        if (validate) { fprintf(stderr, "--validate requires multi/speculate mode off\n"); return 2; }
        batch = multi > 0 ? multi : speculate;
        const bool packed_width = speculate == 24 && packed_verify;
        if (batch < 1 || batch > 64 || ((batch & (batch - 1)) && !packed_width)) {
            fprintf(stderr, "unsupported verification width\n"); return 2;
        }
        // note: multi stays 0 — the E040 verify branch checks multi>0; spec mode uses its own loop
    }
    if (!eagle_path.empty() && (!mtp_path.empty() || !speculate || multi || validate || ctx > 4096)) {
        fprintf(stderr, "--eagle4 requires single-user speculative mode, no --mtp, and context at most 4096\n");
        return 2;
    }
    const char* capture_layers = getenv("GUM_MID_CAPTURE");
    const char* capture_path = getenv("GUM_MID_CAPTURE_FILE");
    if (capture_layers && (batch <= 1 || multi || speculate || validate || !capture_path || !*capture_path)) {
        fprintf(stderr, "GUM_MID_CAPTURE requires dense batch decode and GUM_MID_CAPTURE_FILE\n");
        return 2;
    }
    const bool mtp_tree = !mtp_path.empty() && speculate > 0 && mtp_tree_mode == 1;
    const bool parallel_on = !parallel_path.empty();
    if (parallel_on) resident_output = true;
    if ((resident_output && (!speculate || !mtp_path.empty() || !eagle_path.empty())) ||
        (parallel_on && !oracle_ids_path.empty())) {
        fprintf(stderr, "resident output requires speculative mode with no MTP or EAGLE4; choose one draft source\n");
        return 2;
    }
    struct stat parallel_stat{};
    if (parallel_on && (stat(parallel_path.c_str(), &parallel_stat) != 0 ||
                        parallel_stat.st_size <= 0 || parallel_stat.st_size > (128ull << 20))) {
        fprintf(stderr, "invalid parallel draft file\n");
        return 2;
    }
    std::vector<long> oracle_ids;
    if (!oracle_ids_path.empty()) {
        if (speculate < 2 || !explicit_draft_max || draft_max < 1 || draft_max >= speculate ||
            expect_ids_path.empty() || !eagle_path.empty()) {
            fprintf(stderr, "--oracle-ids requires --expect-ids and --draft-max in 1..speculate-1, with no EAGLE4\n");
            return 2;
        }
        FILE* f = fopen(oracle_ids_path.c_str(), "r");
        if (!f) { perror("--oracle-ids"); return 2; }
        long id;
        int status;
        while ((status = fscanf(f, "%ld", &id)) == 1) oracle_ids.push_back(id);
        fclose(f);
        if (status != EOF || (long)oracle_ids.size() < gen_tokens) {
            fprintf(stderr, "invalid or short oracle ID stream\n");
            return 2;
        }
    }
    const bool oracle = !oracle_ids.empty();
    const long seqs = (multi > 0 || speculate > 0) ? 1 : batch;
    g_batch = batch;   // MUST follow the multi/speculate adjustment (E040 regression: GEMV<1> skipped rows 1..B-1)
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
    if (std::any_of(oracle_ids.begin(), oracle_ids.end(),
                    [&](long id) { return id < 0 || id >= m.n_vocab; })) {
        fprintf(stderr, "oracle ID outside the target vocabulary\n");
        return 2;
    }

    std::vector<GTensor> mtp_tensors;
    uint64_t mtp_data_start = 0;
    bool share_output = resident_output;
    if (resident_output && (outw->type != kGGML_Q4_0 || outw->ne[0] != m.n_embd || outw->ne[1] != m.n_vocab)) {
        fprintf(stderr, "resident output requires the target's Q4_0 output matrix\n");
        return 2;
    }
    if (!mtp_path.empty()) {
        GgufKV mtp_kv;
        if (!parse_gguf(mtp_path.c_str(), mtp_kv, mtp_tensors, mtp_data_start)) {
            fprintf(stderr, "mtp gguf parse failed\n");
            return 1;
        }
        for (const GTensor& t : mtp_tensors) {
            if (t.name != "output.weight" || t.type != kGGML_Q4_0 || outw->type != t.type ||
                t.nd != 2 || outw->nd != 2 || t.ne[0] != outw->ne[0] || t.ne[1] != outw->ne[1] ||
                t.bytes != outw->bytes) continue;
            const int target_fd = open(model.c_str(), O_RDONLY);
            const int draft_fd = open(mtp_path.c_str(), O_RDONLY);
            std::vector<uint8_t> target(1 << 20), draft(target.size());
            share_output = target_fd >= 0 && draft_fd >= 0;
            for (uint64_t off = 0; share_output && off < t.bytes; off += target.size()) {
                const size_t n = std::min<uint64_t>(target.size(), t.bytes - off);
                share_output = pread(target_fd, target.data(), n, data_start + outw->off + off) == (ssize_t)n &&
                               pread(draft_fd, draft.data(), n, mtp_data_start + t.off + off) == (ssize_t)n &&
                               memcmp(target.data(), draft.data(), n) == 0;
            }
            if (target_fd >= 0) close(target_fd);
            if (draft_fd >= 0) close(draft_fd);
            break;
        }
        printf("shared_output,verified=%d,bytes=%llu\n", share_output, (unsigned long long)(share_output ? outw->bytes : 0));
    }

    if (!eagle_path.empty()) {
        if (m.arch != "qwen35" || m.n_embd != Eagle4::hidden || m.n_layer != 64 || m.n_vocab != Eagle4::vocab ||
            m.sss_state != 128 || m.sss_conv != 4 || outw->type != kGGML_Q4_0) {
            fprintf(stderr, "EAGLE draft architecture does not match the target\n");
            return 2;
        }
        share_output = true;
    }
    const bool session_drafter = (mtp_tree && packed_verify) || !eagle_path.empty();
    if (session && (!session_drafter || A != "qwen35" || !share_output || dbg_dump || oracle ||
                    prompt_ids_s.empty() || !out_ids_path.empty() || !expect_ids_path.empty())) {
        fprintf(stderr, "--session needs packed native MTP or EAGLE4 and --prompt-ids; each turn emits session_ids on stdout\n");
        return 2;
    }
    if ((prefill_chunks != 1 && prefill_chunks != 2 && prefill_chunks != 4 && prefill_chunks != 8) ||
        (prefill_chunks > 1 && (A != "qwen35" || !mtp_tree || !packed_verify || !share_output || dbg_dump || oracle))) {
        fprintf(stderr, "--prefill-chunks takes 1, 2, 4 or 8; grouping needs packed native MTP with a shared output head\n");
        return 2;
    }
    if (session) show_text = 0;  // keep generated text out of the line protocol
    if (session_tail && (!session || !mtp_tree || !packed_verify || speculate > kPrefillChunk ||
                         m.sss_state != 128 || m.sss_conv != 4)) {
        fprintf(stderr, "--session-tail requires a packed native MTP session with --speculate at most 32\n");
        return 2;
    }
    if (session_prefix_tokens < 0 ||
        (session_prefix_tokens && (!session_tail || session_prefix_tokens % kPrefillChunk ||
            session_prefix_tokens >= ctx - gen_tokens))) {
        fprintf(stderr, "--session-prefix-tokens needs --session-tail 1 and a multiple of 32 below ctx minus gen-tokens\n");
        return 2;
    }
    if (session_draft_hints && (!session_tail || speculate != kPrefillChunk || prefill_chunks < 4)) {
        fprintf(stderr, "--session-draft-hints needs --session-tail 1, --speculate 32 and at least four prefill chunks\n");
        return 2;
    }
    if (session_stream && !session) {
        fprintf(stderr, "--session-stream requires --session 1\n");
        return 2;
    }
    const bool can_fuse_prefill_tail = session_tail && prefill_chunks >= 2 &&
        !session_draft_hints && (session_prefix_tokens == 0);
    if (fuse_prefill_tail < -1 || fuse_prefill_tail > 1 || (fuse_prefill_tail == 1 && !can_fuse_prefill_tail)) {
        fprintf(stderr, "--fuse-prefill-tail needs a grouped native MTP session without draft hints or a fixed shared prefix\n");
        return 2;
    }
    if (fuse_prefill_tail < 0) fuse_prefill_tail = can_fuse_prefill_tail;
    const bool tier_kv = kv_mode == "turboquant" || kv_mode == "turboquant-ttkv";
    if (kv_mode != "fp32" && (!tier_kv || A != "qwen35" || m.head_v != kTierHead || ctx % kTierOldGroup ||
                              kv_recent < 0 || kv_recent > ctx || kv_recent >= kTierOld ||
                              !mtp_tree || !packed_verify || oracle || validate || dbg_dump ||
                              session_draft_hints || session_prefix_tokens)) {
        fprintf(stderr, "--kv-cache turboquant[-ttkv] needs packed native MTP and context divisible by 16, without oracle or session hints\n");
        return 2;
    }
    printf("config,model=%s,arch=%s,layers=%ld,recr=%ld,attn=%ld,n_embd=%ld,n_head=%ld,n_kv=%ld,head=%ld,rot=%ld,ffn=%ld,vocab=%ld,ssm_state=%ld,ssm_group=%ld,ssm_dt=%ld,ssm_inner=%ld,rope_base=%.0f,ctx=%ld,batch=%ld,hot_mib=%ld,cache_mib=%ld,validate=%d\n",
           model.c_str(), A.c_str(), m.n_layer, m.n_recr, m.n_attn, m.n_embd, m.n_head, m.n_kv, m.head_v, m.n_rot, m.n_ff,
           m.n_vocab, m.sss_state, m.sss_group, m.sss_dt, m.sss_inner, m.rope_base, ctx, seqs, hot_mib, cache_mib, validate);
    if (speculate > 0) {
        printf("spec_config,user_sequences=1,max_verify_positions=%ld,draft_max=%ld,draft_vocab=%ld,p_min=%.3f,pld_mode=%d,balanced_residency=%d,drafter=%s,mtp_group=%d,packed_verify=%d\n",
               speculate, draft_max, draft_vocab, p_min, pld_enabled, balance_hot,
               oracle ? "oracle" : parallel_on ? "parallel" : !eagle_path.empty() ? "eagle4" :
               mtp_tree ? "mtp_tree" : !mtp_path.empty() ? "mtp" : "history", mtp_tree ? mtp_group : 1, packed_verify);
    }

    if (packed_verify && ((!mtp_tree && eagle_path.empty() && !resident_output) || speculate > 64)) {
        fprintf(stderr, "--packed-verify requires a resident output matrix and up to 64 verification positions\n"); return 2;
    }
    if (compress_packed && !packed_verify) {
        fprintf(stderr, "--compress-packed requires --packed-verify\n");
        return 2;
    }
    g_packed_widths = packed_verify;
    // ---- items: chunk every Q4_0 tensor except the embedding table ----
    struct Item { std::string tensor; long K, N, row0, rows; uint64_t file_off, bytes, transfer, aligned_off, head, read_len; int tier = 0; bool packed = false; int64_t res_off = -1; };
    enum { kNvme = 0, kHot = 1, kStaged = 2 };
    std::map<std::string, std::vector<int>> tensor_items;
    std::vector<Item> items; uint64_t total_bytes = 0;
    for (const GTensor& t : gtensors) {
        if (t.type != kGGML_Q4_0 || t.name == "token_embd.weight") continue;
        if (share_output && t.name == "output.weight") continue;
        if (t.ne[0] % 32) { fprintf(stderr, "tensor %s K not /32\n", t.name.c_str()); return 1; }
        const uint64_t rowbytes = (uint64_t)(t.ne[0] / 32) * 18;
        const std::string tn = t.name;
        const bool packed = packed_verify && t.ne[0] >= 5120 && t.ne[0] % 1024 == 0 &&
            t.ne[1] >= 1024 && t.ne[1] % 16 == 0 && tn != "output.weight";
        const long raw_per = (long)(kMaxItem / rowbytes);
        const long per = packed ? raw_per / 16 * 16 : raw_per;
        for (long r = 0; r < t.ne[1]; r += per) {
            Item it; it.tensor = tn; it.K = t.ne[0]; it.N = t.ne[1];
            it.row0 = r; it.rows = std::min<long>(per, t.ne[1] - r); it.packed = packed;
            it.file_off = data_start + t.off + (uint64_t)r * rowbytes;
            it.bytes = (uint64_t)it.rows * rowbytes;
            it.transfer = it.bytes;
            it.aligned_off = it.file_off & ~4095ull; it.head = it.file_off - it.aligned_off;
            it.read_len = align_up(it.head + it.bytes, 4096);
            tensor_items[tn].push_back((int)items.size());
            items.push_back(it); total_bytes += it.bytes;
        }
    }

    // Spread the hot budget across the layers in file order. Putting the entire
    // hot arena at the front of the model leaves PCIe idle while those layers run.
    uint64_t hot_used = 0, staged_used = 0;
    const uint64_t hot_limit = (uint64_t)hot_mib << 20;
    const double hot_fraction = total_bytes ? std::min(1.0, (double)hot_limit / total_bytes) : 1.0;
    uint64_t visited_bytes = 0;
    for (Item& it : items) {
        visited_bytes += it.bytes;
        if ((!balance_hot || speculate <= 0 || hot_used < visited_bytes * hot_fraction) &&
            hot_used + it.bytes + 16 <= hot_limit) {
            it.tier = kHot; it.res_off = (int64_t)hot_used; hot_used = align_up(hot_used + it.bytes, 256);
        }
    }
    // Fill the remaining hot tail space with any whole item that fits.
    for (Item& it : items) {
        if (it.tier == kHot) continue;
        if (hot_used + it.bytes + 16 <= hot_limit) {
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
    // VRAM preflight: hot + slots + states must keep >= 2 GiB free.
    // E044: in multi/speculate mode the rows are positions of ONE sequence (kv_seq_stride 0,
    // single-slice SSM) — states are sized for 1 sequence, not B. At batch 8 this frees
    // ~1.27 GiB of VRAM (was the reason gamma=16 could not run at the fast hot config).
    const long sss_v = m.sss_dt ? m.sss_inner / m.sss_dt : 0;
    const uint64_t ssm_vram = ((uint64_t)m.n_recr * m.sss_dt * (uint64_t)sss_v * sss_v * 4
                            + (uint64_t)m.n_recr * m.qkv_out * (m.sss_conv ? m.sss_conv - 1 : 0) * 4) * seqs;
    const bool wide_prefill = !eagle_path.empty() || ((mtp_tree || resident_output) && share_output && !dbg_dump);
    // E167: ordinary 64-row dispatch can change near-tie argmaxes. Grouped
    // prefill keeps the per-row reduction and shares the weight transfer.
    const long prefill_width = wide_prefill ? kPrefillChunk : speculate;
    const long standard_rows = std::max(batch, prefill_width);
    const long B = std::max(batch, prefill_chunks * prefill_width);
    const bool large_tier = tier_kv && ctx > 24576;
    TierCache tier_cache;
    tier_cache.recent_tokens = (int)kv_recent;
    tier_cache.ring = (int)kv_recent + 2 * B + kTierOldGroup;
    tier_cache.capacity = (int)ctx;
    tier_cache.pooling = kv_mode == "turboquant-ttkv";
    // Old protected groups still have their individual rows. Only the active
    // middle tier needs 4:1 rows at larger capacities.
    tier_cache.middle_groups = large_tier ?
        std::min(ctx, (long)kTierOld + 2 * B + kTierOldGroup) / kTierMidGroup : ctx / kTierMidGroup;
    const long tier_splits = (ctx + kTierSplit - 1) / kTierSplit;
    const long tier_batch = large_tier ? std::min(B, kPrefillChunk) : B;
    const uint64_t tier_scratch = (uint64_t)m.n_head *
        (B * kTierHead + tier_batch * tier_splits * (kTierHead + 2)) * sizeof(float) + ctx;
    const uint64_t kv_vram = tier_kv ? tier_cache.bytes(m.n_attn * m.n_kv) + tier_scratch :
        (uint64_t)m.n_attn * m.n_kv * ctx * m.head_v * 4 * 2 * seqs;
    const long mtp_window = large_tier ? 8192 : ctx;
    const long mtp_capacity = large_tier ? mtp_window + 2 * B + 2 * batch : ctx;
    const uint64_t mtp_kv_bytes = (uint64_t)m.n_kv * mtp_capacity * m.head_v * 2 * sizeof(float);
    const uint64_t draft_reserve = !eagle_path.empty() ? Eagle4::device_bytes(ctx, B) :
        resident_output ? outw->bytes + (uint64_t)parallel_stat.st_size + (4ull << 20) :
        (mtp_path.empty() ? 0 : (916ull << 20) + std::max(mtp_kv_bytes, uint64_t{64} << 20));
    size_t total_vram = 0;
    // E167: speculative mode keeps per-layer replay inputs for the accepted-prefix
    // re-feed; sized by the (already adjusted) batch, so deep gamma grows it
    const uint64_t replay_reserve = (speculate > 0 && m.n_recr && m.sss_dt)
        ? (uint64_t)m.n_recr * 2 * ((uint64_t)batch * (m.qkv_out + m.sss_dt)) * 4 : 0;
    // Additional per-row scratch for the Qwen 27B grouped-prefill path.
    const uint64_t prefill_reserve = (uint64_t)(B - standard_rows) * (512ull << 10);
    const uint64_t replay_bytes = session_draft_hints || fuse_prefill_tail
        ? std::max(replay_reserve, ssm_vram) : replay_reserve;
    const uint64_t vram_plan = hot_used + (uint64_t)kSlots * kMaxItem + ssm_vram + kv_vram + draft_reserve +
        replay_bytes + prefill_reserve + (16u << 20);
    {
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "props");
        total_vram = prop.totalGlobalMem;
        const long score_limit = tier_kv ? mtp_window + (large_tier ? batch : 0) : ctx;
        if (m.n_attn && (uint64_t)score_limit * sizeof(float) + 2 * sizeof(float) > prop.sharedMemPerBlock) {
            cudaFuncAttributes attr{};
            check(cudaFuncGetAttributes(&attr, k_attn_decode), "attention resources");
            const uint64_t attention_bytes = (uint64_t)score_limit * sizeof(float);
            if (attention_bytes + attr.sharedSizeBytes > prop.sharedMemPerBlockOptin) {
                fprintf(stderr, "context %ld exceeds the GPU attention scratch capacity\n", ctx);
                return 4;
            }
            check(cudaFuncSetAttribute(k_attn_decode, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)attention_bytes), "long-context attention scratch");
        }
    }
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
    for (const GTensor& t : gtensors) if (t.type == kGGML_Q4_0 && t.name != "token_embd.weight" && t.name != "output.weight") max_rows = std::max(max_rows, t.ne[1]);
    max_rows += 16;
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
    const bool compact_target_logits = speculate > 0 && !dbg_dump;
    const long logit_rows = compact_target_logits ? speculate : B;
    check(cudaMalloc((void**)&logits, logit_rows * m.n_vocab * sizeof(float)), "logits");
    const int logit_tiles = (m.n_vocab + 1023) / 1024;
    LogitMax *tile_maxima = nullptr, *row_maxima = nullptr;
    std::vector<LogitMax> target_maxima(compact_target_logits ? B : 0);
    if (compact_target_logits) {
        check(cudaMalloc(&tile_maxima, logit_rows * logit_tiles * sizeof(LogitMax)), "logit tiles");
        check(cudaMalloc(&row_maxima, logit_rows * sizeof(LogitMax)), "logit maxima");
    }
    const uint64_t xq_bytes = (uint64_t)B * max_K;
    const uint64_t qmeta_bytes = (uint64_t)B * max_K / 32 * sizeof(int2);
    // E190/A6: keep [xq][xmeta] in BOTH modes so a per-tensor persisting window can
    // span a tensor's whole quantized slice contiguously from the arena base. The old
    // prefill layout put xmeta first, which forced the E052 window to span the gap and
    // blew the persisting cap — the window then stayed disabled at all widths.
    const uint64_t meta_off = align_up(xq_bytes, 256);
    const uint64_t quant_off = 0;
    const uint64_t activation_bytes = meta_off + qmeta_bytes;
    const uint64_t decode_activation_bytes = B > standard_rows
        ? quant_off + (uint64_t)standard_rows * max_K : activation_bytes;
    uint8_t* activation_buf = nullptr;
    check(cudaMalloc((void**)&activation_buf, activation_bytes), "activation arena");
    check(cudaMemset(activation_buf, 0, activation_bytes), "initialize padded activation rows");
    int8_t* xq = (int8_t*)(activation_buf + quant_off);
    int2* xmeta = (int2*)(activation_buf + meta_off);
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
    // KV + SSM state (E044: sized for seqs — 1 slice suffices in multi/speculate mode,
    // rows share slice 0; batch mode keeps one slice per sequence)
    float *kv_dev = nullptr, *ssm_dev = nullptr, *conv_dev = nullptr;
    const uint64_t kv_per_layer = (uint64_t)m.n_kv * ctx * m.head_v * 4;
    float *tier_queries = nullptr, *tier_partial = nullptr;
    uint8_t* tier_protection = nullptr;
    std::vector<uint8_t> tier_flags;
    std::vector<uint8_t> tier_structural;
    long tier_packed = 0, tier_written = 0;
    if (tier_kv) {
        tier_cache.allocate(m.n_attn * m.n_kv);
        check(cudaMalloc(&tier_queries, (size_t)B * m.n_head * kTierHead * sizeof(float)), "rotated KV queries");
        check(cudaMalloc(&tier_partial, (size_t)tier_batch * m.n_head * tier_splits * (kTierHead + 2) * sizeof(float)), "tiered attention partials");
        check(cudaMalloc(&tier_protection, ctx), "TTKV structural protection");
        tier_flags.resize(ctx);
        printf("kv_cache,mode=%s,cache_mib=%.1f,scratch_mib=%.1f,recent=%d,old_age=%d\n",
            kv_mode.c_str(), tier_cache.bytes(m.n_attn * m.n_kv) / 1048576.0, tier_scratch / 1048576.0,
            tier_cache.recent_tokens, kTierOld);
    } else if (m.n_attn) check(cudaMalloc((void**)&kv_dev, kv_per_layer * 2 * m.n_attn * seqs), "kv cache");
    const uint64_t sss = m.sss_inner / m.sss_dt;
    if (m.n_recr) {
        check(cudaMalloc((void**)&ssm_dev, (size_t)m.n_recr * m.sss_dt * sss * sss * 4 * seqs), "ssm state");
        check(cudaMalloc((void**)&conv_dev, (size_t)m.n_recr * m.qkv_out * (m.sss_conv - 1) * 4 * seqs), "conv state");
        check(cudaMemset(ssm_dev, 0, (size_t)m.n_recr * m.sss_dt * sss * sss * 4 * seqs), "ssm zero");
        check(cudaMemset(conv_dev, 0, (size_t)m.n_recr * m.qkv_out * (m.sss_conv - 1) * 4 * seqs), "conv zero");
    }
    // pinned stage ring + staged cache
    // E048: stage + pinned cache are write-combined — the CPU only ever WRITES them;
    // WC lifts the H2D rate (the GPU is the only reader)
    uint8_t* stage[kStage]; for (auto& s : stage) check(cudaHostAlloc((void**)&s, kStageBytes + 4096, cudaHostAllocPortable), "stage");
    uint8_t* staged_cache = nullptr; uint8_t* pinned_cache = nullptr;
    if (staged_used) {
        if (pin_bytes >= staged_used) {
            // full-pin regime (memlock allows): one pinned copy, no unpinned duplicate.
            // Every kStaged item is in_pin, so staged_cache is never dereferenced.
            check(cudaHostAlloc((void**)&pinned_cache, staged_used, cudaHostAllocPortable), "pinned cache");
        } else {
            if (posix_memalign((void**)&staged_cache, 4096, staged_used)) { fprintf(stderr, "cache alloc\n"); return 1; }
            if (pin_bytes) check(cudaHostAlloc((void**)&pinned_cache, pin_bytes, cudaHostAllocPortable), "pinned cache");
        }
    }
    std::vector<char> cached(items.size(), 0);
    int fd = open(model.c_str(), O_RDONLY | (o_direct ? O_DIRECT : 0)); if (fd < 0) { perror("open o_direct"); return 1; }
    int fdc = open(model.c_str(), O_RDONLY); if (fdc < 0) { perror("open"); return 1; }
    cudaStream_t st, stc, st3; check(cudaStreamCreate(&st), "stream"); check(cudaStreamCreate(&stc), "compute stream");
    check(cudaStreamCreate(&st3), "cpu exchange stream");   // E193: D2H/H2D for the CPU row split
    check(cudaFuncSetSharedMemConfig(k_mat_q4_shared<16>, cudaSharedMemBankSizeEightByte), "GEMV shared bank mode");
    if (B >= 12) check(cudaFuncSetSharedMemConfig(k_mat_q4_shared<12>, cudaSharedMemBankSizeEightByte), "12-position GEMV shared bank mode");
    if (B >= 24) check(cudaFuncSetSharedMemConfig(k_mat_q4_shared<24>, cudaSharedMemBankSizeEightByte), "24-position GEMV shared bank mode");
    if (B >= 32) check(cudaFuncSetSharedMemConfig(k_mat_q4_shared<32>, cudaSharedMemBankSizeEightByte), "wide GEMV shared bank mode");
    // E052: keep the small, repeatedly-read quantized activation working set in L2 while
    // one-pass model weights stream through. xq/xmeta share one arena so the window
    // covers all three arrays. Skip cleanly if the configured batch exceeds the device's
    // persisting-cache window.
    cudaDeviceProp l2prop{}; check(cudaGetDeviceProperties(&l2prop, 0), "L2 cache properties");
    // E190/A6: the persisting window is set PER TENSOR inside mat() (span = that
    // tensor's xq+xmeta slice, hitRatio scaled to the cap when it overflows). This
    // replaces the E052 whole-arena window, which never engaged at prefill widths.
    bool l2_win = l2_win_flag && l2prop.persistingL2CacheMaxSize > 0;
    if (l2_win) {
        check(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)l2prop.persistingL2CacheMaxSize), "persisting L2 limit");
        printf("activation_l2_window,mode=per-tensor,persisting_limit=%d\n", l2prop.persistingL2CacheMaxSize);
    } else {
        printf("activation_l2_window,disabled=%s\n", l2_win_flag ? "device" : "flag");
    }
    // ---- E193: CPU row-split second stream ------------------------------------
    // Worker thread computes a fraction of each tensor's rows on the CPU (E191
    // kernel, packed-tile layout straight from pinned_cache) while the GPU streams
    // the rest. Split is per-tensor at item granularity; activations reach the CPU
    // via D2H on st (PCIe is full-duplex — this does not steal H2D weight bytes).
    cudaEvent_t cpu_qev[2], cpu_qh[2], cpu_oh[2];
    int8_t* cpu_xq_stage[2] = {nullptr, nullptr};
    int2* cpu_meta_stage[2] = {nullptr, nullptr};
    float* cpu_out_stage[2] = {nullptr, nullptr};
    bool cpu_worker_on = false;
    long cpu_max_N = 0;
    int cpu_ring = 0;
    cudaEvent_t cpu_xq_guard = nullptr;   // E193: last outstanding quant D2H; xq must not be overwritten before it lands
    std::atomic<int> cpu_job_ready{0}, cpu_job_done{1};
    struct CpuRowJob {
        const uint8_t* w; int K; int rows; int B; long N; long row_start;
        const int8_t* xq; const int2* xmeta; float* out;
        cudaEvent_t qev;
    } cpu_job{};
    if (cpu_rows_frac > 0.001f && standard_rows <= 64) {
        for (const auto& kv : tensor_items) {
            const Item& it0 = items[kv.second[0]];
            if (it0.packed) cpu_max_N = std::max<long>(cpu_max_N, m.by_name.at(kv.first)->ne[1]);
        }
        if (cpu_max_N > 0) {
            for (int i = 0; i < 2; ++i) {
                check(cudaHostAlloc((void**)&cpu_xq_stage[i], (size_t)standard_rows * max_K, cudaHostAllocDefault), "cpu xq stage");
                check(cudaHostAlloc((void**)&cpu_meta_stage[i], (size_t)standard_rows * (max_K / 32) * sizeof(int2), cudaHostAllocDefault), "cpu meta stage");
                check(cudaHostAlloc((void**)&cpu_out_stage[i], (size_t)standard_rows * cpu_max_N * 4, cudaHostAllocDefault), "cpu out stage");
                check(cudaEventCreateWithFlags(&cpu_qev[i], cudaEventDisableTiming), "cpu qev");
                check(cudaEventCreateWithFlags(&cpu_qh[i], cudaEventDisableTiming), "cpu qh");
                check(cudaEventCreateWithFlags(&cpu_oh[i], cudaEventDisableTiming), "cpu oh");
            }
            cpu_worker_on = true;
            std::thread([](std::atomic<int>* ready, std::atomic<int>* done, CpuRowJob* job) {
                for (;;) {
                    while (!ready->load(std::memory_order_acquire)) std::this_thread::yield();
                    ready->store(0, std::memory_order_release);
                    if (getenv("GUM_E193_DEBUG")) E193LOG("worker sync qev\n");
                    check(cudaEventSynchronize(job->qev), "cpu worker quant sync");
                    if (getenv("GUM_E193_DEBUG")) E193LOG("worker compute rows=%d K=%d\n", job->rows, job->K);
                    cpu_q4_rows(job->w, job->K, job->rows, job->B, job->xq, job->xmeta, job->out, job->N);
                    if (getenv("GUM_E193_DEBUG")) E193LOG("worker done\n");
                    done->store(1, std::memory_order_release);
                }
            }, &cpu_job_ready, &cpu_job_done, &cpu_job).detach();
            printf("cpu_rows,frac=%.3f,max_N=%ld,stage_K=%ld\n", cpu_rows_frac, cpu_max_N, (long)max_K);
        }
    }

    cudaEvent_t ev[kStage], evc[kSlots], h2d_start[kStage];
    for (auto& e : ev) check(cudaEventCreate(&e), "event");
    for (auto& e : evc) check(cudaEventCreate(&e), "event");
    if (measure_h2d) for (auto& e : h2d_start) check(cudaEventCreate(&e), "H2D timing event");
    for (auto& e : evc) check(cudaEventRecord(e, stc), "evc prime");
    check(cudaStreamSynchronize(stc), "evc prime sync");

    // E169 retains the E159 record layout: three fp32 header values, then one
    // batch of residuals per selected layer (ascending), then the final batch.
    FILE* capture_file = nullptr;
    float *capture_dev = nullptr, *capture_host = nullptr;
    std::vector<long> capture_slots;
    long capture_mid = 0;
    size_t capture_row_floats = 0, capture_payload_floats = 0;
    if (capture_layers) {
        capture_slots.assign(m.n_layer, -1);
        for (const char* p = capture_layers; ; ) {
            char* end = nullptr;
            const long layer = strtol(p, &end, 10);
            if (end == p || layer < 0 || layer >= m.n_layer || (*end && *end != ',')) {
                fprintf(stderr, "invalid GUM_MID_CAPTURE layer list: %s\n", capture_layers);
                return 2;
            }
            capture_slots[layer] = 0;
            if (!*end) break;
            p = end + 1;
        }
        for (long& slot : capture_slots) if (slot == 0) slot = capture_mid++;
        capture_row_floats = (size_t)batch * m.n_embd;
        capture_payload_floats = (capture_mid + 1) * capture_row_floats;
        capture_file = fopen(capture_path, "wb");
        if (!capture_file) { perror(capture_path); return 1; }
        check(cudaMalloc(&capture_dev, capture_payload_floats * sizeof(float)), "capture dev");
        check(cudaHostAlloc(&capture_host, (capture_payload_floats + 3) * sizeof(float),
            cudaHostAllocDefault), "capture host");
        printf("mid_capture,layers=%s,file=%s\n", capture_layers, capture_path);
    }

    // ---- schedule: item indices in decode order ----
    std::vector<int> sched; std::vector<int> sched_row0;  // logits row0 if output chunk
    auto push_tensor = [&](const std::string& tn) {
        auto itn = tensor_items.find(tn);
        if (itn == tensor_items.end()) { fprintf(stderr, "missing tensor %s\n", tn.c_str()); exit(1); }
        for (int idx : itn->second) sched.push_back(idx);
    };
    for (long il = 0; il < m.n_layer; ++il) {
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
    if (!share_output) push_tensor("output.weight");

    // ---- tokenizer ----
    Tok tok;
    tok.vocab = kv.arr_strs["tokenizer.ggml.tokens"];
    for (size_t i = 0; i < tok.vocab.size(); ++i) tok.id_of[tok.vocab[i]] = (long)i;
    tok.bos = m.bos_id; tok.eos = m.eos_id;
    if (tier_kv) {
        tier_structural.reserve(tok.vocab.size());
        for (const auto& piece : tok.vocab) {
            const auto token = tok.unescape(piece);
            const bool structural = token.find_first_of("0123456789_/\\") != std::string::npos ||
                token.compare(0, 2, "<|") == 0 || token.find("tool_") != std::string::npos ||
                (token.find_first_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 1) != std::string::npos &&
                 token.find_first_of("abcdefghijklmnopqrstuvwxyz") != std::string::npos);
            tier_structural.push_back(structural);
        }
    }
    std::vector<std::vector<long>> prompts;
    auto parse_session_ids = [&](const std::string& text, std::vector<long>& ids) {
        for (const char* p = text.c_str(); ; ) {
            char* end = nullptr;
            const long id = strtol(p, &end, 10);
            if (end == p || (*end && *end != ',') || id < 0 || id >= m.n_vocab) {
                fprintf(stderr, "session expects one comma-separated sequence of valid token IDs per line\n");
                return false;
            }
            ids.push_back(id);
            if (!*end) return true;
            p = end + 1;
        }
    };
    // Session clients can prefix the usual request with a smaller output limit.
    auto parse_session_request = [&](std::string text, std::vector<long>& ids, std::vector<long>& draft,
                                     long& request_tokens) {
        request_tokens = gen_tokens;
        const size_t colon = text.find(':');
        if (colon != std::string::npos) {
            const std::string limit = text.substr(0, colon);
            char* end = nullptr;
            request_tokens = strtol(limit.c_str(), &end, 10);
            if (*end || request_tokens < 1 || request_tokens > gen_tokens) {
                fprintf(stderr, "session output limit must be in 1..%ld\n", gen_tokens);
                return false;
            }
            text.erase(0, colon + 1);
        }
        const size_t separator = session_draft_hints ? text.find('|') : std::string::npos;
        if (separator == std::string::npos) return parse_session_ids(text, ids);
        return parse_session_ids(text.substr(0, separator), ids) &&
            (separator + 1 == text.size() || parse_session_ids(text.substr(separator + 1), draft));
    };
    std::vector<long> first_draft;
    long first_gen_tokens = gen_tokens;
    if (session) {
        if (prompt_ids_s == "-" && !std::getline(std::cin, prompt_ids_s)) {
            fprintf(stderr, "missing first session request on stdin\n");
            return 2;
        }
        prompts.emplace_back();
        if (!parse_session_request(prompt_ids_s, prompts.back(), first_draft, first_gen_tokens)) return 2;
    } else if (!prompt_ids_s.empty()) {
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
    std::vector<long> feed;   // E040: teacher-forced continuation (the model's own tokens)
    if (!feed_ids_s.empty()) {
        size_t p = 0;
        while (p < feed_ids_s.size()) {
            size_t q = feed_ids_s.find(',', p);
            if (q == std::string::npos) q = feed_ids_s.size();
            feed.push_back(atol(feed_ids_s.substr(p, q - p).c_str()));
            p = q + 1;
        }
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
        const auto hot_start = std::chrono::steady_clock::now();
        uint8_t* raw_device = slots;
        if (packed_verify && !raw_device) check(cudaMalloc(&raw_device, kMaxItem), "hot pack scratch");
        const long upload_slots = packed_verify && !slots ? 1 : std::min(kStage, kSlots);
        long n_hot_items = 0;
        for (const Item& it : items) {
            if (it.tier != kHot) continue;
            const long slot = n_hot_items % upload_slots;
            check(cudaEventSynchronize(evc[slot]), "hot upload slot");
            const ssize_t got = pread(fdc, stage[slot], it.bytes, (off_t)it.file_off);
            if (got < (ssize_t)it.bytes) { fprintf(stderr, "hot pread %s\n", it.tensor.c_str()); return 1; }
            uint8_t* destination = it.packed ? raw_device + slot * kMaxItem : hot + it.res_off;
            check(cudaMemcpyAsync(destination, stage[slot], it.bytes, cudaMemcpyHostToDevice, st), "hot upload");
            if (it.packed) {
                check(cudaEventRecord(ev[slot], st), "hot raw upload ready");
                check(cudaStreamWaitEvent(stc, ev[slot], 0), "hot pack wait");
                const int blocks = (int)(it.K / 32);
                const int tiles = (int)(it.rows / 16) * blocks;
                k_pack_q4_tiles<<<(tiles + 3) / 4, 128, 0, stc>>>(
                    reinterpret_cast<uint16_t*>(hot + it.res_off), reinterpret_cast<const uint16_t*>(destination), blocks, tiles);
                check(cudaGetLastError(), "hot pack kernel");
            }
            check(cudaEventRecord(evc[slot], it.packed ? stc : st), "hot upload slot complete");
            ++n_hot_items;
        }
        check(cudaStreamSynchronize(st), "hot uploads complete");
        check(cudaStreamSynchronize(stc), "hot packing complete");
        if (raw_device != slots) check(cudaFree(raw_device), "free hot pack scratch");
        if (verify_hot) {
            long bad = 0;
            std::vector<uint8_t> bounce(kMaxItem), roundtrip(kMaxItem), repacked(packed_verify ? kMaxItem : 0);
            for (size_t i = 0; i < items.size(); ++i) {
                Item& it = items[i]; if (it.tier != kHot) continue;
                const ssize_t got = pread(fdc, bounce.data(), it.bytes, (off_t)it.file_off);
                check(cudaMemcpy(roundtrip.data(), hot + it.res_off, it.bytes, cudaMemcpyDeviceToHost), "hot verify d2h");
                const uint8_t* expect = bounce.data();
                if (it.packed && got == (ssize_t)it.bytes) {
                    pack_q4_tiles(repacked.data(), bounce.data(), it.K, it.rows);
                    expect = repacked.data();
                }
                if (got < (ssize_t)it.bytes || memcmp(expect, roundtrip.data(), it.bytes) != 0) {
                    fprintf(stderr, "HOT MISMATCH item %zu tensor %s rows %ld-%ld\n", i, it.tensor.c_str(), it.row0, it.row0 + it.rows);
                    ++bad;
                }
            }
            printf("hot_verify,items=%ld,bad=%ld\n", n_hot_items, bad);
            if (bad) return 1;
        }
        printf("hot_load,items=%ld,GiB=%.2f,seconds=%.4f,gpu_pack=%d\n", n_hot_items, hot_used / 1073741824.0,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - hot_start).count(), packed_verify);
    }

    // ---- decode loop ----
    const long hd = m.head_v, nh_q = m.n_head, nh_k = m.n_kv;
    const long Sv = m.sss_dt ? m.sss_inner / m.sss_dt : 1;
    const float attn_scale = 1.f / sqrtf((float)hd);
    const float dn_scale = 1.f / sqrtf((float)Sv);
    const long emb_nb = m.n_embd / 32;
    uint8_t* emb_buf = nullptr; check(cudaHostAlloc((void**)&emb_buf, (size_t)B * emb_nb * 18 + 4096, cudaHostAllocPortable), "emb buf");
    std::vector<float> emb_row((size_t)B * m.n_embd), logits_h((size_t)(compact_target_logits ? 1 : B) * m.n_vocab);

    // ---- E046: MTP (nextn) drafter ----
    // The qwen3.5 nextn block (blk.64 in the mtp gguf): hnorm(h) + enorm(emb(token)) ->
    // eh_proj -> one gated-attention layer (own KV, same rope/heads as the main model) ->
    // shared_head_norm -> shared lm_head. token_embd and output.weight are byte-identical
    // to the main model's (sha256-verified), so embeddings reuse the main pread path and
    // only ~230 MiB of unique weights are loaded; the head is another 682 MiB.
    // Scheme (mirrors llama.cpp speculative.cpp): the main pass stores its final-normed
    // hidden per position (h_tgt); the MTP block is synced over accepted rows with h
    // shifted one position back, then drafts recursively (h = the MTP's own output).
    bool mtp_on = false;
    float* mtp_tree_hidden = nullptr;
    float* mtp_tree_inputs = nullptr;
    int *mtp_tree_depth = nullptr, *mtp_tree_path = nullptr;
    const bool eagle_on = !eagle_path.empty();
    const uint8_t* output_device = nullptr;
    Eagle4 eagle;
    ParallelDraft parallel;
    uint8_t *mtp_dev = nullptr, *mtp_small_dev = nullptr;
    float *mtp_kv_dev = nullptr, *mtp_last_hidden = nullptr;
    float *mtp_cat = nullptr, *mtp_normed = nullptr, *mtp_resid = nullptr, *mtp_hout = nullptr, *mtp_hout2 = nullptr;
    std::map<std::string, uint64_t> mtp_off, mtp_small_off;
    uint64_t mtp_vram = 0;
    if (!mtp_path.empty()) {
        uint64_t q4tot = 0, f32tot = 0;
        std::vector<std::pair<std::string, const GTensor*>> q4s, f32s;
        for (const GTensor& t : mtp_tensors) {
            if (t.type == kGGML_Q4_0 && t.name != "token_embd.weight") { q4s.push_back({t.name, &t}); q4tot += align_up(t.bytes, 256); }
            else if (t.type == kGGML_F32) { f32s.push_back({t.name, &t}); f32tot += align_up(t.bytes, 256); }
        }
        mtp_vram = q4tot + f32tot + mtp_kv_bytes + (uint64_t)m.n_embd * 4;
        check(cudaMalloc((void**)&mtp_dev, q4tot), "mtp q4 blob");
        check(cudaMalloc((void**)&mtp_small_dev, f32tot), "mtp f32 blob");
        check(cudaMalloc((void**)&mtp_kv_dev, mtp_kv_bytes), "mtp kv");
        check(cudaMemset(mtp_kv_dev, 0, mtp_kv_bytes), "mtp kv zero");
        check(cudaMalloc(&mtp_last_hidden, m.n_embd * sizeof(float)), "last target hidden state");
        check(cudaMemset(mtp_last_hidden, 0, m.n_embd * sizeof(float)), "initial target hidden state");
        check(cudaMalloc((void**)&mtp_cat, (size_t)B * 2 * m.n_embd * 4), "mtp cat");
        check(cudaMalloc((void**)&mtp_normed, (size_t)B * m.n_embd * 4), "mtp normed");
        check(cudaMalloc((void**)&mtp_resid, (size_t)B * m.n_embd * 4), "mtp resid");
        check(cudaMalloc((void**)&mtp_hout, (size_t)B * m.n_embd * 4), "mtp hout");
        check(cudaMalloc((void**)&mtp_hout2, (size_t)(mtp_tree ? mtp_group : 1) * m.n_embd * 4), "mtp hout2");
        { uint64_t o = 0; std::vector<uint8_t> hb;
          for (auto& p : q4s) {
              hb.resize(p.second->bytes);
              FILE* f = fopen(mtp_path.c_str(), "rb");
              if (!f || fseek(f, (long)(mtp_data_start + p.second->off), SEEK_SET) || fread(hb.data(), 1, p.second->bytes, f) != p.second->bytes) { fprintf(stderr, "mtp read %s\n", p.first.c_str()); return 1; }
              fclose(f);
              check(cudaMemcpy(mtp_dev + o, hb.data(), p.second->bytes, cudaMemcpyHostToDevice), "mtp q4 h2d");
              mtp_off[p.first] = o; o = align_up(o + p.second->bytes, 256);
          } }
        { uint64_t o = 0; std::vector<uint8_t> hb;
          for (auto& p : f32s) {
              hb.resize(p.second->bytes);
              FILE* f = fopen(mtp_path.c_str(), "rb");
              if (!f || fseek(f, (long)(mtp_data_start + p.second->off), SEEK_SET) || fread(hb.data(), 1, p.second->bytes, f) != p.second->bytes) { fprintf(stderr, "mtp read %s\n", p.first.c_str()); return 1; }
              fclose(f);
              check(cudaMemcpy(mtp_small_dev + o, hb.data(), p.second->bytes, cudaMemcpyHostToDevice), "mtp f32 h2d");
              mtp_small_off[p.first] = o; o = align_up(o + p.second->bytes, 256);
          } }
        mtp_on = true;
        if (share_output) output_device = mtp_dev + mtp_off.at("output.weight");
        printf("mtp_plan,q4_MiB=%.0f,f32_MiB=%.0f,total_MiB=%.0f,window=%ld\n",
            q4tot / 1048576.0, f32tot / 1048576.0, mtp_vram / 1048576.0, mtp_window);
    }

    if (eagle_on || resident_output) {
        uint8_t* head = nullptr;
        check(cudaMalloc(&head, outw->bytes), "target output tensor");
        std::vector<uint8_t> buffer(kMaxItem);
        for (uint64_t off = 0; off < outw->bytes; off += buffer.size()) {
            const size_t bytes = std::min<uint64_t>(buffer.size(), outw->bytes - off);
            if (pread(fdc, buffer.data(), bytes, data_start + outw->off + off) != (ssize_t)bytes) {
                fprintf(stderr, "target output read failed\n"); return 1;
            }
            check(cudaMemcpy(head + off, buffer.data(), bytes, cudaMemcpyHostToDevice), "target output upload");
        }
        output_device = head;
        if (eagle_on) eagle.load(eagle_path, (int)ctx, (int)B, stc);
        if (parallel_on && !parallel.load(parallel_path, m.n_embd, m.n_vocab)) {
            fprintf(stderr, "parallel draft checkpoint could not be loaded\n");
            return 2;
        }
    }
    auto eagle_embeddings = [&](const std::vector<long>& ids) {
        for (size_t row = 0; row < ids.size(); ++row) {
            const ssize_t bytes = emb_nb * 18;
            if (pread(fdc, emb_buf + row * bytes, bytes, data_start + embd->off + ids[row] * bytes) != bytes) {
                fprintf(stderr, "EAGLE embedding read failed\n");
                exit(1);
            }
            cpu_dequant_row_q4(emb_buf + row * bytes, m.n_embd, emb_row.data() + row * m.n_embd);
        }
        check(cudaMemcpy(eagle.embeddings, emb_row.data(), ids.size() * m.n_embd * sizeof(float),
            cudaMemcpyHostToDevice), "EAGLE embeddings");
    };

    double best_tok_s = 1e30, sum_steady_s = 0; long n_steady = 0; long free_min = free_after_alloc;
    uint64_t steady_cold_bytes = 0; long n_steady_cold = 0;
    long checked = 0, mismatches = 0; double worst_rel = 0;
    std::vector<long> all_ids;          // every token fed (prompt + generated)
    std::vector<long> generated;        // argmax tokens produced
    long cur_pos = 0;                   // sequence position for rope/kv

    const int row_pos = (multi > 0 || speculate > 0) ? 1 : 0;   // rows are positions of one sequence
    const size_t replay_qkv = (size_t)batch * m.qkv_out;
    const size_t replay_gate = (size_t)batch * m.sss_dt;
    const size_t replay_layer = 2 * (replay_qkv + replay_gate);
    float* replay_inputs = nullptr;
    if (speculate > 0 && m.n_recr) {
        check(cudaMalloc((void**)&replay_inputs, replay_bytes), "recurrent replay inputs");
    }
    const size_t trial_ssm_layer = m.sss_dt * Sv * Sv;
    const size_t trial_conv_layer = m.qkv_out * (m.sss_conv - 1);
    float* trial_ssm = session_draft_hints || fuse_prefill_tail ? replay_inputs : nullptr;
    float* trial_conv = trial_ssm ? trial_ssm + m.n_recr * trial_ssm_layer : nullptr;
    int *tree_depth = nullptr, *tree_paths = nullptr, *tree_leaves = nullptr;
    int *tree_lengths = nullptr, *tree_owners = nullptr;
    int tree_leaf_count = 0;
    if (eagle_on || mtp_tree) {
        check(cudaMalloc(&tree_depth, B * sizeof(int)), "tree depths");
        check(cudaMalloc(&tree_paths, B * B * sizeof(int)), "tree paths");
        check(cudaMalloc(&tree_leaves, B * B * sizeof(int)), "tree leaf paths");
        check(cudaMalloc(&tree_lengths, B * sizeof(int)), "tree leaf lengths");
        check(cudaMalloc(&tree_owners, B * sizeof(int)), "tree output owners");
    }
    if (mtp_tree) {
        check(cudaMalloc(&mtp_tree_hidden, (size_t)batch * m.n_embd * sizeof(float)), "MTP tree hidden states");
        check(cudaMalloc(&mtp_tree_depth, mtp_group * sizeof(int)), "MTP draft depth");
        check(cudaMalloc(&mtp_tree_path, (size_t)mtp_group * batch * sizeof(int)), "MTP draft path");
        if (mtp_group > 1) check(cudaMalloc(&mtp_tree_inputs, (size_t)mtp_group * m.n_embd * sizeof(float)), "MTP branch inputs");
    }
    const bool reuse_quant = packed_verify;
    const bool fuse_quant = reuse_quant && !dbg_dump;
    const bool defer_recurrent = replay_inputs && Sv == 128;
    struct PrefillDraft {
        long rows = 0;
        long split = 0;
        long checkpoint_row = 0;
    };
    auto tier_commit = [&](const std::vector<long>& ids, long end) {
        if (!tier_kv || end <= tier_packed) return;
        if (tier_cache.pooling) {
            const long flag_start = tier_packed / kTierOldGroup * kTierOldGroup;
            for (long pos = flag_start; pos < end; ++pos) {
                const bool protect = pos < 32 || tier_structural[ids[pos]];
                if (pos % kTierOldGroup == 0) tier_flags[pos] = 0;
                if (pos % kTierMidGroup == 0) tier_flags[pos + 1] = 0;
                if (protect) {
                    tier_flags[pos / kTierOldGroup * kTierOldGroup] = 1;
                    tier_flags[pos / kTierMidGroup * kTierMidGroup + 1] = 1;
                }
            }
            const long flag_end = std::min(ctx, (long)align_up(end, kTierOldGroup));
            check(cudaMemcpyAsync(tier_protection + flag_start, tier_flags.data() + flag_start,
                flag_end - flag_start, cudaMemcpyHostToDevice, stc), "TTKV retention mask");
        }
        const int heads = (int)(m.n_attn * m.n_kv);
        for (int width : {1, kTierMidGroup, kTierOldGroup}) {
            const long first = tier_packed / width, count = end / width - first;
            if (count && (width == 1 || tier_cache.pooling))
                k_tier_pack<<<dim3(heads, count), kTierHead, 0, stc>>>(tier_cache, first, width, (int)m.n_rot, (float)m.rope_base);
        }
        tier_packed = end;
    };
    // E187 negative: pre-draft prefetch of the next pass's first copies (position-keyed
    // into the slot ring, skip-on-consume in mat) measured NET NEGATIVE — token-exact
    // but −0% at hot=0 (draft readbacks simply contend for the saturated link) and
    // −4% at hot=4378 (verify +9 ms, mtp +7.5 ms; link had 32 ms of slack, so the loss
    // is ordering/L2-eviction interference, not bandwidth). Reverted; patch and raw
    // runs preserved in results/e187-prefetch-negative-20261113/. Do not re-attempt
    // without a mechanism for readback isolation.
    auto run_token = [&](const std::vector<long>& toks, long B, int pass, double* tok_s_out, uint64_t* cold_out,
                         bool record_recurrent = false, bool tree = false, long head_row = -1, bool capture = false,
                         long physical_rows = 0, PrefillDraft draft = PrefillDraft{}, double* h2d_ms_out = nullptr) {
        if (tier_kv) tier_written = std::max(tier_written, cur_pos + B);
        g_batch = mat_width(physical_rows ? physical_rows : B);
        const long prompt_rows = B - draft.rows;
        const bool grouped = prefill_chunks > 1 && B > kPrefillChunk && head_row == -2;
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
            std::vector<uint8_t> repacked, encoded;
            for (size_t k = (size_t)t; k < cold_list.size(); k += (size_t)readers) {
                const int idx = cold_list[k]; Item& it = items[idx];
                const bool in_pin = (uint64_t)it.res_off + it.bytes <= pin_bytes;
                if (pass > 0 && in_pin && cached[idx]) continue;   // keep position, no fill
                while (released.load() + kStage <= k && !fail.load()) std::this_thread::yield();
                if (fail.load()) return;
                const int b = (int)(k % kStage);
                if (it.tier == kStaged && cached[idx]) {
                    // E043 fix: pinned-head items live in pinned_cache, NOT staged_cache —
                    // the old unconditional staged_cache read returned never-written bytes
                    // for them. Only triggered on a pass-0 re-entry with populated caches
                    // (spec cycle 0): pass>=1 reads pinned items via direct_pin instead.
                    memcpy(stage[b] + it.head, (in_pin ? pinned_cache : staged_cache) + it.res_off, it.transfer);
                } else {
                    const ssize_t got = pread(fd, stage[b], it.read_len, (off_t)it.aligned_off);
                    if (got < (ssize_t)(it.head + it.bytes)) { fprintf(stderr, "pread %zd < %llu\n", got, (unsigned long long)(it.head + it.bytes)); fail = true; return; }
                    if (it.packed) {
                        repacked.resize(it.bytes);
                        pack_q4_tiles(repacked.data(), stage[b] + it.head, it.K, it.rows);
                        const uint8_t* packed = repacked.data();
                        if (compress_packed && it.tier == kStaged) {
                            it.transfer = encode_scale_panels(encoded, packed, it.bytes);
                            if (it.transfer < it.bytes) {
                                if (!verify_scale_panels(encoded.data(), packed, it.bytes)) {
                                    fprintf(stderr, "compressed scale round trip mismatch\n");
                                    fail = true;
                                    return;
                                }
                                packed = encoded.data();
                            }
                        }
                        memcpy(stage[b] + it.head, packed, it.transfer);
                    }
                    if (it.tier == kStaged) { memcpy((in_pin ? pinned_cache : staged_cache) + it.res_off, stage[b] + it.head, it.transfer); cached[idx] = 1; }
                }
                ready[b].store((long)k + 1);
            }
        });
        size_t cold_k = 0;
        double h2d_ms = 0.0;
        std::vector<int> pending;
        auto S = [&](const std::string& tn) -> const float* { return (const float*)(small_dev + m.small_off.at(tn)); };
        auto mat = [&](const std::string& tn, const float* xin, long K, float* out, long out_row0, long ystride,
                       bool reuse = false) {
            const long nb = K / 32;
            // batched GEMV writes y[v*ystride + row]; consumers index dense [B][N], so the
            // stride must be this tensor's own row count — NOT the max_rows buffer pitch
            ystride = m.by_name.at(tn)->ne[1];
            if (!reuse || !reuse_quant) {
                // E193: a previous split tensor's D2H may still be reading xq — the
                // event has usually long fired (the D2H raced ahead during the prior
                // tensor's GEMVs), so this wait is free in the steady state.
                if (cpu_xq_guard) check(cudaStreamWaitEvent(stc, cpu_xq_guard, 0), "cpu xq guard");
                cpu_xq_guard = nullptr;
                k_quant_x<<<dim3((unsigned)((nb * B + 127) / 128)), 128, 0, stc>>>(xin, nb * B, xq, xmeta);
            }
            if (l2_win && !reuse) {
                // E190/A6: persist this tensor's quantized-activation slice for the
                // duration of its GEMV tiles. Weight bytes stream through the remaining
                // L2 (missProp=Streaming). Overflowing slices scale hitRatio to the cap
                // (partial persistence of e.g. ffn_down's 5 MB slice at 256 rows).
                const uint64_t span = (uint64_t)B * K + (uint64_t)B * K / 32 * sizeof(int2);
                cudaStreamAttrValue w{};
                w.accessPolicyWindow.base_ptr = activation_buf;
                w.accessPolicyWindow.num_bytes = (size_t)std::min<uint64_t>(span, activation_bytes);
                w.accessPolicyWindow.hitRatio = std::min(1.0f,
                    (float)l2prop.persistingL2CacheMaxSize / (float)std::max<uint64_t>(span, 1));
                w.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                w.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                check(cudaStreamSetAttribute(stc, cudaStreamAttributeAccessPolicyWindow, &w), "per-tensor L2 window");
            }
            // E193: per-tensor CPU/GPU item split (item granularity, row-suffix to CPU)
            long cpu_from = -1, cpu_row_start = 0;
            if (cpu_worker_on && cpu_rows_frac > 0.001f && !reuse && !dbg_dump
                && pass > 0 && B <= standard_rows) {
                uint64_t tbytes = 0;
                for (int idx : tensor_items.at(tn)) tbytes += items[idx].bytes;
                const uint64_t cut = (uint64_t)((double)tbytes * (1.0 - (double)cpu_rows_frac));
                uint64_t cacc = 0;
                for (size_t ii = 0; ii < tensor_items.at(tn).size(); ++ii) {
                    const Item& it = items[tensor_items.at(tn)[ii]];
                    if (cacc >= cut) {
                        bool ok = true;
                        for (size_t jj = ii; jj < tensor_items.at(tn).size(); ++jj) {
                            const Item& cj = items[tensor_items.at(tn)[jj]];
                            if (!cj.packed || cj.rows % 16
                                || !(pass > 0 && (uint64_t)cj.res_off + cj.bytes <= pin_bytes
                                     && cached[tensor_items.at(tn)[jj]])) { ok = false; break; }
                        }
                        if (ok && ii < tensor_items.at(tn).size()) {   // E193: cpu_from must index an EXISTING item
                            cpu_from = (long)ii; cpu_row_start = it.row0;
                        }
                        break;
                    }
                    cacc += it.bytes;
                }
            }
            if (cpu_from >= 0) {
                if (getenv("GUM_E193_DEBUG")) E193LOG("split tn=%s from=%ld start=%ld\n", tn.c_str(), cpu_from, cpu_row_start);
                cpu_ring = (cpu_ring + 1) & 1;
                check(cudaEventSynchronize(cpu_oh[cpu_ring]), "cpu out ring sync");
                check(cudaEventRecord(cpu_qev[cpu_ring], stc), "cpu quant ev");
                check(cudaStreamWaitEvent(st3, cpu_qev[cpu_ring], 0), "cpu quant wait");
                check(cudaMemcpyAsync(cpu_xq_stage[cpu_ring], xq, (size_t)B * K, cudaMemcpyDeviceToHost, st3), "cpu xq d2h");
                check(cudaMemcpyAsync(cpu_meta_stage[cpu_ring], xmeta, (size_t)B * (K / 32) * sizeof(int2), cudaMemcpyDeviceToHost, st3), "cpu meta d2h");
                check(cudaEventRecord(cpu_qh[cpu_ring], st3), "cpu quant d2h ev");
                // Airtight ordering: EVERY later stc op (k_rms_quant, store_quantized_warp,
                // next k_quant_x...) is blocked until the D2H snapshot of xq has landed.
                // The wait is ~0.1 ms (the D2H fires right after the quant kernel); the
                // worker's compute still overlaps the tensor's GEMVs.
                check(cudaStreamWaitEvent(stc, cpu_qh[cpu_ring], 0), "stc wait xq d2h");
                const int first_idx = tensor_items.at(tn)[cpu_from];
                cpu_job.w = pinned_cache + items[first_idx].res_off;
                cpu_job.K = (int)K;
                cpu_job.rows = (int)(ystride - cpu_row_start);
                cpu_job.B = (int)B;
                cpu_job.N = ystride;
                cpu_job.row_start = cpu_row_start;
                cpu_job.xq = cpu_xq_stage[cpu_ring];
                cpu_job.xmeta = cpu_meta_stage[cpu_ring];
                cpu_job.out = cpu_out_stage[cpu_ring] + cpu_row_start;   // E193: kernel writes item-relative rows; stage is read tensor-relative
                cpu_job.qev = cpu_qh[cpu_ring];
                cpu_job_done.store(0, std::memory_order_release);
                cpu_job_ready.store(1, std::memory_order_release);
            }
            for (size_t tii = 0; tii < tensor_items.at(tn).size(); ++tii) {
                const int idx = tensor_items.at(tn)[tii];
                Item& it = items[idx];
                const bool cpu_owned = cpu_from >= 0 && (long)tii >= cpu_from;
                const bool cpu_shadow = cpu_owned && cpu_verify;
                if (cpu_owned && !cpu_verify) { ++cold_k; continue; }   // E193: worker computes it; no slot, no DMA — but cold_k MUST advance: it is the cold_list position the readers' ready[]/back-pressure index by
                auto multiply = [&](const uint8_t* weights) {
                    float* output = out + out_row0 + it.row0;
                    if (draft.rows && it.packed && B > kPrefillChunk) {
                        // Packed groups use the same per-row FP32 reduction
                        // for every width, including the final prompt slice.
                        if (it.transfer < it.bytes)
                            launch_packed_q4<true>(weights, K, it.rows, xq, xmeta, output, ystride, stc, B, true);
                        else
                            launch_packed_q4(weights, K, it.rows, xq, xmeta, output, ystride, stc, B, true);
                    } else if (draft.rows || draft.checkpoint_row) {
                        // After a reused prefix the first canonical chunk can
                        // start partway through, with only its suffix in xq.
                        auto slice = [&](long row, long width) {
                            g_batch = mat_width(width);
                            const int8_t* q = xq + row * K;
                            const int2* meta = xmeta + row * nb;
                            float* dst = output + row * ystride;
                            if (it.transfer < it.bytes)
                                launch_packed_q4<true>(weights, K, it.rows, q, meta, dst, ystride, stc);
                            else if (it.packed)
                                launch_packed_q4(weights, K, it.rows, q, meta, dst, ystride, stc);
                            else
                                launch_mat_q4(weights, K, it.rows, q, meta, dst, ystride, stc);
                        };
                        const long split = draft.checkpoint_row ? draft.checkpoint_row : draft.split;
                        for (long row = 0; row < split; ) {
                            slice(row, kPrefillChunk);
                            row += draft.checkpoint_row ? kPrefillChunk - (cur_pos + row) % kPrefillChunk : kPrefillChunk;
                        }
                        slice(split, physical_rows);
                        for (long row = prompt_rows; row < B; row += kPrefillChunk)
                            slice(row, std::min(kPrefillChunk, B - row));
                    } else if (it.transfer < it.bytes)
                        launch_packed_q4<true>(weights, it.K, it.rows, xq, xmeta, output, ystride, stc, B, grouped);
                    else if (it.packed)
                        launch_packed_q4(weights, it.K, it.rows, xq, xmeta, output, ystride, stc, B, grouped);
                    else if (grouped) {
                        g_batch = kPrefillChunk;
                        for (long row = 0; row < B; row += kPrefillChunk)
                            launch_mat_q4(weights, it.K, it.rows, xq + row * it.K, xmeta + row * nb,
                                output + row * ystride, ystride, stc);
                        g_batch = mat_width(B);
                    } else launch_mat_q4(weights, it.K, it.rows, xq, xmeta, output, ystride, stc);
                };
                if (it.tier == kHot) {
                    multiply(hot + it.res_off);
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
                if (measure_h2d) check(cudaEventRecord(h2d_start[cold_k % kStage], st), "H2D timing start");
                check(cudaMemcpyAsync(dst, csrc, it.transfer, cudaMemcpyHostToDevice, st), "copy");
                check(cudaEventRecord(ev[cold_k % kStage], st), "ev");
                check(cudaStreamWaitEvent(stc, ev[cold_k % kStage], 0), "wait copy");
                multiply(dst);
                check(cudaEventRecord(evc[cold_k % kSlots], stc), "evc");
                pending.push_back((int)cold_k);
                if (pending.size() >= kWindow) {   // E047 deeper window; E186 sweeps it (must be < kStage)
                    const int o = pending.front(); pending.erase(pending.begin());
                    check(cudaEventSynchronize(ev[o % kStage]), "ev sync");
                    if (measure_h2d) {
                        float ms = 0.0f;
                        check(cudaEventElapsedTime(&ms, h2d_start[o % kStage], ev[o % kStage]), "H2D elapsed");
                        h2d_ms += ms;
                    }
                    released.store((size_t)o + 1);
                }
                ++cold_k;
            }
                    if (cpu_from >= 0) {
                if (getenv("GUM_E193_DEBUG")) E193LOG("main waiting done tn=%s rows=%ld\n", tn.c_str(), ystride - cpu_row_start);
                while (!cpu_job_done.load(std::memory_order_acquire)) std::this_thread::yield();
                if (getenv("GUM_E193_DEBUG")) E193LOG("main got done tn=%s\n", tn.c_str());
                if (cpu_verify) {
                    const long w = ystride - cpu_row_start;
                    std::vector<float> g((size_t)B * w);
                    check(cudaMemcpy2DAsync(g.data(), w * 4, out + out_row0 + cpu_row_start, ystride * 4,
                                            (size_t)w * 4, B, cudaMemcpyDeviceToHost, st3), "cpu verify d2h");
                    check(cudaEventRecord(cpu_oh[cpu_ring], st3), "cpu oh v");
                    check(cudaEventSynchronize(cpu_oh[cpu_ring]), "cpu verify sync");
                    double maxd = 0; long bad = 0; int vbad[64] = {0}; long rbad[2048] = {0};
                    long firstv = -1, firstr = -1;
                    for (long v = 0; v < B; ++v)
                        for (long r = 0; r < w; ++r) {
                            const double dv = std::fabs((double)g[(size_t)v * w + r]
                                - (double)cpu_out_stage[cpu_ring][(size_t)v * ystride + cpu_row_start + r]);
                            if (dv > maxd) maxd = dv;
                            if (dv > 1e-2 * std::max(1.0, std::fabs((double)g[(size_t)v * w + r]))) {
                                ++bad; ++vbad[v]; ++rbad[r];
                                if (firstv < 0) { firstv = v; firstr = r; }
                            }
                        }
                    {   // scalar reference for element (v=0, suffix row 0) from the worker's own snapshot
                        const int vv = 0;
                        const int blocks = cpu_job.K / 32;
                        double acc_s = 0;
                        for (int b = 0; b < blocks; ++b) {
                            const uint8_t* tb = cpu_job.w + (size_t)b * 288;
                            const uint8_t* nib = tb;   // suffix row 0 -> tile row 0
                            float dw; uint16_t hb; memcpy(&hb, tb + 256, 2); dw = cpu_fp16_to_f32(hb);
                            const int2 mx = cpu_job.xmeta[(size_t)vv * blocks + b];
                            const float dx = *(const float*)&mx.x;
                            const int8_t* ax = cpu_job.xq + (size_t)vv * cpu_job.K + b * 32;
                            int sref = 0;
                            for (int j = 0; j < 16; ++j) {
                                sref += ((int)(nib[j] & 0xF) - 8) * ax[j];        // weights 0-15 = low nibbles
                                sref += ((int)(nib[j] >> 4) - 8) * ax[16 + j];    // weights 16-31 = high nibbles
                            }
                            acc_s += (double)dw * dx * (double)(sref - 8 * mx.y);
                        }
                        fprintf(stderr, "e193 scalar,gpu=%.9g simd=%.9g scalar=%.9g\n",
                                (double)g[0], (double)cpu_out_stage[cpu_ring][cpu_row_start], acc_s);
                    }
                    fprintf(stderr, "e193 verify,tn=%s,w=%ld,maxdiff=%.3e,bad=%ld,firstv=%ld,firstr=%ld,rowsbad=", tn.c_str(), w, maxd, bad, firstv, firstr);
                    for (int v = 0; v < B; ++v) fprintf(stderr, "%d", vbad[v] ? 1 : 0);
                    fprintf(stderr, "\n");
                } else {
                    check(cudaMemcpy2DAsync(out + out_row0 + cpu_row_start, ystride * 4,
                                            cpu_out_stage[cpu_ring] + cpu_row_start, ystride * 4,
                                            (size_t)(ystride - cpu_row_start) * 4, B,
                                            cudaMemcpyHostToDevice, st3), "cpu partials h2d");
                    check(cudaEventRecord(cpu_oh[cpu_ring], st3), "cpu oh");
                    check(cudaStreamWaitEvent(stc, cpu_oh[cpu_ring], 0), "stc wait cpu partials");
                }
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
            if (fuse_quant) k_rms_quant<<<(unsigned)B, 256, 0, stc>>>(xq, xmeta, x,
                S(L + ".attn_norm.weight"), (int)m.n_embd, (float)m.eps);
            else k_rms_norm<<<grid1, 256, 0, stc>>>(t_norm, x, S(L + ".attn_norm.weight"), (int)m.n_embd, (float)m.eps);
            if (dbg_dump && il <= 1 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("emb" + std::to_string(il)).c_str(), x, m.n_embd); dbg(("norm" + std::to_string(il)).c_str(), t_norm, m.n_embd); }
            if (A == "qwen35" && m.is_recr[il]) {
                // gated delta net path (linear attention with persistent state)
                mat(L + ".attn_qkv.weight", t_norm, m.n_embd, yA, 0, max_rows, fuse_quant);
                mat(L + ".attn_gate.weight", t_norm, m.n_embd, yB, 0, max_rows, true);
                mat(L + ".ssm_beta.weight", t_norm, m.n_embd, t_g48a, 0, max_rows, true);
                if (dbg_dump && il == 0 && pass == 0 && B > 1) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg2("betaraw", t_g48a, m.sss_dt, 48, 1); dbg2("betaraw0", t_g48a, m.sss_dt, 48, 0); }
                mat(L + ".ssm_alpha.weight", t_norm, m.n_embd, t_g48b, 0, max_rows, true);
                k_sigmoid<<<(unsigned)((m.sss_dt * B + 63) / 64), 64, 0, stc>>>(t_g48a, t_g48a, (int)(m.sss_dt * B));
                k_gate_ssm<<<(unsigned)((m.sss_dt * B + 63) / 64), 64, 0, stc>>>(t_v, t_g48b,
                    S(L + ".ssm_dt.bias"), S(L + ".ssm_a"), (int)m.sss_dt, (int)B);
                const long ri = il - (il + 1) / 4;   // recurrent-layer index (pattern 1,1,1,0)
                const long conv_seq_stride = m.n_recr * m.qkv_out * (m.sss_conv - 1);
                const long ssm_seq_stride = m.n_recr * m.sss_dt * Sv * Sv;
                float* saved = record_recurrent && replay_inputs ? replay_inputs + ri * replay_layer : nullptr;
                if (saved) check(cudaMemcpyAsync(saved, yA, (size_t)B * m.qkv_out * sizeof(float), cudaMemcpyDeviceToDevice, stc), "save conv input");
                if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg("qkv0", yA, m.qkv_out); dbg("zraw0", yB, m.sss_inner); dbg("beta0", t_g48a, m.sss_dt); dbg("alpharaw0", t_g48b, m.sss_dt); if (B > 1) { dbg2("qkv0", yA, m.qkv_out, m.qkv_out, 1); dbg2("beta0", t_g48a, m.sss_dt, 64, 1); } }
                if (tree)
                    k_conv_tree<<<dim3((unsigned)((m.qkv_out + 255) / 256), (unsigned)B), 256, 0, stc>>>(
                        conv_dev + ri * m.qkv_out * (m.sss_conv - 1), yA, S(L + ".ssm_conv1d.weight"),
                        t_conv, (int)m.qkv_out, tree_depth, tree_paths, (int)batch);
                else if (row_pos)
                    k_conv1d_silu_multi<<<(unsigned)((m.qkv_out + 255) / 256), 256, 0, stc>>>(
                        conv_dev + ri * trial_conv_layer, yA, S(L + ".ssm_conv1d.weight"),
                        t_conv, (int)m.qkv_out, (int)B, !saved || !defer_recurrent,
                        (draft.rows || draft.checkpoint_row) ? trial_conv + ri * trial_conv_layer : nullptr,
                        (int)(draft.checkpoint_row ? draft.checkpoint_row : prompt_rows), draft.checkpoint_row != 0);
                else
                    k_conv1d_silu<<<dim3((unsigned)((m.qkv_out + 255) / 256), (unsigned)B), 256, 0, stc>>>(conv_dev + (size_t)ri * m.qkv_out * (m.sss_conv - 1), yA, S(L + ".ssm_conv1d.weight"), t_conv, (int)m.qkv_out, conv_seq_stride);
                if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg("convout0", t_conv, m.qkv_out); if (B > 1) dbg2("convout0", t_conv, m.qkv_out, m.qkv_out, 1); }
                const long Sdim = m.sss_state;
                const long conv_slice_stride = A == "qwen35" ? m.qkv_out : 0;
                k_l2_heads<<<dim3((unsigned)m.sss_group, (unsigned)B), 128, 0, stc>>>(t_conv, (int)m.sss_group, (int)Sdim, (float)m.eps, conv_slice_stride);
                // K lives at [Sdim*group, 2*Sdim*group) — V (at 2*Sdim*group) must NOT be normalized
                k_l2_heads<<<dim3((unsigned)m.sss_group, (unsigned)B), 128, 0, stc>>>(t_conv + Sdim * m.sss_group, (int)m.sss_group, (int)Sdim, (float)m.eps, conv_slice_stride);
                if (saved) {
                    check(cudaMemcpyAsync(saved + replay_qkv, t_conv, (size_t)B * m.qkv_out * sizeof(float), cudaMemcpyDeviceToDevice, stc), "save delta inputs");
                    check(cudaMemcpyAsync(saved + 2 * replay_qkv, t_v, (size_t)B * m.sss_dt * sizeof(float), cudaMemcpyDeviceToDevice, stc), "save delta decay");
                    check(cudaMemcpyAsync(saved + 2 * replay_qkv + replay_gate, t_g48a, (size_t)B * m.sss_dt * sizeof(float), cudaMemcpyDeviceToDevice, stc), "save delta beta");
                }
                if (tree)
                    k_delta_net_128<true, true><<<dim3((unsigned)m.sss_dt, 4, tree_leaf_count), 32, 0, stc>>>(
                        ssm_dev + (size_t)ri * m.sss_dt * Sv * Sv, t_conv, t_v, t_g48a, t_dn,
                        (int)m.sss_dt, (int)m.sss_group, dn_scale, m.qkv_out, m.sss_inner, (int)B,
                        tree_leaves, tree_lengths, tree_owners, (int)batch);
                else if (row_pos)
                    launch_delta_multi(ssm_dev + (size_t)ri * m.sss_dt * Sv * Sv, t_conv, t_v, t_g48a, t_dn,
                                       (int)Sdim, (int)m.sss_dt, (int)m.sss_group, dn_scale, m.qkv_out,
                                       m.sss_inner, (int)B, stc, nullptr, !saved || !defer_recurrent,
                                       (draft.rows || draft.checkpoint_row) ? trial_ssm + ri * trial_ssm_layer : nullptr,
                                       (int)(draft.checkpoint_row ? draft.checkpoint_row : prompt_rows), draft.checkpoint_row != 0);
                else
                    k_delta_net<<<dim3((unsigned)m.sss_dt, (unsigned)B), 128, 0, stc>>>(ssm_dev + (size_t)ri * m.sss_dt * Sv * Sv, t_conv, t_conv + Sdim * m.sss_group,
                                                                 t_conv + 2 * Sdim * m.sss_group, t_v, t_g48a, t_dn, (int)Sdim, (int)m.sss_dt, (int)m.sss_group, dn_scale, ssm_seq_stride, (long)m.qkv_out);
                if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg("dnout0", t_dn, m.sss_inner); if (B > 1) { dbg2("dnout0", t_dn, m.sss_inner, m.sss_inner, 1); dbg2("vsl", t_conv + 2 * m.sss_state * m.sss_group, 8, m.qkv_out, 1); dbg2("vsl0", t_conv + 2 * m.sss_state * m.sss_group, 8, m.qkv_out, 0); dbg2("gsl", t_v, 8, 4096, 1); dbg2("gsl0", t_v, 8, 4096, 0); } }
                k_gated_norm<<<dim3((unsigned)m.sss_dt, (unsigned)B), 128, 0, stc>>>(t_att, t_dn, S(L + ".ssm_norm.weight"), yB, (int)m.sss_dt, (int)Sv, (float)m.eps);
                if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg("gatednorm0", t_att, m.sss_inner); if (B > 1) dbg2("gatednorm0", t_att, m.sss_inner, m.sss_inner, 1); }
                mat(L + ".ssm_out.weight", t_att, m.sss_inner, yA, 0, max_rows);
            } else {
                // full attention path with KV cache
                mat(L + ".attn_q.weight", t_norm, m.n_embd, yA, 0, max_rows, fuse_quant);
                mat(L + ".attn_k.weight", t_norm, m.n_embd, yB, 0, max_rows, true);
                mat(L + ".attn_v.weight", t_norm, m.n_embd, t_v, 0, max_rows, true);
                if (dbg_dump && il <= 1 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("v" + std::to_string(il)).c_str(), t_v, nh_k * hd); }
                const int q_stride = A == "qwen35" ? 2 * (int)hd : (int)hd;
                k_rms_heads<<<dim3((unsigned)nh_q, (unsigned)B), 128, 0, stc>>>(yA, S(L + ".attn_q_norm.weight"), (int)nh_q, (int)hd, (float)m.eps, q_stride);
                k_rms_heads<<<dim3((unsigned)nh_k, (unsigned)B), 128, 0, stc>>>(yB, S(L + ".attn_k_norm.weight"), (int)nh_k, (int)hd, (float)m.eps, (int)hd);
                // ggml uses NEOX half-split pairing for NORMAL?no: NORM is adjacent; NEOX/MROPE/IMROPE
                // are half-split (ops.cpp rope switch). qwen35 = IMROPE = half-split; for text the
                // mrope section positions are all equal, so it is plain half-split RoPE.
                const int rope_style = 0;
                k_rope<<<dim3((unsigned)(nh_q + nh_k), (unsigned)B), 128, 0, stc>>>(yA, yB, (int)cur_pos, (int)nh_q, (int)nh_k, (int)hd, (int)m.n_rot, (float)m.rope_base, q_stride, rope_style, row_pos, tree ? tree_depth : nullptr);
                const long li = A == "qwen35" ? il / 4 : il;
                if (tier_kv) {
                    const auto layer_cache = tier_cache.layer((int)nh_k, (int)li);
                    const int splits = (int)((cur_pos + B + kTierSplit - 1) / kTierSplit);
                    k_tier_append<<<dim3(nh_k, B), kTierHead, 0, stc>>>(layer_cache, yB, t_v, cur_pos, nh_k);
                    k_tier_query<<<dim3(nh_q, B), kTierHead, 0, stc>>>(tier_queries, yA, nh_q, q_stride);
                    for (int first_row = 0; first_row < B; first_row += tier_batch) {
                        const int rows = std::min<long>(tier_batch, B - first_row);
                        k_tier_attention<<<dim3(nh_q, rows, splits), kTierHead, 0, stc>>>(
                            t_att, tier_partial, layer_cache, yA, tier_queries, tier_protection, cur_pos,
                            nh_q, nh_k, q_stride, tree ? tree_depth : nullptr,
                            tree ? tree_paths : nullptr, batch, splits, first_row, attn_scale);
                        if (splits > 1)
                            k_tier_merge<<<nh_q * rows, kTierHead, 0, stc>>>(
                                t_att + (size_t)first_row * nh_q * hd, tier_partial, splits);
                    }
                } else {
                    // batch mode: one slice per sequence; multi/verify mode: rows are positions of
                    // ONE sequence and must share slice 0 (full-history causal attention)
                    const long kv_seq_stride = row_pos ? 0 : (long)m.n_attn * 2 * m.n_kv * ctx * m.head_v;   // FLOATS
                    float* Kc = (float*)((uint8_t*)kv_dev + (size_t)li * 2 * kv_per_layer);
                    float* Vc = (float*)((uint8_t*)kv_dev + (size_t)li * 2 * kv_per_layer + kv_per_layer);
                    // NOTE: appends must run on the compute stream — they read yB/t_v produced
                    // by GEMVs on stc (a copy-stream race here corrupted every layer >= 1)
                    k_append_kv<<<dim3((unsigned)((nh_k * hd + 255) / 256), (unsigned)B), 256, 0, stc>>>(
                        Kc, Vc, yB, t_v, (int)cur_pos, (int)ctx, (int)nh_k, (int)hd, kv_seq_stride, row_pos);
                    if (dbg_dump && il <= 1 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("vc" + std::to_string(il)).c_str(), Vc, nh_k * hd); }
                    k_attn_decode<<<dim3((unsigned)nh_q, (unsigned)B), 256, (unsigned)((cur_pos + (row_pos ? B : 1)) * 4), stc>>>(t_att, Kc, Vc, yA, (int)cur_pos, (int)ctx, (int)nh_q, (int)nh_k, (int)hd, attn_scale, q_stride, kv_seq_stride, row_pos,
                        tree ? tree_depth : nullptr, tree ? tree_paths : nullptr, (int)batch);
                    if (dbg_dump && il == 0 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("att" + std::to_string(il)).c_str(), t_att, nh_q * hd); if (B > 1) { dbg2(("att" + std::to_string(il)).c_str(), t_att, nh_q * hd, nh_q * hd, 1); dbg2(("vv" + std::to_string(il)).c_str(), t_v, nh_k * hd, nh_k * hd, 1); } }
                }
                if (A == "qwen35") k_attn_gate_apply<<<dim3((unsigned)((nh_q * hd + 255) / 256), (unsigned)B), 256, 0, stc>>>(t_att, yA, (int)nh_q, (int)hd);
                mat(L + ".attn_output.weight", t_att, nh_q * hd, yA, 0, max_rows);
                if (dbg_dump && il == 0 && pass == 0 && B > 1) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg2("wo0", yA, m.n_embd, m.n_embd, 1); dbg2("xpre0", x, m.n_embd, m.n_embd, 1); }
            }
            k_add<<<(unsigned)((m.n_embd * B + 255) / 256), 256, 0, stc>>>(x, yA, (int)(m.n_embd * B));
            if (dbg_dump && il == 0 && pass == 0 && B > 1) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg2("res0", x, m.n_embd, m.n_embd, 1); }
            if (dbg_dump && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("attn_out" + std::to_string(il)).c_str(), x, m.n_embd); if (B > 1) for (long vv = 1; vv < std::min<long>(B, 2); ++vv) dbg2(("attnout" + std::to_string(il)).c_str(), x, m.n_embd, m.n_embd, vv); }
            const std::string ln2 = A == "qwen35" ? L + ".post_attention_norm.weight" : L + ".ffn_norm.weight";
            if (fuse_quant) k_rms_quant<<<(unsigned)B, 256, 0, stc>>>(xq, xmeta, x, S(ln2), (int)m.n_embd, (float)m.eps);
            else k_rms_norm<<<grid1, 256, 0, stc>>>(t_norm, x, S(ln2), (int)m.n_embd, (float)m.eps);
            mat(L + ".ffn_gate.weight", t_norm, m.n_embd, yA, 0, max_rows, fuse_quant);
            mat(L + ".ffn_up.weight", t_norm, m.n_embd, yB, 0, max_rows, true);
            if (fuse_quant) k_silu_quant<<<(unsigned)((m.n_ff * B + 255) / 256), 256, 0, stc>>>(xq, xmeta, yA, yB, m.n_ff * B);
            else k_silu_mul<<<(unsigned)((m.n_ff * B + 255) / 256), 256, 0, stc>>>(t_ffnh, yA, yB, (int)(m.n_ff * B));
            mat(L + ".ffn_down.weight", t_ffnh, m.n_ff, yA, 0, max_rows, fuse_quant);
            k_add<<<(unsigned)((m.n_embd * B + 255) / 256), 256, 0, stc>>>(x, yA, (int)(m.n_embd * B));
            if (capture && capture_slots[il] >= 0) {
                check(cudaMemcpyAsync(capture_dev + capture_slots[il] * capture_row_floats, x,
                    capture_row_floats * sizeof(float), cudaMemcpyDeviceToDevice, stc), "capture mid");
            }
            if (eagle_on) eagle.tap((int)il, x, (int)B);
            if (dbg_dump && il <= 1 && pass == 0) { check(cudaStreamSynchronize(stc), "dbg sync"); dbg(("ffn_out" + std::to_string(il)).c_str(), x, m.n_embd); }
            if ((il & 7) == 0) safety("token");
        }
        if (capture)
            check(cudaMemcpyAsync(capture_dev + capture_mid * capture_row_floats, x,
                capture_row_floats * sizeof(float), cudaMemcpyDeviceToDevice, stc), "capture final pre-norm");
        // Prompt chunks only need the last row's prediction at the handoff.
        // The MTP cache still consumes every final-normalized target row.
        if (mtp_on || head_row == -1)
            k_rms_norm<<<grid1, 256, 0, stc>>>(t_norm, x, S("output_norm.weight"), (int)m.n_embd, (float)m.eps);
        else if (head_row >= 0)
            k_rms_norm<<<1, 256, 0, stc>>>(t_norm + head_row * m.n_embd, x + head_row * m.n_embd,
                S("output_norm.weight"), (int)m.n_embd, (float)m.eps);
        auto read_target_maxima = [&](long start, long rows) {
            k_logit_tile_max<<<dim3(logit_tiles, (unsigned)rows), 256, 0, stc>>>(logits, (int)m.n_vocab, tile_maxima);
            k_logit_row_max<<<(unsigned)rows, 256, 0, stc>>>(tile_maxima, logit_tiles, row_maxima);
            check(cudaMemcpyAsync(target_maxima.data() + start, row_maxima,
                rows * sizeof(LogitMax), cudaMemcpyDeviceToHost, stc), "target argmax d2h");
        };
        auto score_target_rows = [&](long start, long rows) {
            const float* input = t_norm + start * m.n_embd;
            const long blocks = m.n_embd / 32 * rows;
            g_batch = mat_width(rows);
            k_quant_x<<<(unsigned)((blocks + 127) / 128), 128, 0, stc>>>(input, blocks, xq, xmeta);
            launch_mat_q4(output_device, m.n_embd, m.n_vocab, xq, xmeta, logits, m.n_vocab, stc);
            if (compact_target_logits) read_target_maxima(start, rows);
        };
        if (share_output && head_row != -2) {
            score_target_rows(std::max(0L, head_row), head_row >= 0 ? 1 : B);
            if (draft.rows)
                for (long row = prompt_rows; row < B; row += speculate)
                    score_target_rows(row, std::min(speculate, B - row));
        } else if (!share_output) {
            mat("output.weight", t_norm, m.n_embd, logits, 0, m.n_vocab);
            if (compact_target_logits) read_target_maxima(0, B);
        }
        if (capture)
            check(cudaMemcpyAsync(capture_host + 3, capture_dev, capture_payload_floats * sizeof(float),
                cudaMemcpyDeviceToHost, stc), "capture d2h");
        check(cudaGetLastError(), "kernel");
        check(cudaStreamSynchronize(stc), "sync c");
        check(cudaStreamSynchronize(st), "sync copy");
        if (measure_h2d) for (int o : pending) {
            check(cudaEventSynchronize(ev[o % kStage]), "H2D final sync");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, h2d_start[o % kStage], ev[o % kStage]), "H2D final elapsed");
            h2d_ms += ms;
        }
        released.store(cold_list.size() + (size_t)kStage);
        for (auto& th : pool) th.join();
        if (capture) {
            capture_host[0] = (float)cur_pos;
            capture_host[1] = (float)capture_mid;
            capture_host[2] = (float)B;
            const size_t record_floats = capture_payload_floats + 3;
            if (fwrite(capture_host, sizeof(float), record_floats, capture_file) != record_floats) {
                perror(capture_path);
                exit(1);
            }
        }
        if (!compact_target_logits && head_row != -2) {
            const long rows = head_row >= 0 ? 1 : B;
            check(cudaMemcpy(logits_h.data() + std::max(0L, head_row) * m.n_vocab, logits,
                rows * m.n_vocab * sizeof(float), cudaMemcpyDeviceToHost), "logits d2h");
        }
        if (dbg_dump) {
            long t1 = 0;
            for (long j = 1; j < m.n_vocab; ++j) if (logits_h[j] > logits_h[t1]) t1 = j;
            printf("dbg_gpu,logits_top,%ld,%ld:%.4f\n", cur_pos, t1, logits_h[t1]);
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        *tok_s_out = s;
        if (h2d_ms_out) *h2d_ms_out = h2d_ms;
        uint64_t cb = 0; for (int idx : cold_list) cb += items[idx].transfer;
        *cold_out = cb;
        if (fail.load()) exit(1);
        cur_pos += row_pos ? B : 1;   // multi-position verify advances B positions per step
    };

    // ---- SSM state snapshot/rollback for speculative decoding ----
    // Rejected draft positions permanently corrupt the cumulative delta-net state (unlike
    // KV, it cannot be overwritten) — snapshot before each verify cycle, restore on partial
    // accept. ~150 MB D2D, ~1 ms.
    float *ssm_snap = nullptr, *conv_snap = nullptr;
    const uint64_t ssm_bytes = (size_t)m.n_recr * m.sss_dt * Sv * Sv * 4;
    const uint64_t conv_bytes = (size_t)m.n_recr * m.qkv_out * (m.sss_conv - 1) * 4;
    if (speculate > 0 && m.n_recr && !defer_recurrent) {
        check(cudaMalloc((void**)&ssm_snap, ssm_bytes), "ssm snap");
        check(cudaMalloc((void**)&conv_snap, conv_bytes), "conv snap");
    }

    // ---- run: prompt then greedy generation ----
    auto argmax = [&](const float* lg) {
        long best = 0; float bv = lg[0];
        for (long i = 1; i < m.n_vocab; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        return best;
    };
    auto target_argmax = [&](const float* lg) {
        const long id = argmax(lg);
        if (!std::isfinite(lg[id]) || (lg[id] == 0.f &&
            std::all_of(lg, lg + m.n_vocab, [](float value) { return value == 0.f; }))) {
            fprintf(stderr, "invalid target logits\n");
            exit(1);
        }
        return id;
    };
    auto target_id = [&](long row) -> long {
        if (!compact_target_logits) return target_argmax(logits_h.data() + row * m.n_vocab);
        const LogitMax result = target_maxima[row];
        if (result.flags != 1 || result.token < 0 || result.token >= m.n_vocab) {
            fprintf(stderr, "invalid target logits\n");
            exit(1);
        }
        return result.token;
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
    if (!max_plen || ctx <= 0 || max_plen > (size_t)ctx || (size_t)first_gen_tokens > (size_t)ctx - max_plen) {
        fprintf(stderr, "context is too small for the prompt and generation limit\n");
        return 2;
    }
    if (multi > 0 && speculate == 0) {
        // ---- E040: verification-ceiling benchmark ----
        // One sequence; teacher-forced tokens (prompt + --feed-ids, which should be the
        // model's own greedy continuation) processed `multi` positions per step. The cold
        // set streams ONCE per step; every position is "accepted" by construction, so
        // B / T_step is the zero-draft-cost upper bound on speculative throughput.
        std::vector<long> seq = prompts[0];
        const long plen0 = (long)prompts[0].size();   // teacher-forced argmax checks are
                                                      // only valid past the prompt
        seq.insert(seq.end(), feed.begin(), feed.end());
        const long Bp = batch;
        if ((long)seq.size() % Bp) seq.resize(seq.size() / Bp * Bp);
        if ((long)seq.size() < 2 * Bp) { fprintf(stderr, "need at least %ld tokens in prompt+feed\n", 2 * Bp); return 2; }
        printf("verify_mode,positions_per_step=%ld,seq_len=%zu,steps=%zu\n", Bp, seq.size(), seq.size() / Bp);
        long soft_mism = 0, soft_checked = 0;
        double best_step = 1e30, sum_step = 0; long n_steps = 0;
        for (size_t base = 0; base + Bp <= seq.size(); base += Bp) {
            std::vector<long> toks(seq.begin() + base, seq.begin() + base + Bp);
            double s; uint64_t cold;
            run_token(toks, Bp, (int)(base / Bp), &s, &cold);
            // teacher-forced check: row p's argmax should predict seq[base+p+1] — but ONLY
            // where seq[base+p+1] is the model's own greedy output (position >= prompt
            // length); inside the prompt the true next token is not the argmax.
            // E043 fix: the old check compared against seq[base+Bp+p] (wrong index) AND
            // computed gap = lg[expected] - lg[pred], which is <= 0 for an argmax pred —
            // every mismatch was excused as a tie, making the whole soft check vacuous.
            for (long p = 0; p < Bp && base + p + 1 < (long)seq.size(); ++p) {
                if (base + p + 1 < plen0) continue;   // prompt region: argmax not guaranteed
                const long pred = argmax(&logits_h[(size_t)p * m.n_vocab]);
                if (dbg_dump) printf("dbg,e040row,step=%zu,row=%ld,pred=%ld,want=%ld\n", base / Bp, p, pred, seq[base + p + 1]);
                ++soft_checked;
                float rowmax = 0.f;
                for (long j = 0; j < m.n_vocab; ++j) rowmax = fmaxf(rowmax, fabsf(logits_h[(size_t)p * m.n_vocab + j]));
                if (rowmax < 1e-2f) { ++soft_mism; fprintf(stderr, "VERIFY GARBAGE row %ld (all-zero logits)\n", p); continue; }
                if (pred != seq[base + p + 1]) {
                    const float* lg = &logits_h[(size_t)p * m.n_vocab];
                    const double gap = fabs((double)lg[seq[base + p + 1]] - (double)lg[pred]);
                    if (gap > 0.25) { ++soft_mism; fprintf(stderr, "VERIFY MISMATCH step %zu row %ld: pred=%ld(%s) want=%ld(%s) gap=%.4f\n",
                        base / Bp, p, pred, pred < (long)tok.vocab.size() ? tok.vocab[pred].c_str() : "?",
                        seq[base + p + 1], seq[base + p + 1] < (long)tok.vocab.size() ? tok.vocab[seq[base + p + 1]].c_str() : "?", gap); }
                }
            }
            const long fv = free_vram_mib();
            if (fv >= 0 && fv < free_min) free_min = fv;
            safety("verify");
            steady_cold_bytes += cold;
            if (base >= Bp) { sum_step += s; ++n_steps; if (s < best_step) best_step = s; }   // skip first (cold caches)
            printf("verify_step,%zu,positions=%ld,seconds=%.4f,tok_per_s=%.3f,cold_bytes=%llu,free_vram_MiB=%ld\n",
                   base / Bp, Bp, s, (double)Bp / s, (unsigned long long)cold, fv);
        }
        const double mean_step = n_steps ? sum_step / n_steps : 0;
        printf("verify_summary,positions_per_step=%ld,steps=%ld,mean_step_s=%.4f,best_step_s=%.4f,mean_tok_per_s=%.3f,best_tok_per_s=%.3f,cold_GiB_per_step=%.3f,cold_GiB_per_token=%.4f,soft_mismatches=%ld,soft_checked=%ld,min_free_vram_MiB=%ld\n",
               Bp, n_steps, mean_step, best_step, mean_step > 0 ? Bp / mean_step : 0, best_step > 0 ? Bp / best_step : 0,
               (double)steady_cold_bytes / 1073741824.0, n_steps ? ((double)steady_cold_bytes / n_steps / Bp / 1073741824.0) : 0,
               soft_mism, soft_checked, free_min);
        return 0;
    }
    // ---- E046: MTP forward — one gated-attention layer (the nextn block) ----
    // h_in: [B][n_embd] device rows (h_tgt of the previous position for a sync pass, the
    // MTP's own output for a draft step). toks: the tokens at positions pos0..pos0+B-1.
    // Appends MTP KV causally; h_out = [B][n_embd] shared-head-normed hidden. With
    // want_logits, the head GEMV runs for row `logit_row` and its logits land in logits[0].
    auto mtp_S = [&](const std::string& tn) -> const float* { return (const float*)(mtp_small_dev + mtp_small_off.at(tn)); };
    const long mtp_kv_layer = m.n_kv * mtp_capacity * m.head_v;
    std::vector<int> draft_ids;
    std::vector<char> draft_seen(mtp_on ? m.n_vocab : 0);
    int* draft_device = nullptr;
    const long common = std::clamp(draft_vocab, 0L, m.n_vocab);
    auto add_draft_id = [&](long token) {
        if (token >= 0 && token < m.n_vocab && !draft_seen[token]) {
            draft_seen[token] = true;
            draft_ids.push_back((int)token);
        }
    };
    if (mtp_on && common && common < m.n_vocab) {
        check(cudaMalloc(&draft_device, m.n_vocab * sizeof(int)), "draft vocabulary");
    }
    struct MtpBranchBatch {
        int first;
        int stride;
        int max_depth;
        std::vector<int> depth;
        std::vector<int> paths;
    };
    auto mtp_forward = [&](const float* h_in, const std::vector<long>& toks, long B, long pos0,
                           float* h_out, bool want_logits, long logit_row, bool shortlist = false,
                           const MtpBranchBatch* grouped = nullptr) {
        g_batch = mat_width(B);
        const int storage = (int)pos0 + (grouped ? grouped->first : 0);
        if (grouped) {
            check(cudaMemcpyAsync(mtp_tree_depth, grouped->depth.data(), B * sizeof(int), cudaMemcpyHostToDevice, stc), "MTP depths upload");
            check(cudaMemcpyAsync(mtp_tree_path, grouped->paths.data(), grouped->paths.size() * sizeof(int), cudaMemcpyHostToDevice, stc), "MTP paths upload");
        }
        for (long v = 0; v < B; ++v) {
            if (pread(fdc, emb_buf + (size_t)v * emb_nb * 18, emb_nb * 18, (off_t)(data_start + embd->off + (uint64_t)toks[v] * emb_nb * 18)) != (ssize_t)(emb_nb * 18)) { fprintf(stderr, "mtp embd pread\n"); exit(1); }
            cpu_dequant_row_q4(emb_buf + (size_t)v * emb_nb * 18, m.n_embd, &emb_row[(size_t)v * m.n_embd]);
        }
        check(cudaMemcpy(x, emb_row.data(), B * m.n_embd * 4, cudaMemcpyHostToDevice), "mtp emb h2d");
        // enorm(emb) -> cat[..n_embd), hnorm(h) -> cat[n_embd..): concat is [e; h]
        k_rms_norm<<<dim3(1, (unsigned)B), 256, 0, stc>>>(mtp_cat, x, mtp_S("blk.64.nextn.enorm.weight"), (int)m.n_embd, (float)m.eps, (int)(2 * m.n_embd));
        k_rms_norm<<<dim3(1, (unsigned)B), 256, 0, stc>>>(mtp_cat + m.n_embd, h_in, mtp_S("blk.64.nextn.hnorm.weight"), (int)m.n_embd, (float)m.eps, (int)(2 * m.n_embd));
        const dim3 grid1m(1, (unsigned)B);
        k_quant_x<<<dim3((unsigned)((2 * m.n_embd / 32 * B + 127) / 128)), 128, 0, stc>>>(mtp_cat, 2 * m.n_embd / 32 * B, xq, xmeta);
        launch_mat_q4(mtp_dev + mtp_off.at("blk.64.nextn.eh_proj.weight"), 2 * m.n_embd, m.n_embd, xq, xmeta, mtp_resid, m.n_embd, stc);
        k_rms_norm<<<grid1m, 256, 0, stc>>>(mtp_normed, mtp_resid, mtp_S("blk.64.attn_norm.weight"), (int)m.n_embd, (float)m.eps);
        k_quant_x<<<dim3((unsigned)((m.n_embd / 32 * B + 127) / 128)), 128, 0, stc>>>(mtp_normed, m.n_embd / 32 * B, xq, xmeta);
        if (h_out) launch_mat_q4(mtp_dev + mtp_off.at("blk.64.attn_q.weight"), m.n_embd, 12288, xq, xmeta, yA, 12288, stc);
        launch_mat_q4(mtp_dev + mtp_off.at("blk.64.attn_k.weight"), m.n_embd, 1024, xq, xmeta, yB, 1024, stc);
        launch_mat_q4(mtp_dev + mtp_off.at("blk.64.attn_v.weight"), m.n_embd, 1024, xq, xmeta, t_v, 1024, stc);
        if (h_out) k_rms_heads<<<dim3((unsigned)nh_q, (unsigned)B), 128, 0, stc>>>(yA, mtp_S("blk.64.attn_q_norm.weight"), (int)nh_q, (int)hd, (float)m.eps, 2 * (int)hd);
        k_rms_heads<<<dim3((unsigned)nh_k, (unsigned)B), 128, 0, stc>>>(yB, mtp_S("blk.64.attn_k_norm.weight"), (int)nh_k, (int)hd, (float)m.eps, (int)hd);
        k_rope<<<dim3((unsigned)(nh_q + nh_k), (unsigned)B), 128, 0, stc>>>(yA, yB,
            (int)pos0, (int)nh_q, (int)nh_k, (int)hd, (int)m.n_rot,
            (float)m.rope_base, 2 * (int)hd, 0, 1, grouped ? mtp_tree_depth : nullptr);
        k_append_kv<<<dim3((unsigned)((nh_k * hd + 255) / 256), (unsigned)B), 256, 0, stc>>>(
            mtp_kv_dev, mtp_kv_dev + mtp_kv_layer, yB, t_v, storage, (int)mtp_capacity, (int)nh_k, (int)hd, 0, true);
        if (!h_out) {
            check(cudaGetLastError(), "mtp KV kernel");
            return;
        }
        // Limit only the proposer. The target verifier keeps its full context.
        const long first_pos = large_tier ? std::max(0L, pos0 + 1 - mtp_window) : 0;
        const long score_rows = pos0 + (grouped ? grouped->max_depth + 1 : B) - first_pos;
        k_attn_decode<<<dim3((unsigned)nh_q, (unsigned)B), 256, (unsigned)(score_rows * sizeof(float)), stc>>>(
            t_att, mtp_kv_dev, mtp_kv_dev + mtp_kv_layer, yA, (int)pos0, (int)mtp_capacity,
            (int)nh_q, (int)nh_k, (int)hd, attn_scale, 2 * (int)hd, 0, 1,
            grouped ? mtp_tree_depth : nullptr, grouped ? mtp_tree_path : nullptr, grouped ? grouped->stride : 0, first_pos);
        k_attn_gate_apply<<<dim3((unsigned)((nh_q * hd + 255) / 256), (unsigned)B), 256, 0, stc>>>(t_att, yA, (int)nh_q, (int)hd);
        k_quant_x<<<dim3((unsigned)((nh_q * hd / 32 * B + 127) / 128)), 128, 0, stc>>>(t_att, nh_q * hd / 32 * B, xq, xmeta);
        launch_mat_q4(mtp_dev + mtp_off.at("blk.64.attn_output.weight"), nh_q * hd, m.n_embd, xq, xmeta, yA, m.n_embd, stc);
        const unsigned addgrid_m = (unsigned)((m.n_embd * B + 255) / 256);
        k_add<<<addgrid_m, 256, 0, stc>>>(mtp_resid, yA, (int)(m.n_embd * B));
        k_rms_norm<<<grid1m, 256, 0, stc>>>(mtp_normed, mtp_resid, mtp_S("blk.64.post_attention_norm.weight"), (int)m.n_embd, (float)m.eps);
        k_quant_x<<<dim3((unsigned)((m.n_embd / 32 * B + 127) / 128)), 128, 0, stc>>>(mtp_normed, m.n_embd / 32 * B, xq, xmeta);
        launch_mat_q4(mtp_dev + mtp_off.at("blk.64.ffn_gate.weight"), m.n_embd, m.n_ff, xq, xmeta, yA, m.n_ff, stc);
        launch_mat_q4(mtp_dev + mtp_off.at("blk.64.ffn_up.weight"), m.n_embd, m.n_ff, xq, xmeta, yB, m.n_ff, stc);
        if (fuse_quant) k_silu_quant<<<(unsigned)((m.n_ff * B + 255) / 256), 256, 0, stc>>>(xq, xmeta, yA, yB, m.n_ff * B);
        else {
            k_silu_mul<<<(unsigned)((m.n_ff * B + 255) / 256), 256, 0, stc>>>(t_ffnh, yA, yB, (int)(m.n_ff * B));
            k_quant_x<<<(unsigned)((m.n_ff / 32 * B + 127) / 128), 128, 0, stc>>>(t_ffnh, m.n_ff / 32 * B, xq, xmeta);
        }
        launch_mat_q4(mtp_dev + mtp_off.at("blk.64.ffn_down.weight"), m.n_ff, m.n_embd, xq, xmeta, yA, m.n_embd, stc);
        k_add<<<addgrid_m, 256, 0, stc>>>(mtp_resid, yA, (int)(m.n_embd * B));
        k_rms_norm<<<grid1m, 256, 0, stc>>>(h_out, mtp_resid, mtp_S("blk.64.nextn.shared_head_norm.weight"), (int)m.n_embd, (float)m.eps);
        if (want_logits) {
            const long rows = grouped ? B : 1;
            const long blocks = rows * m.n_embd / 32;
            k_quant_x<<<(blocks + 127) / 128, 128, 0, stc>>>(h_out + (size_t)logit_row * m.n_embd, blocks, xq, xmeta);
            const uint8_t* weights = mtp_dev + mtp_off.at("output.weight");
            const long vocab = shortlist ? (long)draft_ids.size() : m.n_vocab;
            const int* ids = shortlist ? draft_device : nullptr;
            if (rows == 1) k_mat_q4<1><<<(vocab + 3) / 4, 128, 0, stc>>>(weights, m.n_embd, vocab, xq, xmeta, logits, vocab, ids);
            else if (rows == 2) k_mat_q4<2><<<(vocab + 3) / 4, 128, 0, stc>>>(weights, m.n_embd, vocab, xq, xmeta, logits, vocab, ids);
            else k_mat_q4<4><<<(vocab + 3) / 4, 128, 0, stc>>>(weights, m.n_embd, vocab, xq, xmeta, logits, vocab, ids);
            const size_t count = (size_t)rows * vocab;
            if (count > logits_h.size()) logits_h.resize(count);
            check(cudaMemcpyAsync(logits_h.data(), logits, count * sizeof(float), cudaMemcpyDeviceToHost, stc), "mtp logits d2h");
        }
        check(cudaGetLastError(), "mtp kernel");
        check(cudaStreamSynchronize(stc), "mtp sync");
    };

    auto mtp_commit = [&](const std::vector<long>& tokens, long base, int final_row, const int* path, bool dump,
                          long target_offset = 0) {
        // MTP conditions on the previous target hidden state at each position.
        // The accepted rows populate its KV cache, then only the last state carries forward.
        const int count = (int)tokens.size();
        const long elements = (long)count * m.n_embd;
        const float* target_rows = t_norm + target_offset * m.n_embd;
        k_mtp_shift_hidden<<<(elements + 255) / 256, 256, 0, stc>>>(mtp_last_hidden, target_rows, path,
            mtp_hout, (int)m.n_embd, count);
        check(cudaMemcpyAsync(mtp_last_hidden, target_rows + (size_t)final_row * m.n_embd,
            m.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stc), "advance last target hidden state");
        mtp_forward(mtp_hout, tokens, count, base, dump ? mtp_hout : nullptr, false, -1);
    };

    // Attention KV stays in place as generation appends. Keep recurrent
    // state in host RAM at the prompt checkpoint and optional shared prefix.
    std::vector<long> checkpoint_ids;
    std::vector<long> committed_ids;
    std::vector<long> shared_prefix_ids;
    std::vector<uint8_t> checkpoint_state;
    std::vector<uint8_t> shared_prefix_state;
    auto copy_checkpoint = [&](std::vector<uint8_t>& state, bool restore, bool boundary = false) {
        const size_t hidden_bytes = mtp_on ? m.n_embd * sizeof(float) : 0;
        state.resize(ssm_bytes + conv_bytes + hidden_bytes);
        uint8_t* host = state.data();
        const std::pair<void*, size_t> regions[] = {
            {boundary ? trial_ssm : ssm_dev, ssm_bytes},
            {boundary ? trial_conv : conv_dev, conv_bytes}, {mtp_last_hidden, hidden_bytes}
        };
        check(cudaStreamSynchronize(stc), "checkpoint sync");
        for (const auto& [device, bytes] : regions) {
            if (!bytes) continue;
            if (restore) check(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "restore prefill state");
            else check(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "save prefill state");
            host += bytes;
        }
    };
    auto run_request = [&](const std::vector<long>& request, const std::vector<long>& hint, long gen_tokens) -> int {
        // One sequence. Verify a draft against the unchanged Q4 target, then
        // commit only the recurrent inputs and attention KV of the accepted prefix.
        const auto prefill_start = std::chrono::steady_clock::now();
        const long G = speculate;
        std::vector<long> hist = request;
        std::vector<long> emitted;
        auto stream_tokens = [&](size_t count) {
            if (!session_stream || count == 0) return;
            printf("session_delta");
            for (size_t i = emitted.size() - count; i < emitted.size(); ++i)
                printf(",%ld", emitted[i]);
            printf("\n");
            fflush(stdout);
        };
        long known = -1;
        if (draft_device) {
            draft_ids.clear();
            std::fill(draft_seen.begin(), draft_seen.end(), false);
            for (long token = 0; token < common; ++token) add_draft_id(token);
            for (long token : request) add_draft_id(token);
            add_draft_id(m.eos_id);
            check(cudaMemcpy(draft_device, draft_ids.data(), draft_ids.size() * sizeof(int),
                cudaMemcpyHostToDevice), "draft ids");
        }
        const size_t complete_prefix = (hist.size() - 1) / prefill_width * prefill_width;
        const size_t shared_prefix = (size_t)session_prefix_tokens;
        size_t reused = 0;
        if (session) {
            if (hist.size() > (size_t)(ctx - gen_tokens)) {
                fprintf(stderr, "session context is too small for the prompt and generation limit\n");
                return 2;
            }
            // A fresh request may give its final chunk a different shape and
            // output head. Save only completed nonfinal prefill chunks.
            if (session_tail && !committed_ids.empty() && committed_ids.size() <= complete_prefix &&
                std::equal(committed_ids.begin(), committed_ids.end(), hist.begin())) {
                // The cache and recurrent state are already at this position.
                reused = committed_ids.size();
            } else if (!checkpoint_ids.empty() && checkpoint_ids.size() <= complete_prefix &&
                (!tier_kv || (long)checkpoint_ids.size() >= tier_written - (tier_cache.ring - tier_cache.recent_tokens)) &&
                std::equal(checkpoint_ids.begin(), checkpoint_ids.end(), hist.begin())) {
                reused = checkpoint_ids.size();
                copy_checkpoint(checkpoint_state, true);
            } else if (!shared_prefix_ids.empty() && shared_prefix_ids.size() <= complete_prefix &&
                std::equal(shared_prefix_ids.begin(), shared_prefix_ids.end(), hist.begin())) {
                reused = shared_prefix_ids.size();
                copy_checkpoint(shared_prefix_state, true);
                checkpoint_ids.clear();
            } else if (cur_pos) {
                check(cudaMemset(ssm_dev, 0, ssm_bytes), "reset session recurrent state");
                check(cudaMemset(conv_dev, 0, conv_bytes), "reset session convolution state");
                if (mtp_on)
                    check(cudaMemset(mtp_last_hidden, 0, m.n_embd * sizeof(float)), "reset session MTP state");
                checkpoint_ids.clear();
            }
            if (reused < shared_prefix_ids.size()) shared_prefix_ids.clear();
            cur_pos = (long)reused;
            if (tier_kv) tier_packed = tier_written = cur_pos;
            free_min = free_vram_mib();
        }
        const long hint_limit = std::min({(long)hint.size(), gen_tokens - 1, 2 * kPrefillChunk});
        long prefill_passes = 0;
        bool hint_rejected = false;
        for (size_t base = reused; base < hist.size(); ) {
            const size_t chunk_start = base / prefill_width * prefill_width;
            const long canonical_rows = std::min<long>(prefill_width, hist.size() - chunk_start);
            long nb = canonical_rows - (long)(base - chunk_start);
            size_t grouped_end = std::min(complete_prefix, chunk_start + prefill_chunks * prefill_width);
            if (shared_prefix_ids.empty() && base < shared_prefix)
                grouped_end = std::min(grouped_end, shared_prefix);
            if (prefill_chunks > 1 && grouped_end > base + kPrefillChunk)
                nb = (long)(grouped_end - base);
            long physical_rows = canonical_rows;
            long prefill_split = 0;
            long checkpoint_row = 0;
            // Keep the nonfinal checkpoint while scoring the final prompt rows
            // in the same weight pass using the original matrix widths.
            if (fuse_prefill_tail && base + nb == complete_prefix &&
                nb + mat_width(hist.size() - complete_prefix) <= B) {
                checkpoint_row = nb;
                nb = (long)(hist.size() - base);
                physical_rows = nb - checkpoint_row;
            }
            // Preserve the prompt's canonical chunks, but read each weight
            // only once. Keep any earlier checkpoint for changed-prefix replay.
            if (reused && hint_limit && base + nb == complete_prefix &&
                (!shared_prefix_ids.empty() || shared_prefix != complete_prefix) &&
                hist.size() - base + hint_limit + kPrefillChunk <= (size_t)B) {
                prefill_split = nb;
                nb = (long)(hist.size() - base);
                physical_rows = nb - prefill_split;
            }
            std::vector<long> toks(hist.begin() + base, hist.begin() + base + nb);
            double s; uint64_t cold;
            const long head_row = share_output && !dbg_dump ? (base + nb == hist.size() ? nb - 1 : -2) : -1;
            const long fused = head_row >= 0 ? std::max(0L, std::min(hint_limit, B - nb - kPrefillChunk)) : 0;
            if (fused) toks.insert(toks.end(), hint.begin(), hint.begin() + fused);
            run_token(toks, nb + fused, (int)(base / prefill_width), &s, &cold, false, false, head_row, false,
                fused || nb < canonical_rows || checkpoint_row ? physical_rows : 0,
                {fused, prefill_split, checkpoint_row});
            tier_commit(hist, (long)(base + nb));
            ++prefill_passes;
            if (session_stream && prefill_passes % 8 == 0) {
                printf("session_prefill,processed=%zu,prompt=%zu\n", base + nb, hist.size());
                fflush(stdout);
            }
            if (base + nb == hist.size()) {
                known = target_id(nb - 1);
                if (parallel_on) parallel.commit(t_norm + (size_t)(nb - 1) * m.n_embd, stc);
            }
            if (fused) {
                long matched = 0;
                while (matched < fused && target_id(nb + matched - 1) == hint[matched] && hint[matched] != m.eos_id)
                    ++matched;
                hint_rejected = matched < fused;
                printf("prefill_hint,proposed=%ld,matched=%ld,committed=%ld,cold_bytes=%llu\n",
                       fused, matched, hint_rejected ? 0 : matched, (unsigned long long)cold);
                if (!hint_rejected) {
                    check(cudaMemcpyAsync(ssm_dev, trial_ssm, ssm_bytes, cudaMemcpyDeviceToDevice, stc), "commit draft recurrent state");
                    check(cudaMemcpyAsync(conv_dev, trial_conv, conv_bytes, cudaMemcpyDeviceToDevice, stc), "commit draft convolution state");
                    emitted.assign(hint.begin(), hint.begin() + matched);
                    known = target_id(nb + matched - 1);
                }
                cur_pos = (long)(base + nb + emitted.size());
                toks.resize(nb + emitted.size());
            }
            if (nb > prefill_width || fused || checkpoint_row) {
                for (long row = 0; row < (long)toks.size(); ) {
                    const long end = row < nb ? nb : (long)toks.size();
                    const long width = row < nb ? kPrefillChunk - (long)(base + row) % kPrefillChunk : kPrefillChunk;
                    const long count = std::min(end - row, width);
                    std::vector<long> chunk(toks.begin() + row, toks.begin() + row + count);
                    mtp_commit(chunk, (long)base + row, count - 1, nullptr, false, row);
                    row += count;
                    if (checkpoint_row && row == checkpoint_row) {
                        copy_checkpoint(checkpoint_state, false, true);
                        checkpoint_ids.assign(hist.begin(), hist.begin() + complete_prefix);
                    }
                }
            } else if (mtp_on && !oracle) mtp_commit(toks, (long)base, (int)nb - 1, nullptr, dbg_dump);
            if (eagle_on) {
                eagle.project((int)base, (int)nb);
                eagle_embeddings(toks);
                eagle.attention(eagle.hidden_context + base * m.n_embd, (int)base, (int)nb, false);
            }
            if (session && base + nb == complete_prefix) {
                copy_checkpoint(checkpoint_state, false);
                checkpoint_ids.assign(hist.begin(), hist.begin() + complete_prefix);
            }
            if (shared_prefix && base + nb == shared_prefix && shared_prefix <= complete_prefix) {
                copy_checkpoint(shared_prefix_state, false);
                shared_prefix_ids.assign(hist.begin(), hist.begin() + shared_prefix);
            }
            // teacher-forced MTP sanity: row r's head argmax should predict the token
            // after toks[r] for most rows — a low rate means the MTP forward is broken
            if (mtp_on && !oracle && dbg_dump) {
                g_batch = 1;
                long tf_ok = 0, tf_n = 0;
                for (long r = 0; r + 1 < nb; ++r) {
                    k_quant_x<<<dim3((unsigned)((m.n_embd / 32 + 127) / 128)), 128, 0, stc>>>(mtp_hout + (size_t)r * m.n_embd, m.n_embd / 32, xq, xmeta);
                    launch_mat_q4(mtp_dev + mtp_off.at("output.weight"), m.n_embd, m.n_vocab, xq, xmeta, logits, m.n_vocab, stc);
                    check(cudaStreamSynchronize(stc), "tf sync");
                    check(cudaMemcpy(logits_h.data(), logits, m.n_vocab * 4, cudaMemcpyDeviceToHost), "tf d2h");
                    if (argmax(logits_h.data()) == toks[r + 1]) ++tf_ok;
                    else if (dbg_dump && base == 0 && r < 6)
                        printf("dbg,mtftfmiss,r=%ld,pred=%ld,want=%ld\n", r, argmax(logits_h.data()), toks[r + 1]);
                    ++tf_n;
                }
                printf("dbg,mtp_tf,pb=%zu,ok=%ld,n=%ld\n", base, tf_ok, tf_n);
            }
            base += nb;
        }
        if (prefill_chunks > 1)
            printf("grouped_prefill,passes=%ld,processed_tokens=%zu\n", prefill_passes, hist.size() - reused);
        safety("spec-prefill");
        check(cudaStreamSynchronize(stc), "spec prefill");
        const auto decode_start = std::chrono::steady_clock::now();
        const double prefill_s = std::chrono::duration<double>(decode_start - prefill_start).count();
        long cycles = 0;
        long mtp_accepted = 0, mtp_draft = 0, pld_accepted = 0, pld_draft = 0;
        long direct_accepted = 0, direct_draft = 0;   // E167: acceptance of direct deep chains only
        double chain_q_ewma = 1.0;   // E168: EWMA recovers after a low-acceptance stretch
        std::vector<long> draft_hist;   // E167: draft length (rows-1) per cycle
        long eagle_accepted = 0, eagle_draft = 0;
        long parallel_accepted = 0, parallel_proposed = 0;
        double verify_s = 0, draft_s = 0, repair_s = 0;
        auto lookup_drafts = [&](long limit) {
            // E167: prefer the occurrence with the LONGEST continuation (an early repeat
            // with lots of following text), not just the most recent one — deep chains
            // need continuations past the tree budget. Longest-match-n still wins first.
            const long size = (long)hist.size();
            for (long n = std::min<long>(6, size - 1); n >= 1; --n) {
                std::vector<long> best;
                size_t best_len = 0;
                for (long j = size - n - 2; j >= 0; --j) {
                    if (!std::equal(hist.begin() + j, hist.begin() + j + n, hist.end() - n)) continue;
                    const long end = std::min(size, j + n + limit);
                    const size_t len = (size_t)(end - (j + n));
                    if (len > best_len) {
                        best_len = len;
                        best.assign(hist.begin() + j + n, hist.begin() + end);
                    }
                }
                if (!best.empty()) return best;
            }
            return std::vector<long>{};
        };
        auto lookup_after_proposal = [&](const std::vector<long>& drafts, long proposal, long limit) {
            std::vector<long> continuation;
            const long hlen = (long)hist.size();
            const long length = hlen + (long)drafts.size();
            // Only propose continuations seen in the real history. The draft model
            // must agree with their first token, after at least four matching tokens.
            long best = 3;
            for (long j = hlen - 4; j > 0; --j) {
                if (hist[j] != proposal) continue;
                long match = 0;
                while (match < std::min({j, length, 32L})) {
                    const long p = length - match - 1;
                    const long token = p < hlen ? hist[p] : drafts[p - hlen];
                    if (token != hist[j - match - 1]) break;
                    ++match;
                }
                if (match <= best) continue;
                best = match;
                continuation.assign(hist.begin() + j, hist.begin() + std::min(hlen, j + limit));
            }
            return continuation;
        };
        // The first token comes from prompt evaluation. Each later token comes
        // either from a matched draft or from the target's next-token prediction.
        const long prefilled_output = (long)emitted.size();
        hist.insert(hist.end(), emitted.begin(), emitted.end());
        if (!prefilled_output || known != m.eos_id) {
            hist.push_back(known);
            emitted.push_back(known);
        }
        const long emitted_before_decode = (long)emitted.size();
        stream_tokens(emitted.size());
        if (prefilled_output && draft_device) {
            const size_t old = draft_ids.size();
            for (long id : emitted) add_draft_id(id);
            if (draft_ids.size() > old)
                check(cudaMemcpy(draft_device + old, draft_ids.data() + old, (draft_ids.size() - old) * sizeof(int),
                                 cudaMemcpyHostToDevice), "extend prefill draft vocabulary");
        }
        bool stop = known == m.eos_id;
        bool hint_matches = !hint_rejected && prefilled_output < (long)hint.size() && hint[prefilled_output] == known;
        while (!stop && (long)emitted.size() < gen_tokens) {
            const auto cycle_start = std::chrono::steady_clock::now();
            const long base = cur_pos;
            const long remaining = gen_tokens - (long)emitted.size();
            const long limit = std::min(G, remaining) - 1;
            const long mtp_limit = std::min(limit, draft_max);
            const size_t hint_start = std::min(emitted.size(), hint.size());
            const long hint_count = hint_matches ? std::min(limit, (long)(hint.size() - hint_start)) : 0;
            std::vector<long> drafts = !oracle && !parallel_on && !eagle_on && (!mtp_on || pld_enabled == 1)
                ? lookup_drafts(limit) : std::vector<long>{};
            // E167 deep long-match override: an exact history chain longer than the tree
            // budget verifies as one linear pass — no drafter forwards, no tree. Gated on
            // an EWMA of DIRECT-chain acceptance (guided-tree acceptance is a different
            // statistic; the EWMA recovers after a low-acceptance stretch, E168): a deep
            // pass pays the full width even when the target diverges early.
            if (mtp_on && pld_enabled == 2 && !oracle && !parallel_on && !eagle_on) {
                std::vector<long> chain = lookup_drafts(limit);
                if ((long)chain.size() > (chain_min ? chain_min : mtp_limit) && chain_q_ewma >= 0.75) {
                    // Cold start: bounded probe — +16 positions past the tree budget
                    // buys the acceptance evidence without paying the full width.
                    if (direct_draft < 8)
                        chain.resize((size_t)std::min<long>((long)chain.size(), mtp_limit + 16));
                    drafts = std::move(chain);
                }
            }
            const bool use_pld = !mtp_on || (drafts.size() >= 6);
            double this_draft_s = 0;
            long model_drafts = 0;
            DraftTree tree_drafts;
            const bool tree_mtp = mtp_tree && !oracle && !use_pld && !hint_count;
            auto guided_history = [&](const std::vector<long>& prefix, long token, long count) {
                return pld_enabled == 2 ? lookup_after_proposal(prefix, token, count) : std::vector<long>{};
            };
            if (oracle) {
                const size_t first = emitted.size();
                if (known != oracle_ids[first - 1]) {
                    fprintf(stderr, "oracle root differs from target at position %zu\n", first - 1);
                    return 1;
                }
                drafts.assign(oracle_ids.begin() + first, oracle_ids.begin() + first + mtp_limit);
            } else if (hint_count) {
                drafts.assign(hint.begin() + hint_start, hint.begin() + hint_start + hint_count);
            } else if (parallel_on) {
                const auto draft_start = std::chrono::steady_clock::now();
                const ssize_t bytes = emb_nb * 18;
                if (pread(fdc, emb_buf, bytes, data_start + embd->off + known * bytes) != bytes) {
                    fprintf(stderr, "parallel draft embedding read failed\n");
                    return 1;
                }
                cpu_dequant_row_q4(emb_buf, m.n_embd, emb_row.data());
                drafts = parallel.propose(emb_row.data(), mtp_limit, m.eos_id, stc);
                model_drafts = (long)drafts.size();
                this_draft_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - draft_start).count();
            } else if (eagle_on) {
                const auto draft_start = std::chrono::steady_clock::now();
                // Tree node budget stays at mtp_limit: every model-expanded node costs a
                // drafter forward, so the deep-γ ceiling (limit, up to 63) applies only to
                // PLD history chains, never to tree expansion (novel chat unchanged).
                tree_drafts = eagle.draft(known, m.eos_id, (int)base, (int)mtp_limit, (int)mtp_limit, p_min,
                    [&](long token) { eagle_embeddings({token}); }, guided_history);
                drafts = std::move(tree_drafts.tokens);
                model_drafts = (long)std::count(tree_drafts.from_history.begin(), tree_drafts.from_history.end(), false);
                this_draft_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - draft_start).count();
            } else if (tree_mtp) {
                const auto draft_start = std::chrono::steady_clock::now();
                std::vector<int> ranking(m.n_vocab);
                // Tree budget stays at mtp_limit (E149 behavior): deeper trees only added
                // rejected width, and every model-expanded node costs a drafter forward.
                tree_drafts = build_draft_tree(m.eos_id, (int)mtp_limit, (int)mtp_limit, p_min, mtp_group,
                    [&](int first, int count, const std::vector<std::vector<int>>& paths, const DraftTree& tree, DraftOptions* options) {
                        const bool shortlist = first > 0 && draft_device;
                        const int vocab = shortlist ? (int)draft_ids.size() : (int)m.n_vocab;
                        MtpBranchBatch input{first, (int)batch, 0, {}, {}};
                        std::vector<long> tokens;
                        input.paths.resize((size_t)count * input.stride, -1);
                        const float* hidden_batch = mtp_tree_inputs;
                        for (int row = 0; row < count; ++row) {
                            const int node = first + row;
                            const auto& path = paths[node];
                            input.depth.push_back((int)path.size() - 1);
                            input.max_depth = std::max(input.max_depth, input.depth.back());
                            std::copy(path.begin(), path.end(), input.paths.begin() + (size_t)row * input.stride);
                            const float* hidden = node ? mtp_tree_hidden + (size_t)tree.parents[node] * m.n_embd : mtp_last_hidden;
                            if (count == 1) hidden_batch = hidden;
                            else check(cudaMemcpyAsync(mtp_tree_inputs + (size_t)row * m.n_embd, hidden,
                                m.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stc), "MTP branch input");
                            tokens.push_back(node ? tree.tokens[node - 1] : known);
                        }
                        mtp_forward(hidden_batch, tokens, count, base, mtp_hout2, true, 0, shortlist, &input);
                        check(cudaMemcpyAsync(mtp_tree_hidden + (size_t)first * m.n_embd, mtp_hout2,
                            (size_t)count * m.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stc), "MTP branch results");
                        const int top = std::min(128, vocab);
                        for (int row = 0; row < count; ++row) {
                            const float* logits = logits_h.data() + (size_t)row * vocab;
                            for (int i = 0; i < vocab; ++i) ranking[i] = i;
                            std::partial_sort(ranking.begin(), ranking.begin() + top, ranking.begin() + vocab,
                                [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
                            const double peak = logits[ranking[0]];
                            double mass = 0;
                            for (int i = 0; i < top; ++i) mass += exp((logits[ranking[i]] - peak) / 1.2);
                            auto& proposal = options[row];
                            proposal.count = std::min(kDraftFanout, top);
                            for (int i = 0; i < proposal.count; ++i) {
                                const int pick = ranking[i];
                                proposal.choices[i] = {shortlist ? draft_ids[pick] : pick, exp((logits[pick] - peak) / 1.2) / mass};
                            }
                        }
                    }, guided_history, (int)mtp_limit);
                drafts = std::move(tree_drafts.tokens);
                model_drafts = (long)std::count(tree_drafts.from_history.begin(), tree_drafts.from_history.end(), false);
                this_draft_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - draft_start).count();
            } else if (!use_pld) {
                drafts.clear();
                const auto mtp_start = std::chrono::steady_clock::now();
                const float* hidden = mtp_last_hidden;
                long token = known;
                for (long k = 0; k < mtp_limit; ++k) {
                    // Score the full draft vocabulary when anchored by a target
                    // hidden state. Later draft steps can use the smaller list.
                    const bool shortlist = draft_device && k > 0;
                    const long vocab = shortlist ? (long)draft_ids.size() : m.n_vocab;
                    mtp_forward(hidden, {token}, 1, base + k, mtp_hout2, true, 0, shortlist);
                    const long index = std::max_element(logits_h.begin(), logits_h.begin() + vocab) - logits_h.begin();
                    const long proposal = shortlist ? draft_ids[index] : index;
                    if (p_min > 0.0) {
                        const double peak = logits_h[index];
                        double sum = 0.0;
                        for (long j = 0; j < vocab; ++j) sum += exp((double)logits_h[j] - peak);
                        if (1.0 / sum < p_min) break;
                    }
                    ++model_drafts;
                    if (pld_enabled == 2 && limit - k >= 4) {
                        auto continuation = lookup_after_proposal(drafts, proposal, limit - k);
                        if (continuation.size() >= 4) {
                            drafts.insert(drafts.end(), continuation.begin(), continuation.end());
                            break;
                        }
                    }
                    drafts.push_back(proposal);
                    if (proposal == m.eos_id) break;
                    token = proposal;
                    hidden = mtp_hout2;
                }
                this_draft_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - mtp_start).count();
            }

            std::vector<long> feed = {known};
            feed.insert(feed.end(), drafts.begin(), drafts.end());
            const auto& parents = tree_drafts.parents;
            bool tree = false;
            for (size_t r = 1; r < parents.size(); ++r) tree |= parents[r] != (int)r - 1;
            std::vector<int> paths;
            if (tree) {
                const int rows = (int)feed.size();
                const int width = (int)G;
                std::vector<int> depths(rows), owners(rows, -1), leaf_paths, lengths;
                paths.resize(rows * width, -1);
                paths[0] = 0;
                for (int row = 1; row < rows; ++row) {
                    const int parent = parents[row];
                    depths[row] = depths[parent] + 1;
                    std::copy_n(paths.begin() + parent * width, depths[row], paths.begin() + row * width);
                    paths[row * width + depths[row]] = row;
                }
                std::vector<bool> terminal(rows, true);
                for (int row = 1; row < rows; ++row) terminal[parents[row]] = false;
                for (int leaf = 0; leaf < rows; ++leaf) {
                    if (!terminal[leaf]) continue;
                    for (int d = 0; d <= depths[leaf]; ++d) {
                        const int row = paths[leaf * width + d];
                        if (owners[row] < 0) owners[row] = (int)lengths.size();
                    }
                    leaf_paths.insert(leaf_paths.end(), paths.begin() + leaf * width, paths.begin() + (leaf + 1) * width);
                    lengths.push_back(depths[leaf] + 1);
                }
                tree_leaf_count = (int)lengths.size();
                check(cudaMemcpy(tree_depth, depths.data(), rows * sizeof(int), cudaMemcpyHostToDevice), "tree depths");
                check(cudaMemcpy(tree_paths, paths.data(), paths.size() * sizeof(int), cudaMemcpyHostToDevice), "tree paths");
                check(cudaMemcpy(tree_leaves, leaf_paths.data(), leaf_paths.size() * sizeof(int), cudaMemcpyHostToDevice), "tree leaf paths");
                check(cudaMemcpy(tree_lengths, lengths.data(), lengths.size() * sizeof(int), cudaMemcpyHostToDevice), "tree leaf lengths");
                check(cudaMemcpy(tree_owners, owners.data(), owners.size() * sizeof(int), cudaMemcpyHostToDevice), "tree row owners");
            }
            const long rows = (long)feed.size();
            if (replay_inputs && rows > 1 && !tree && !defer_recurrent) {
                check(cudaMemcpyAsync(ssm_snap, ssm_dev, ssm_bytes, cudaMemcpyDeviceToDevice, stc), "ssm snapshot");
                check(cudaMemcpyAsync(conv_snap, conv_dev, conv_bytes, cudaMemcpyDeviceToDevice, stc), "conv snapshot");
            }
            double pass_s;
            double pass_h2d_ms = 0.0;
            uint64_t cold;
            run_token(feed, rows, (int)cycles + 1, &pass_s, &cold, rows > 1, tree, -1, false,
                      0, PrefillDraft{}, &pass_h2d_ms);
            draft_hist.push_back(rows - 1);
            std::vector<long> prefix{feed[0]};
            int leaf = 0;
            for (;;) {
                known = target_id(leaf);
                if (known == m.eos_id) break;
                int child = -1;
                if (tree) {
                    for (int r = leaf + 1; r < rows; ++r)
                        if (parents[r] == leaf && feed[r] == known) { child = r; break; }
                } else if (leaf + 1 < rows && feed[leaf + 1] == known) {
                    child = leaf + 1;
                }
                if (child < 0) break;
                leaf = child;
                prefix.push_back(known);
            }
            const long committed = (long)prefix.size();
            const long accepted = committed - 1;
            stop = known == m.eos_id;
            if (hint_count) {
                const size_t next = hint_start + accepted;
                hint_matches = next < hint.size() && hint[next] == known;
                printf("decode_hint,proposed=%ld,accepted=%ld\n", hint_count, accepted);
            }
            const long new_tokens = accepted + (stop ? 0 : 1);
            emitted.insert(emitted.end(), prefix.begin() + 1, prefix.end());
            if (!stop) emitted.push_back(known);
            hist.insert(hist.end(), emitted.end() - new_tokens, emitted.end());
            if (oracle && accepted != rows - 1) {
                fprintf(stderr, "oracle continuation rejected at generated position %zu\n",
                        emitted.size() - new_tokens + accepted);
                return 1;
            }
            if (draft_device && !oracle) {
                const size_t old = draft_ids.size();
                for (auto it = emitted.end() - new_tokens; it != emitted.end(); ++it) add_draft_id(*it);
                if (draft_ids.size() > old) check(cudaMemcpy(draft_device + old, draft_ids.data() + old,
                    (draft_ids.size() - old) * sizeof(int), cudaMemcpyHostToDevice), "extend draft vocabulary");
            }
            long model_accepted = std::min(accepted, model_drafts);
            if (eagle_on || tree_mtp) {
                model_accepted = 0;
                for (int d = 1; d < committed; ++d) {
                    const int row = tree ? paths[leaf * G + d] : d;
                    model_accepted += !tree_drafts.from_history[row - 1];
                }
            }
            if (parallel_on) {
                parallel_proposed += model_drafts;
                parallel_accepted += model_accepted;
            } else if (eagle_on) {
                eagle_draft += model_drafts;
                eagle_accepted += model_accepted;
            } else {
                mtp_draft += model_drafts;
                mtp_accepted += model_accepted;
            }
            if (!oracle && !hint_count) {
                pld_draft += rows - 1 - model_drafts;
                pld_accepted += accepted - model_accepted;
            }
            if (!tree && rows > 1 && mtp_on && !oracle && !hint_count) {
                direct_draft += rows - 1;
                direct_accepted += accepted;
                chain_q_ewma = 0.8 * chain_q_ewma + 0.2 * ((double)accepted / (double)(rows - 1));
            }

            double commit_s = 0;
            if (session_tail || (!stop && (long)emitted.size() < gen_tokens)) {
                const auto commit_start = std::chrono::steady_clock::now();
                const int* accepted_path = tree ? tree_paths + leaf * G : nullptr;
                if (tree) {
                    if (tier_kv) {
                        k_tier_compact<<<m.n_attn * nh_k, kTierHead, 0, stc>>>(tier_cache, base, committed, accepted_path);
                    } else {
                        const int columns = (int)(m.n_attn * 2 * nh_k * hd);
                        k_compact_kv<<<(columns + 255) / 256, 256, 0, stc>>>(kv_dev, (int)ctx, (int)hd,
                            columns, (int)base, (int)committed, accepted_path);
                    }
                    const size_t stride = 3 * m.n_embd;
                    for (long d = 0; eagle_on && d < committed; ++d) {
                        const int from = paths[leaf * G + d];
                        if (from == d) continue;
                        check(cudaMemcpyAsync(eagle.features + d * stride, eagle.features + from * stride,
                            stride * sizeof(float), cudaMemcpyDeviceToDevice, stc), "commit EAGLE features");
                    }
                }
                if (replay_inputs && (tree || committed < rows || (defer_recurrent && rows > 1))) {
                    const auto repair_start = std::chrono::steady_clock::now();
                    if (!tree && !defer_recurrent) check(cudaMemcpyAsync(ssm_dev, ssm_snap, ssm_bytes, cudaMemcpyDeviceToDevice, stc), "ssm rollback");
                    const size_t conv_layer = (size_t)m.qkv_out * (m.sss_conv - 1);
                    const size_t ssm_layer = (size_t)m.sss_dt * Sv * Sv;
                    for (long ri = 0; ri < m.n_recr; ++ri) {
                        const float* saved = replay_inputs + ri * replay_layer;
                        k_conv_commit<<<(unsigned)((m.qkv_out + 255) / 256), 256, 0, stc>>>(
                            conv_dev + ri * conv_layer, ((tree || defer_recurrent) ? conv_dev : conv_snap) + ri * conv_layer, saved, (int)m.qkv_out, (int)committed, accepted_path);
                        launch_delta_multi(
                            ssm_dev + ri * ssm_layer, saved + replay_qkv, saved + 2 * replay_qkv,
                            saved + 2 * replay_qkv + replay_gate, nullptr, (int)Sv, (int)m.sss_dt,
                            (int)m.sss_group, dn_scale, m.qkv_out, m.sss_inner, (int)committed, stc, accepted_path);
                    }
                    check(cudaGetLastError(), "replay accepted recurrent prefix");
                    check(cudaStreamSynchronize(stc), "replay sync");
                    repair_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - repair_start).count();
                }
                commit_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - commit_start).count();
                if (eagle_on) {
                    const auto sync_start = std::chrono::steady_clock::now();
                    eagle.project((int)base, (int)committed);
                    eagle_embeddings(prefix);
                    eagle.attention(eagle.hidden_context + (size_t)base * m.n_embd, (int)base, (int)committed, false);
                    check(cudaStreamSynchronize(stc), "EAGLE context sync");
                    this_draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - sync_start).count();
                }
                if (mtp_on && !oracle) {
                    const auto sync_start = std::chrono::steady_clock::now();
                    mtp_commit(prefix, base, leaf, accepted_path, false);
                    check(cudaStreamSynchronize(stc), "mtp KV sync");
                    this_draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - sync_start).count();
                }
                if (parallel_on) {
                    const auto sync_start = std::chrono::steady_clock::now();
                    parallel.commit(t_norm + (size_t)leaf * m.n_embd, stc);
                    check(cudaStreamSynchronize(stc), "parallel draft commit");
                    this_draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - sync_start).count();
                }
                cur_pos = base + committed;
                tier_commit(hist, cur_pos);
            }

            verify_s += pass_s;
            draft_s += this_draft_s;
            const long fv = free_vram_mib();
            if (fv >= 0 && fv < free_min) free_min = fv;
            safety("spec");
            const double cycle_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - cycle_start).count();
            if (oracle)
                printf("oracle_cycle,%ld,draft=%ld,matched=%ld,emitted=%ld,verify_seconds=%.6f,commit_seconds=%.6f,seconds=%.6f,cold_bytes=%llu,free_vram_MiB=%ld\n",
                       cycles, rows - 1, accepted, new_tokens, pass_s, commit_s, cycle_s, (unsigned long long)cold, fv);
            printf("spec_cycle,%ld,draft=%ld,matched=%ld,emitted=%ld,verify_seconds=%.4f,%s=%.4f,seconds=%.4f,cold_bytes=%llu,cold_h2d_ms=%.3f,free_vram_MiB=%ld,tree=%d,tree_leaves=%d\n",
                   cycles++, rows - 1, accepted, new_tokens, pass_s, parallel_on ? "parallel_seconds" : eagle_on ? "eagle_seconds" : "mtp_seconds",
                   this_draft_s, cycle_s, (unsigned long long)cold, pass_h2d_ms, fv, tree, tree ? tree_leaf_count : 1);
            stream_tokens((size_t)new_tokens);
        }
        const double decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
        const long tot_emitted = (long)emitted.size() - emitted_before_decode;
        const long tot_accepted = mtp_accepted + eagle_accepted + pld_accepted + parallel_accepted;
        const long tot_draft = mtp_draft + eagle_draft + pld_draft + parallel_proposed;
        if (show_text) printf("generated_text,%s\n", tok.decode(emitted).c_str());
        if (parallel_on)
            printf("parallel_summary,proposed=%ld,accepted=%ld,mean_prefix=%.4f,mean_draft_s=%.6f\n",
                   parallel_proposed, parallel_accepted, cycles ? (double)parallel_accepted / cycles : 0,
                   cycles ? draft_s / cycles : 0);
        printf("spec_summary,user_sequences=1,cycles=%ld,emitted=%zu,decode_tokens=%ld,mean_emitted_per_cycle=%.3f,draft_acceptance=%.3f,mtp_draft_acceptance=%.3f,mtp_accepted=%ld,mtp_draft=%ld,eagle_draft_acceptance=%.3f,eagle_accepted=%ld,eagle_draft=%ld,pld_draft_acceptance=%.3f,pld_accepted=%ld,pld_draft=%ld,mean_verify_s=%.4f,%s=%.4f,mean_repair_s=%.4f,mean_cycle_s=%.4f,decode_s=%.4f,tok_per_s=%.3f,prefill_s=%.4f,request_tok_per_s=%.3f,min_free_vram_MiB=%ld\n",
               cycles, emitted.size(), tot_emitted, cycles ? (double)tot_emitted / cycles : 0, tot_draft ? (double)tot_accepted / tot_draft : 0,
               mtp_draft ? (double)mtp_accepted / mtp_draft : 0, mtp_accepted, mtp_draft,
               eagle_draft ? (double)eagle_accepted / eagle_draft : 0, eagle_accepted, eagle_draft,
               pld_draft ? (double)pld_accepted / pld_draft : 0, pld_accepted, pld_draft,
               cycles ? verify_s / cycles : 0, parallel_on ? "mean_parallel_s" : eagle_on ? "mean_eagle_s" : "mean_mtp_s",
               cycles ? draft_s / cycles : 0, cycles ? repair_s / cycles : 0,
               cycles ? decode_s / cycles : 0, decode_s, decode_s > 0 ? (double)tot_emitted / decode_s : 0,
               prefill_s, (double)emitted.size() / (prefill_s + decode_s), free_min);
        // E167: draft-length distribution (owner gate metric — expected accepted prefix
        // is the conditional-acceptance sum, so the length distribution is what matters)
        if (!draft_hist.empty()) {
            const char* names[7] = {"1-4", "5-8", "9-16", "17-24", "25-32", "33-48", "49-64"};
            long buckets[7] = {0};
            long sum_len = 0, max_len = 0;
            for (long d : draft_hist) {
                ++buckets[d <= 4 ? 0 : d <= 8 ? 1 : d <= 16 ? 2 : d <= 24 ? 3 : d <= 32 ? 4 : d <= 48 ? 5 : 6];
                sum_len += d; if (d > max_len) max_len = d;
            }
            printf("draft_hist,cycles=%zu,mean_len=%.2f,max_len=%ld", draft_hist.size(),
                   (double)sum_len / (double)draft_hist.size(), max_len);
            for (int i = 0; i < 7; ++i) printf(",%s=%ld", names[i], buckets[i]);
            printf("\n");
        }
        if (out_ids_path.size()) {
            FILE* f = fopen(out_ids_path.c_str(), "w");
            if (f) { for (long id : emitted) fprintf(f, "%ld\n", id); fclose(f); }
        }
        // --expect-ids: lossless gate — diff the speculative stream against a dense greedy
        // reference saved with --out-ids (same prompt, same --gen-tokens). Exit 1 on drift.
        if (expect_ids_path.size()) {
            std::vector<long> expect;
            if (FILE* f = fopen(expect_ids_path.c_str(), "r")) {
                long id; while (fscanf(f, "%ld", &id) == 1) expect.push_back(id);
                fclose(f);
            } else { fprintf(stderr, "cannot open --expect-ids %s\n", expect_ids_path.c_str()); return 2; }
            const size_t ncmp = std::min(expect.size(), emitted.size());
            size_t bad = ncmp, bad_at = 0;
            for (size_t i = 0; i < ncmp; ++i) if (expect[i] != emitted[i]) { bad_at = i; --bad; break; }
            const bool clean = bad == ncmp && expect.size() == emitted.size();
            printf("spec_lossless,expected=%zu,emitted=%zu,compared=%zu,mismatches=%zu,first_mismatch=%zu,match=%s\n",
                   expect.size(), emitted.size(), ncmp, ncmp - bad, bad == ncmp ? 0 : bad_at,
                   clean ? "yes" : "no");
            if (!clean) {
                fprintf(stderr, "LOSSLESS VIOLATION: speculative stream diverges from dense greedy reference\n");
                return 1;
            }
        }
        if (session) {
            if (session_tail) committed_ids.assign(hist.begin(), hist.begin() + cur_pos);
            printf("session_turn,reused_tokens=%zu,completion=%s\n", reused, stop ? "eos" : "length");
            printf("session_ids");
            for (long id : emitted) printf(",%ld", id);
            printf("\n");
            fflush(stdout);
        }
        return 0;
    };
    if (speculate > 0) {
        int status = run_request(prompts[0], first_draft, first_gen_tokens);
        if (status || !session) return status;
        for (std::string line; std::getline(std::cin, line); ) {
            std::vector<long> request, hint;
            long request_tokens;
            if (!parse_session_request(line, request, hint, request_tokens)) return 2;
            status = run_request(request, hint, request_tokens);
            if (status) return status;
        }
        return 0;
    }
    for (size_t i = 0; i < max_plen; ++i) {
        std::vector<long> toks(batch);
        for (long v = 0; v < batch; ++v) toks[v] = i < plen[v] ? prompts[v][i] : prompts[v].back();
        double s; uint64_t cold;
        run_token(toks, batch, (int)i, &s, &cold);
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
        // Only this dense generation call writes records. Prefill leaves no records.
        run_token(toks, batch, pass, &s, &cold, false, false, -1, capture_file != nullptr);
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
    if (capture_file) {
        if (fclose(capture_file) != 0) { perror(capture_path); return 1; }
        check(cudaFreeHost(capture_host), "free capture host");
        check(cudaFree(capture_dev), "free capture dev");
    }
    // stash generated ids of sequence 0 for the summary/text
    generated = gen_ids[0];
    all_ids = prompts[0];
    all_ids.insert(all_ids.end(), gen_ids[0].begin(), gen_ids[0].end());

    // ---- summary ----
    if (out_ids_path.size()) {
        FILE* f = fopen(out_ids_path.c_str(), "w");
        if (f) {
            // E169: dense batch mode writes ONE comma-terminated line per sequence
            // (capture training needs every member's stream); single-sequence modes
            // keep the one-id-per-line gate format.
            if (batch > 1 && !multi && !speculate) {
                for (long v = 0; v < batch; ++v) {
                    for (long id : gen_ids[v]) fprintf(f, "%ld,", id);
                    fprintf(f, "\n");
                }
            } else {
                for (long id : generated) fprintf(f, "%ld\n", id);
            }
            fclose(f);
        }
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
