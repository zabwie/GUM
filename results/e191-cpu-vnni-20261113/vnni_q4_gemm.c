// E191: CPU second-stream feasibility gate.
// Tensor: ffn_down (N=5120 outputs, K=17408), 32 verify rows, Q4_0 weights.
//   contribution_block = d * (sum((q-8)*a) ) * ascale   [Q4_0 x int8-activation]
// Variant A (scalar): unpack per row — reference floor.
// Variant B (AVX2): unpack once per block, maddubs+biased trick, apply to 32 rows.
//   sum((q-8)*a) = sum(q_u8*a) - 8*sum(a); the row's per-block sum(a) precomputes once.
// Gate: variant B >= 1 TFLOP/s.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#define N_OUT 5120
#define K_IN  17408
#define ROWS  32
#define BLK   32
#define NB    (K_IN / BLK)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static inline float fp16_to_f32(uint16_t h) {
    const float s = (h & 0x8000) ? -1.f : 1.f;
    int e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float val;
    if (e == 0) val = m * (1.f / 16384.f);
    else if (e < 31) val = (1.f + m / 1024.f) * (1 << (e - 15));
    else val = m ? 0.f : 1e30f;
    return s * val;
}

int main(int argc, char** argv) {
    int variant = argc > 1 ? atoi(argv[1]) : 2;
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    size_t wbytes = (size_t)N_OUT * NB * 18;
    uint8_t* W = aligned_alloc(64, wbytes);
    int8_t* A = aligned_alloc(64, (size_t)ROWS * K_IN);
    int16_t* Ascale = aligned_alloc(64, (size_t)ROWS * NB * sizeof(int16_t));
    int32_t* Asum = aligned_alloc(64, (size_t)ROWS * NB * sizeof(int32_t));
    float* Y = calloc((size_t)ROWS * N_OUT, sizeof(float));
    srand(7);
    for (size_t i = 0; i < wbytes; i++) W[i] = (uint8_t)rand();
    for (size_t i = 0; i < (size_t)ROWS * K_IN; i++) A[i] = (int8_t)(rand() % 33 - 16);
    for (size_t i = 0; i < (size_t)ROWS * NB; i++) {
        Ascale[i] = 100 + rand() % 200;
        int32_t s = 0;
        for (int j = 0; j < 32; j++) s += A[(size_t)(i / NB) * K_IN + (i % NB) * 32 + j];
        Asum[i] = s;
    }

    double t0 = now_s();
    if (variant == 1) {
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N_OUT; n++) {
            const uint8_t* wr = W + (size_t)n * NB * 18;
            for (int r = 0; r < ROWS; r++) {
                float acc = 0.f;
                for (int b = 0; b < NB; b++) {
                    float d = fp16_to_f32(*(const uint16_t*)(wr + (size_t)b * 18));
                    int32_t s = 0;
                    const uint8_t* p = wr + (size_t)b * 18 + 2;
                    for (int j = 0; j < 16; j++) {
                        s += ((p[j] & 0xF) - 8) * A[(size_t)r * K_IN + b * 32 + 2 * j];
                        s += ((p[j] >> 4) - 8) * A[(size_t)r * K_IN + b * 32 + 2 * j + 1];
                    }
                    acc += d * (float)s * (float)Ascale[(size_t)r * NB + b];
                }
                Y[(size_t)r * N_OUT + n] = acc;
            }
        }
    } else if (variant == 2 || variant == 4) {
        uint8_t* wq = aligned_alloc(64, (size_t)N_OUT * NB * 32);
        float* dv = aligned_alloc(64, (size_t)N_OUT * NB * sizeof(float));
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N_OUT; n++) {
            const uint8_t* wr = W + (size_t)n * NB * 18;
            uint8_t* wqn = wq + (size_t)n * NB * 32;
            float* dvn = dv + (size_t)n * NB;
            for (int b = 0; b < NB; b++) {
                dvn[b] = fp16_to_f32(*(const uint16_t*)(wr + (size_t)b * 18));
                const uint8_t* p = wr + (size_t)b * 18 + 2;
                for (int j = 0; j < 16; j++) {
                    wqn[(size_t)b * 32 + 2 * j]     = (p[j] & 0xF) + 8;
                    wqn[(size_t)b * 32 + 2 * j + 1] = (p[j] >> 4) + 8;
                }
            }
            float acc[ROWS];
            for (int r = 0; r < ROWS; r++) acc[r] = 0.f;
            for (int b = 0; b < NB; b++) {
                __m256i wv = _mm256_loadu_si256((const __m256i*)(wqn + (size_t)b * 32));
                float d = dvn[b];
                int32_t sv[ROWS];
                for (int r = 0; r < ROWS; r++) {
                    // maddubs: first arg UNSIGNED (wq 0..15), second SIGNED (a -16..16)
                    __m256i lo = _mm256_maddubs_epi16(wv,
                        _mm256_loadu_si256((const __m256i*)(A + (size_t)r * K_IN + b * 32)));
                    __m128i s16 = _mm_add_epi16(_mm256_castsi256_si128(lo), _mm256_extracti128_si256(lo, 1));
                    __m128i s32 = _mm_madd_epi16(s16, _mm_set1_epi16(1));
                    s32 = _mm_add_epi32(s32, _mm_shuffle_epi32(s32, _MM_SHUFFLE(1, 0, 3, 2)));
                    s32 = _mm_add_epi32(s32, _mm_shuffle_epi32(s32, _MM_SHUFFLE(2, 3, 0, 1)));
                    sv[r] = _mm_cvtsi128_si32(s32);  // full 32-product sum
                }
                if (variant == 2) {
                    for (int r = 0; r < ROWS; r++)
                        acc[r] += d * (float)(sv[r] - 16 * Asum[(size_t)r * NB + b]) * (float)Ascale[(size_t)r * NB + b];
                } else {
                    // vector tail across rows: acc[r] += (s-8*asum)*d*ascale
                    for (int r = 0; r < ROWS; r += 8) {
                        __m256 svv = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)&sv[r]));
                        __m256i asc = _mm256_loadu_si256((const __m256i*)&Asum[(size_t)r * NB + b]); // stride NB! not contiguous
                        (void)asc;
                        // Asum/Ascale are strided by NB across rows — gather scalar path
                        for (int rr = r; rr < r + 8 && rr < ROWS; rr++)
                            acc[rr] += (svv[0], 0.f); // placeholder no-op; keep scalar below
                        for (int rr = r; rr < r + 8 && rr < ROWS; rr++)
                            acc[rr] += d * (float)(sv[rr] - 16 * Asum[(size_t)rr * NB + b]) * (float)Ascale[(size_t)rr * NB + b];
                    }
                }
            }
            for (int r = 0; r < ROWS; r++) Y[(size_t)r * N_OUT + n] = acc[r];
        }
    } else if (variant == 3) {
        // AVX-VNNI: dpbusd(src, u8, s8): u8 = wq (q+8, 0..15), s8 = activations.
        // s = sum(wq*a) over 32; contribution = (s - 8*asum) * d * ascale.
        uint8_t* wq = aligned_alloc(64, (size_t)N_OUT * NB * 32);
        float* dv = aligned_alloc(64, (size_t)N_OUT * NB * sizeof(float));
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N_OUT; n++) {
            const uint8_t* wr = W + (size_t)n * NB * 18;
            uint8_t* wqn = wq + (size_t)n * NB * 32;
            float* dvn = dv + (size_t)n * NB;
            for (int b = 0; b < NB; b++) {
                dvn[b] = fp16_to_f32(*(const uint16_t*)(wr + (size_t)b * 18));
                const uint8_t* p = wr + (size_t)b * 18 + 2;
                for (int j = 0; j < 16; j++) {
                    wqn[(size_t)b * 32 + 2 * j]     = (p[j] & 0xF) + 8;
                    wqn[(size_t)b * 32 + 2 * j + 1] = (p[j] >> 4) + 8;
                }
            }
            float acc[ROWS];
            for (int r = 0; r < ROWS; r++) acc[r] = 0.f;
            for (int b = 0; b < NB; b++) {
                const __m128i wl = _mm_loadu_si128((const __m128i*)(wqn + (size_t)b * 32));
                const __m128i wh = _mm_loadu_si128((const __m128i*)(wqn + (size_t)b * 32 + 16));
                float d = dvn[b];
                int32_t sv[ROWS];
                for (int r = 0; r < ROWS; r++) {
                    const __m128i a0 = _mm_loadu_si128((const __m128i*)(A + (size_t)r * K_IN + b * 32));
                    const __m128i a1 = _mm_loadu_si128((const __m128i*)(A + (size_t)r * K_IN + b * 32 + 16));
                    __m128i s0 = _mm_dpbusd_epi32(_mm_setzero_si128(), wl, a0);
                    __m128i s1 = _mm_dpbusd_epi32(_mm_setzero_si128(), wh, a1);
                    __m128i s01 = _mm_add_epi32(s0, s1);
                    s01 = _mm_add_epi32(s01, _mm_shuffle_epi32(s01, _MM_SHUFFLE(1, 0, 3, 2)));
                    s01 = _mm_add_epi32(s01, _mm_shuffle_epi32(s01, _MM_SHUFFLE(2, 3, 0, 1)));
                    sv[r] = _mm_cvtsi128_si32(s01);  // full 32-product sum
                }
                for (int r = 0; r < ROWS; r++)
                    acc[r] += d * (float)(sv[r] - 16 * Asum[(size_t)r * NB + b]) * (float)Ascale[(size_t)r * NB + b];
            }
            for (int r = 0; r < ROWS; r++) Y[(size_t)r * N_OUT + n] = acc[r];
        }
        (void)Asum;
    } else {
        fprintf(stderr, "unknown variant\n"); return 2;
    }
    double dt = now_s() - t0;
    double flop = 2.0 * (double)N_OUT * K_IN * ROWS * reps;
    printf("variant=%d reps=%d time=%.3fs rate=%.3f TFLOP/s weight_rate=%.1f GiB/s\n",
           variant, reps, dt, flop / dt / 1e12, (double)reps * wbytes / dt / (1 << 30));
    // checksum so the compiler can't delete the work
    float chk = 0.f; for (size_t i = 0; i < (size_t)ROWS * N_OUT; i += 4096) chk += Y[i];
    printf("checksum=%.6f y0=%.9g y1=%.9g y2=%.9g\n", chk, Y[0], Y[N_OUT], Y[2*N_OUT]);
    return 0;
}
