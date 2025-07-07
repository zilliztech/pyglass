#pragma once

#include <cstdint>
#include <cstdio>
#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "glass/common.hpp"
#include "glass/simd/avx2.hpp"
#include "glass/simd/avx512.hpp"

namespace glass {

template <typename T1, typename T2, typename U, typename... Params>
using Dist = U (*)(const T1 *, const T2 *, int, Params...);

GLASS_INLINE inline void prefetch_L1(const void *address) {
#if defined(__SSE2__)
    _mm_prefetch((const char *)address, _MM_HINT_T0);
#else
    __builtin_prefetch(address, 0, 3);
#endif
}

GLASS_INLINE inline void prefetch_L2(const void *address) {
#if defined(__SSE2__)
    _mm_prefetch((const char *)address, _MM_HINT_T1);
#else
    __builtin_prefetch(address, 0, 2);
#endif
}

GLASS_INLINE inline void prefetch_L3(const void *address) {
#if defined(__SSE2__)
    _mm_prefetch((const char *)address, _MM_HINT_T2);
#else
    __builtin_prefetch(address, 0, 1);
#endif
}

inline void mem_prefetch(char *ptr, const int num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 28:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 27:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 26:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 25:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 24:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 23:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 22:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 21:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 20:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_L1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}

FAST_BEGIN
inline float L2SqrRef(const float *x, const float *y, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sum;
}
FAST_END

FAST_BEGIN
inline float IPRef(const float *x, const float *y, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}
FAST_END

inline float L2Sqr(const float *x, const float *y, int d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    const float *end = x + d;
    while (x < end) {
        auto xx = _mm512_loadu_ps(x);
        x += 16;
        auto yy = _mm512_loadu_ps(y);
        y += 16;
        auto t = _mm512_sub_ps(xx, yy);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(t, t));
    }
    return reduce_add_f32x16(sum);
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    const float *end = x + d;
    while (x < end) {
        auto xx = _mm256_loadu_ps(x);
        x += 8;
        auto yy = _mm256_loadu_ps(y);
        y += 8;
        auto t = _mm256_sub_ps(xx, yy);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
    }
    return reduce_add_f32x8(sum);
#elif defined(__aarch64__)
    float32x4_t sum = vdupq_n_f32(0);
    for (int32_t i = 0; i < d; i += 4) {
        auto xx = vld1q_f32(x + i);
        auto yy = vld1q_f32(y + i);
        auto t = vsubq_f32(xx, yy);
        sum = vmlaq_f32(sum, t, t);
    }
    return vaddvq_f32(sum);
#else
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sum;
#endif
}

inline float IP(const float *x, const float *y, int d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    const float *end = x + d;
    while (x < end) {
        auto xx = _mm512_loadu_ps(x);
        x += 16;
        auto yy = _mm512_loadu_ps(y);
        y += 16;
        sum = _mm512_add_ps(sum, _mm512_mul_ps(xx, yy));
    }
    return -reduce_add_f32x16(sum);
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    const float *end = x + d;
    while (x < end) {
        auto xx = _mm256_loadu_ps(x);
        x += 8;
        auto yy = _mm256_loadu_ps(y);
        y += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
    }
    return -reduce_add_f32x8(sum);
#elif defined(__aarch64__)
    float32x4_t sum = vdupq_n_f32(0);
    for (int32_t i = 0; i < d; i += 4) {
        auto xx = vld1q_f32(x + i);
        auto yy = vld1q_f32(y + i);
        sum = vmlaq_f32(sum, xx, yy);
    }
    return vaddvq_f32(sum);
#else
    float sum = 0.0;
    for (int i = 0; i < d; ++i) {
        sum += x[i] * y[i];
    }
    return -sum;
#endif
}

inline float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi, const float *dif) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    __m512 dot5 = _mm512_set1_ps(0.5f);
    __m512 const_255 = _mm512_set1_ps(255.0f);
    for (int i = 0; i < d; i += 16) {
        auto zz = _mm_loadu_epi8(y + i);
        auto zzz = _mm512_cvtepu8_epi32(zz);
        auto yy = _mm512_cvtepi32_ps(zzz);
        yy = _mm512_add_ps(yy, dot5);
        auto mi512 = _mm512_loadu_ps(mi + i);
        auto dif512 = _mm512_loadu_ps(dif + i);
        yy = _mm512_mul_ps(yy, dif512);
        yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
        auto xx = _mm512_loadu_ps(x + i);
        auto d = _mm512_sub_ps(_mm512_mul_ps(xx, const_255), yy);
        sum = _mm512_fmadd_ps(d, d, sum);
    }
    return reduce_add_f32x16(sum);
#else
    float sum = 0.0;
    for (int i = 0; i < d; ++i) {
        float yy = (y[i] + 0.5f);
        yy = yy * dif[i] + mi[i] * 255.0f;
        auto dif = x[i] * 255.0f - yy;
        sum += dif * dif;
    }
    return sum;
#endif
}

inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi, const float *dif) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    __m512 dot5 = _mm512_set1_ps(0.5f);
    __m512 const_255 = _mm512_set1_ps(255.0f);
    for (int i = 0; i < d; i += 16) {
        auto zz = _mm_loadu_epi8(y + i);
        auto zzz = _mm512_cvtepu8_epi32(zz);
        auto yy = _mm512_cvtepi32_ps(zzz);
        yy = _mm512_add_ps(yy, dot5);
        auto mi512 = _mm512_loadu_ps(mi + i);
        auto dif512 = _mm512_loadu_ps(dif + i);
        yy = _mm512_mul_ps(yy, dif512);
        yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
        auto xx = _mm512_loadu_ps(x + i);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    float sum = 0.0;
    for (int i = 0; i < d; ++i) {
        float yy = y[i] + 0.5f;
        yy = yy * dif[i] + mi[i] * 255.0f;
        sum += x[i] * yy;
    }
    return -sum;
#endif
}

inline int32_t L2SqrSQ4(const uint8_t *x, const uint8_t *y, int d) {
#if defined(__AVX2__)
    __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
    __m256i mask = _mm256_set1_epi8(0xf);
    for (int i = 0; i < d; i += 64) {
        auto xx = _mm256_loadu_si256((__m256i *)(x + i / 2));
        auto yy = _mm256_loadu_si256((__m256i *)(y + i / 2));
        auto xx1 = _mm256_and_si256(xx, mask);
        auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
        auto yy1 = _mm256_and_si256(yy, mask);
        auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
        auto d1 = _mm256_sub_epi8(xx1, yy1);
        auto d2 = _mm256_sub_epi8(xx2, yy2);
        d1 = _mm256_abs_epi8(d1);
        d2 = _mm256_abs_epi8(d2);
        sum1 = _mm256_add_epi16(sum1, _mm256_maddubs_epi16(d1, d1));
        sum2 = _mm256_add_epi16(sum2, _mm256_maddubs_epi16(d2, d2));
    }
    sum1 = _mm256_add_epi32(sum1, sum2);
    return reduce_add_i16x16(sum1);
#else
    int32_t sum = 0;
    for (int i = 0; i < d; ++i) {
        {
            int32_t xx = x[i / 2] & 15;
            int32_t yy = y[i / 2] & 15;
            sum += (xx - yy) * (xx - yy);
        }
        {
            int32_t xx = x[i / 2] >> 4 & 15;
            int32_t yy = y[i / 2] >> 4 & 15;
            sum += (xx - yy) * (xx - yy);
        }
    }
    return sum;
#endif
}

}  // namespace glass
