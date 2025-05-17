#pragma once

#include "helpa/common.hpp"

#if defined(USE_X86)

#include <immintrin.h>

#include "helpa/dot.hpp"
#include "helpa/platform/x86/utils.hpp"
#include "helpa/ref/dot_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline float
dot_fp32_fp32(const float* x, const float* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_fp32(x, y, da) + dot_fp32_fp32_ref(x + da, y + da, d - da);
}

inline float
dot_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_fp16(x, y, da) + dot_fp32_fp16_ref(x + da, y + da, d - da);
}

inline float
dot_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return dota_fp16_fp16(x, y, da) + dot_fp16_fp16_ref(x + da, y + da, d - da);
}

inline float
dot_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_bf16(x, y, da) + dot_fp32_bf16_ref(x + da, y + da, d - da);
}

inline float
dot_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return dota_bf16_bf16(x, y, da) + dot_bf16_bf16_ref(x + da, y + da, d - da);
}

inline int32_t
dot_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    int32_t da = d / 64 * 64;
    return dota_s8_s8(x, y, da) + dot_s8_s8_ref(x + da, y + da, d - da);
}

inline float
dota_fp32_fp32(const float* x, const float* y, const int32_t d) {
#if defined(USE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    for (int32_t i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto yy = _mm512_loadu_ps(y + i);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    for (int32_t i = 0; i < d; i += 8) {
        auto xx = _mm256_loadu_ps(x + i);
        auto yy = _mm256_loadu_ps(y + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
    }
    return -reduce_add_f32x8(sum);
#endif
}

inline float
dota_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
#if defined(USE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtph_ps(zz);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        {
            auto xx = _mm256_loadu_ps(x + i);
            auto zz = _mm_loadu_si128((__m128i*)(y + i));
            auto yy = _mm256_cvtph_ps(zz);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, yy));
        }
        {
            auto xx = _mm256_loadu_ps(x + i + 8);
            auto zz = _mm_loadu_si128((__m128i*)(y + i + 8));
            auto yy = _mm256_cvtph_ps(zz);
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(xx, yy));
        }
    }
    sum1 = _mm256_add_ps(sum1, sum2);
    return -reduce_add_f32x8(sum1);
#endif
}

inline float
dota_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
#if defined(USE_AVX512FP16) && defined(__AVX512FP16__)
    auto sum = _mm512_setzero_ph();
    for (int i = 0; i < d; i += 32) {
        auto xx = _mm512_loadu_ph(x + i);
        auto yy = _mm512_loadu_ph(y + i);
        sum = _mm512_fmadd_ph(xx, yy, sum);
    }
    return -reduce_add_f16x32(sum);
#elif defined(USE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xxx = _mm256_loadu_si256((__m256i*)(x + i));
        auto xx = _mm512_cvtph_ps(xxx);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtph_ps(zz);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        auto xxx = _mm_loadu_si128((__m128i*)(x + i));
        auto xx = _mm256_cvtph_ps(xxx);
        auto zz = _mm_loadu_si128((__m128i*)(y + i));
        auto yy = _mm256_cvtph_ps(zz);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
    }
    return -reduce_add_f32x8(sum);
#endif
}

inline float
dota_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
#if defined(USE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        sum = _mm512_fmadd_ps(xx, (__m512)yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        {
            auto xx = _mm256_loadu_ps(x + i);
            auto zz = _mm_loadu_si128((__m128i*)(y + i));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, (__m256)yy));
        }
        {
            auto xx = _mm256_loadu_ps(x + i + 8);
            auto zz = _mm_loadu_si128((__m128i*)(y + i + 8));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, (__m256)yy));
        }
    }
    sum1 = _mm256_add_ps(sum1, sum2);
    return -reduce_add_f32x8(sum1);
#endif
}

inline float
dota_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
#if defined(USE_AVX512BF16) && defined(__AVX512BF16__)
    auto sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 32) {
        auto xx = (__m512bh)_mm512_loadu_si512(x + i);
        auto yy = (__m512bh)_mm512_loadu_si512(y + i);
        sum = _mm512_dpbf16_ps(sum, xx, yy);
    }
    return -reduce_add_f32x16(sum);
#elif defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xxx = _mm256_loadu_si256((__m256i*)(x + i));
        auto xx = _mm512_cvtepu16_epi32(xxx);
        xx = _mm512_slli_epi32(xx, 16);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        sum = _mm512_fmadd_ps((__m512)xx, (__m512)yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        auto xxx = _mm_loadu_si128((__m128i*)(x + i));
        auto xx = _mm256_cvtepu16_epi32(xxx);
        xx = _mm256_slli_epi32(xx, 16);
        auto zz = _mm_loadu_si128((__m128i*)(y + i));
        auto yy = _mm256_cvtepu16_epi32(zz);
        yy = _mm256_slli_epi32(yy, 16);
        sum = _mm256_add_ps(sum, _mm256_mul_ps((__m256)xx, (__m256)yy));
    }
    return -reduce_add_f32x8(sum);
#endif
}

inline int32_t
dota_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
#if defined(USE_VNNI)
    auto sum = _mm256_setzero_si256();
    for (int i = 0; i < d; i += 32) {
        auto xx = _mm256_loadu_si256((__m256i*)(x + i));
        auto yy = _mm256_loadu_si256((__m256i*)(y + i));
        auto axx = _mm256_sign_epi8(xx, xx);
        auto syy = _mm256_sign_epi8(yy, xx);
        sum = dp_u8s8x32(sum, axx, syy);
    }
    return -reduce_add_i32x8(sum);
#else
    int32_t sum = 0;
    for (int i = 0; i < d; ++i) {
        sum += int32_t(x[i]) * int32_t(y[i]);
    }
    return -sum;
#endif
}

inline int32_t
dota_u8_s8(const uint8_t* x, const int8_t* y, const int32_t d) {
#if defined(USE_VNNI)
    __m512i sum = _mm512_setzero_epi32();
    for (int i = 0; i < d; i += 64) {
        auto xx = _mm512_loadu_si512(x + i);
        auto yy = _mm512_loadu_si512(y + i);
        sum = dp_u8s8x64(sum, xx, yy);
    }
    return -reduce_add_i32x16(sum);
#else
    __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
    for (int i = 0; i < d; i += 64) {
        {
            auto xx = _mm256_loadu_si256((__m256i*)(x + i));
            auto yy = _mm256_loadu_si256((__m256i*)(y + i));
            auto tmp = _mm256_maddubs_epi16(xx, yy);
            sum1 = _mm256_add_epi32(sum1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0)));
            sum1 = _mm256_add_epi32(sum1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 1)));
        }
        {
            auto xx = _mm256_loadu_si256((__m256i*)(x + i + 32));
            auto yy = _mm256_loadu_si256((__m256i*)(y + i + 32));
            auto tmp = _mm256_maddubs_epi16(xx, yy);
            sum2 = _mm256_add_epi32(sum2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0)));
            sum2 = _mm256_add_epi32(sum2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 1)));
        }
    }
    sum1 = _mm256_add_epi32(sum1, sum2);
    return -reduce_add_i32x8(sum1);
#endif
}

}  // namespace helpa

#endif
