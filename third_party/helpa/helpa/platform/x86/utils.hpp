#pragma once

#include "helpa/common.hpp"

#if defined(USE_X86)

#include <immintrin.h>

namespace helpa {

HELPA_INLINE inline float
reduce_add_f32x8(__m256 x) {
    auto sumh = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
    auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
}

HELPA_INLINE inline int32_t
reduce_add_i32x8(__m256i x) {
    auto sumh = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    auto tmp2 = _mm_hadd_epi32(sumh, sumh);
    return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

HELPA_INLINE inline int32_t
reduce_add_i16x16(__m256i x) {
    auto sumh = _mm_add_epi16(_mm256_extracti128_si256(x, 0), _mm256_extracti128_si256(x, 1));
    auto tmp = _mm256_cvtepi16_epi32(sumh);
    auto sumhh = _mm_add_epi32(_mm256_extracti128_si256(tmp, 0), _mm256_extracti128_si256(tmp, 1));
    auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
    return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

HELPA_INLINE inline __m256i
cvti4x32_i8x32(__m128i x) {
    auto mask = _mm_set1_epi8(0x0f);
    auto lo = _mm_and_si128(x, mask);
    auto hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
    auto loo = _mm256_cvtepu8_epi16(lo);
    auto hii = _mm256_cvtepu8_epi16(hi);
    hii = _mm256_slli_si256(hii, 1);
    auto ret = _mm256_or_si256(loo, hii);
    ret = _mm256_slli_epi64(ret, 3);
    return ret;
}

HELPA_INLINE inline __m256i
dp_u8s8x32(__m256i sum, __m256i x, __m256i y) {
#if defined(__AVX512VNNI__)
    return _mm256_dpbusd_epi32(sum, x, y);
#else
    const __m256i ones = _mm256_set1_epi16(1);
    auto tmp = _mm256_maddubs_epi16(x, y);
    return _mm256_add_epi32(sum, _mm256_madd_epi16(tmp, ones));
#endif
}

#if defined(USE_AVX512)

HELPA_INLINE inline float
reduce_add_f32x16(__m512 x) {
    return _mm512_reduce_add_ps(x);
}

HELPA_INLINE inline int32_t
reduce_add_i32x16(__m512i x) {
    return _mm512_reduce_add_epi32(x);
}

HELPA_INLINE inline __m512i
cvti4x64_i8x64(__m256i x) {
    auto mask = _mm256_set1_epi8(0x0f);
    auto lo = _mm256_and_si256(x, mask);
    auto hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask);
    auto loo = _mm512_cvtepu8_epi16(lo);
    auto hii = _mm512_cvtepu8_epi16(hi);
    hii = _mm512_slli_epi16(hii, 8);
    auto ret = _mm512_or_si512(loo, hii);
    ret = _mm512_slli_epi64(ret, 3);
    return ret;
}

HELPA_INLINE inline __m512i
dp_u8s8x64(__m512i sum, __m512i x, __m512i y) {
#if defined(__AVX512VNNI__)
    // GCC bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94663
    // sum = _mm512_dpbusd_epi32(sum, x, y);
    asm("vpdpbusd %1, %2, %0" : "+x"(sum) : "mx"(x), "x"(y));
    return sum;
#else
    const __m512i ones = _mm512_set1_epi16(1);
    auto tmp = _mm512_maddubs_epi16(x, y);
    return _mm512_add_epi32(sum, _mm512_madd_epi16(tmp, ones));
#endif
}

#endif

#if defined(__AVX512FP16__) && defined(USE_AVX512FP16)

HELPA_INLINE inline float
reduce_add_f16x32(__m512h x) {
    return _mm512_reduce_add_ph(x);
}

#endif

}  // namespace helpa

#endif
