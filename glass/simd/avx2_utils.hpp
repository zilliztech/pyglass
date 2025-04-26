#pragma once

#if defined(__AVX2__)

#include "glass/common.hpp"
#include <immintrin.h>

namespace glass {

GLASS_INLINE inline float reduce_add_f32x8(__m256 x) {
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

GLASS_INLINE inline int32_t reduce_add_i32x8(__m256i x) {
  auto sumh =
      _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
  auto tmp2 = _mm_hadd_epi32(sumh, sumh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

GLASS_INLINE inline int32_t reduce_add_i16x16(__m256i x) {
  auto sumh = _mm_add_epi16(_mm256_extracti128_si256(x, 0),
                            _mm256_extracti128_si256(x, 1));
  auto tmp = _mm256_cvtepi16_epi32(sumh);
  auto sumhh = _mm_add_epi32(_mm256_extracti128_si256(tmp, 0),
                             _mm256_extracti128_si256(tmp, 1));
  auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

namespace SQ6 {

GLASS_INLINE inline __m256i load6(const uint16_t *code16) {
  const __m128i perm =
      _mm_set_epi8(-1, 5, 5, 4, 4, 3, -1, 3, -1, 2, 2, 1, 1, 0, -1, 0);
  const __m256i shifts = _mm256_set_epi32(2, 4, 6, 0, 2, 4, 6, 0);

  // load 6 bytes
  __m128i c1 = _mm_set_epi16(0, 0, 0, 0, 0, code16[2], code16[1], code16[0]);

  // put in 8 * 32 bits
  __m128i c2 = _mm_shuffle_epi8(c1, perm);
  __m256i c3 = _mm256_cvtepi16_epi32(c2);

  // shift and mask out useless bits
  __m256i c4 = _mm256_srlv_epi32(c3, shifts);
  __m256i c5 = _mm256_and_si256(_mm256_set1_epi32(63), c4);
  return c5;
}

GLASS_INLINE inline __m256 decode_8_components(const uint8_t *code, int i) {
  __m256i i8 = load6((const uint16_t *)(code + (i >> 2) * 3));
  __m256 f8 = _mm256_cvtepi32_ps(i8);
  // this could also be done with bit manipulations but it is
  // not obviously faster
  __m256 half = _mm256_set1_ps(0.5f);
  f8 = _mm256_add_ps(f8, half);
  __m256 one_63 = _mm256_set1_ps(1.f / 63.f);
  return _mm256_mul_ps(f8, one_63);
}

} // namespace SQ6

GLASS_INLINE inline __m256i cvti4x32_i8x32(__m128i x) {
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

} // namespace glass

#endif