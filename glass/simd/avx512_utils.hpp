#pragma once

#include "glass/common.hpp"
#if defined(__AVX512F__)

#include <cstdint>
#include <immintrin.h>

namespace glass {

GLASS_INLINE inline float reduce_add_f32x16(__m512 x) {
  auto sumh =
      _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
  auto sumhh =
      _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
  auto tmp1 = _mm_hadd_ps(sumhh, sumhh);
  return tmp1[0] + tmp1[1];
  // return _mm_extract_ps(tmp2, 0) + _mm_extract_ps(tmp2, 1);
  // auto tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
  // return tmp1[0] + tmp1[1];
  // auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  // return _mm_cvtss_f32(tmp2);
}

GLASS_INLINE inline int32_t reduce_add_i32x16(__m512i x) {
  auto sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(x, 0),
                               _mm512_extracti32x8_epi32(x, 1));
  auto sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh),
                             _mm256_extracti128_si256(sumh, 1));
  auto tmp1 = _mm_hadd_epi32(sumhh, sumhh);
  return _mm_extract_epi32(tmp1, 0) + _mm_extract_epi32(tmp1, 1);
}

#if defined(USE_AVX512FP16)

GLASS_INLINE inline float reduce_add_f16x32(__m512h x) {
  return _mm512_reduce_add_ph(x);
}

#endif

GLASS_INLINE inline __m512i cvti4x64_i8x64(__m256i x) {
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

} // namespace glass

#endif
