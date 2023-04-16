#pragma once

#if defined(__AVX512F__)

#include <cstdint>
#include <immintrin.h>

namespace glass {

inline float reduce_add_f32x16(__m512 x) {
  auto sumh =
      _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
  auto sumhh =
      _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
  auto tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

inline int32_t reduce_add_i32x16(__m512i x) {
  auto sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(x, 0),
                               _mm512_extracti32x8_epi32(x, 1));
  auto sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh),
                             _mm256_extracti128_si256(sumh, 1));
  auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

} // namespace glass

#endif