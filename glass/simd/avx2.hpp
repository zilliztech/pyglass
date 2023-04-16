#pragma once

#if defined(__AVX2__)

#include <cstdint>
#include <immintrin.h>

namespace glass {

inline float reduce_add_f32x8(__m256 x) {
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

inline int32_t reduce_add_i32x8(__m256i x) {
  auto sumh =
      _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
  auto tmp2 = _mm_hadd_epi32(sumh, sumh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

inline int32_t reduce_add_i16x16(__m256i x) {
  auto sumh = _mm_add_epi16(_mm256_extracti128_si256(x, 0),
                            _mm256_extracti128_si256(x, 1));
  auto tmp = _mm256_cvtepi16_epi32(sumh);
  auto sumhh = _mm_add_epi32(_mm256_extracti128_si256(tmp, 0),
                             _mm256_extracti128_si256(tmp, 1));
  auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

} // namespace glass

#endif