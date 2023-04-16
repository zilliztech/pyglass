#pragma once

#include <cstdint>
#include <cstdio>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

#include "glass/common.hpp"
#include "glass/simd/avx2.hpp"
#include "glass/simd/avx512.hpp"

namespace glass {

template <typename T1, typename T2, typename U, typename... Params>
using Dist = U (*)(const T1 *, const T2 *, int, Params...);

inline void mem_prefetch(char *ptr, const int num_lines) {
  switch (num_lines) {
  default:
    [[fallthrough]];
  case 28:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 27:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 26:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 25:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 24:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 23:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 22:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 21:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 20:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 19:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 18:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 17:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 16:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 15:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 14:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 13:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 12:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 11:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 10:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 9:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 8:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 7:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 6:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 5:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 4:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 3:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 2:
    _mm_prefetch(ptr, _MM_HINT_T0);
    ptr += 64;
    [[fallthrough]];
  case 1:
    _mm_prefetch(ptr, _MM_HINT_T0);
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
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
}
FAST_END

FAST_BEGIN
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
#else
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return sum;
#endif
}
FAST_END

FAST_BEGIN
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
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    sum += x[i] * y[i];
  }
  return -sum;
#endif
}
FAST_END

FAST_BEGIN
inline float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = (y[i] + 0.5f) / 255.0f;
    yy = yy * dif[i] + mi[i];
    sum += (x[i] - yy) * (x[i] - yy);
  }
  return sum;
}
FAST_END

FAST_BEGIN
inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = (y[i] + 0.5f) / 255.0f;
    yy = yy * dif[i] + mi[i];
    sum += x[i] * yy;
  }
  return -sum;
}
FAST_END

} // namespace glass
