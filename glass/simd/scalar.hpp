#pragma once

#include "glass/types.hpp"

namespace glass {

inline float L2SqrE5M2(const float *x, const E5M2 *y, int d) {
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    auto dif = x[i] - float(y[i]);
    sum += dif * dif;
  }
  return sum;
}

inline float IPE5M2(const float *x, const E5M2 *y, int d) {
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    sum += x[i] * float(y[i]);
  }
  return -sum;
}

inline float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = (y[i] + 0.5f);
    yy = yy * dif[i] + mi[i] * 256.0f;
    auto dif = x[i] * 256.0f - yy;
    sum += dif * dif;
  }
  return sum;
}

inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = y[i] + 0.5f;
    yy = yy * dif[i] + mi[i] * 256.0f;
    sum += x[i] * yy;
  }
  return -sum;
}

inline float L2SqrSQ6_ext(const float * /**x*/, const uint8_t * /**y*/,
                          int /**d*/, const float * /**mi*/,
                          const float * /**dif*/) {

  // TODO
  return 0.0f;
}

inline float IPSQ6_ext(const float * /**x*/, const uint8_t * /**y*/, int /**d*/,
                       const float * /**mi*/, const float * /**dif*/) {
  // TODO
  return 0.0f;
}

inline float L2SqrSQ4_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
  float sum = 0.0f;
  for (int i = 0; i < d; i += 2) {
    {
      float yy = (y[i / 2] & 15) + 0.5f;
      yy = yy * dif[i] + mi[i] * 16.0f;
      auto dif = x[i] * 16.0f - yy;
      sum += dif * dif;
    }
    {
      float yy = (y[i / 2] >> 4 & 15) + 0.5f;
      yy = yy * dif[i + 1] + mi[i + 1] * 16.0f;
      auto dif = x[i + 1] * 16.0f - yy;
      sum += dif * dif;
    }
  }
  return sum;
}

inline float IPSQ4_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {
  float sum = 0.0f;
  for (int i = 0; i < d; i += 2) {
    {
      float yy = ((y[i / 2] & 15) + 0.5f) / 16.0f;
      yy = yy * dif[i] + mi[i];
      sum += x[i] * yy;
    }
    {
      float yy = ((y[i / 2] >> 4 & 15) + 0.5f) / 16.0f;
      yy = yy * dif[i + 1] + mi[i + 1];
      sum += x[i + 1] * yy;
    }
  }
  return -sum;
}

inline int32_t L2SqrSQ8SQ4(const uint8_t *x, const uint8_t *y, int d) {
  int32_t sum = 0;
  for (int i = 0; i < d; i += 2) {
    {
      int32_t xx = x[i];
      int32_t yy = (y[i / 2] & 15) * 8 + 4;
      sum += (xx - yy) * (xx - yy);
    }
    {
      int32_t xx = x[i + 1];
      int32_t yy = (y[i / 2] >> 4 & 15) * 8 + 4;
      sum += (xx - yy) * (xx - yy);
    }
  }
  return sum;
}

inline int32_t L2SqrSQ2(const uint8_t *x, const uint8_t *y, int d) {
  int32_t sum = 0;
  for (int i = 0; i < d; i += 4) {
    uint8_t xx0 = x[i / 4] & 3;
    uint8_t xx1 = x[i / 4] >> 2 & 3;
    uint8_t xx2 = x[i / 4] >> 4 & 3;
    uint8_t xx3 = x[i / 4] >> 6 & 3;
    uint8_t yy0 = y[i / 4] & 3;
    uint8_t yy1 = y[i / 4] >> 2 & 3;
    uint8_t yy2 = y[i / 4] >> 4 & 3;
    uint8_t yy3 = y[i / 4] >> 6 & 3;
    uint8_t d0 = xx0 - yy0;
    uint8_t d1 = xx1 - yy1;
    uint8_t d2 = xx2 - yy2;
    uint8_t d3 = xx3 - yy3;
    sum += d0 * d0;
    sum += d1 * d1;
    sum += d2 * d2;
    sum += d3 * d3;
  }
  return sum;
}

inline int32_t L2SqrSQ1(const uint8_t *x, const uint8_t *y, int d) {
  auto xx = (const uint64_t *)x;
  auto yy = (const uint64_t *)y;
  const uint64_t *end = xx + d / 64;
  int32_t sum = 0;
  while (xx < end) {
    sum += __builtin_popcountll(*xx ^ *yy);
    xx += 1;
    yy += 1;
  }
  return sum;
}

} // namespace glass
