#pragma once

#if defined(__aarch64__)

#include <arm_neon.h>

#include "glass/types.hpp"

namespace glass {

inline float L2SqrE5M2(const float *x, const e5m2 *y, int d) {
    float sum = 0.0;
    for (int i = 0; i < d; ++i) {
        auto dif = x[i] - float(y[i]);
        sum += dif * dif;
    }
    return sum;
}

inline float IPE5M2(const float *x, const e5m2 *y, int d) {
    float sum = 0.0;
    for (int i = 0; i < d; ++i) {
        sum += x[i] * float(y[i]);
    }
    return -sum;
}

inline float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi, const float *dif) {
    float sum = 0.0;
    for (int i = 0; i < d; ++i) {
        float yy = (y[i] + 0.5f);
        yy = yy * dif[i] + mi[i] * 256.0f;
        auto dif = x[i] * 256.0f - yy;
        sum += dif * dif;
    }
    return sum;
}

inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi, const float *dif) {
    float sum = 0.0;
    for (int i = 0; i < d; ++i) {
        float yy = y[i] + 0.5f;
        yy = yy * dif[i] + mi[i] * 256.0f;
        sum += x[i] * yy;
    }
    return -sum;
}

inline float L2SqrSQ6_ext(const float * /**x*/, const uint8_t * /**y*/, int /**d*/, const float * /**mi*/,
                          const float * /**dif*/) {
    // TODO
    return 0.0f;
}

inline float IPSQ6_ext(const float * /**x*/, const uint8_t * /**y*/, int /**d*/, const float * /**mi*/,
                       const float * /**dif*/) {
    // TODO
    return 0.0f;
}

inline float L2SqrSQ4_ext(const float *x, const uint8_t *y, int d, const float *mi, const float *dif) {
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

inline float IPSQ4_ext(const float *x, const uint8_t *y, int d, const float *mi, const float *dif) {
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

}  // namespace glass

#endif