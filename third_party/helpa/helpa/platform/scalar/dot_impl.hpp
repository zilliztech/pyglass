
#pragma once

#include "helpa/ref/dot_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline float
dot_fp32_fp32(const float* x, const float* y, const int32_t d) {
    return dot_fp32_fp32_ref(x, y, d);
}

inline float
dot_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
    return dot_fp32_fp16_ref(x, y, d);
}

inline float
dot_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
    return dot_fp16_fp16_ref(x, y, d);
}

inline float
dot_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
    return dot_fp32_bf16_ref(x, y, d);
}

inline float
dot_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
    return dot_bf16_bf16_ref(x, y, d);
}

inline int32_t
dot_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    return dot_s8_s8_ref(x, y, d);
}

inline float
dota_fp32_fp32(const float* x, const float* y, const int32_t d) {
    return dot_fp32_fp32(x, y, d);
}

inline float
dota_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
    return dot_fp32_fp16(x, y, d);
}

inline float
dota_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
    return dot_fp16_fp16(x, y, d);
}

inline float
dota_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
    return dot_fp32_bf16(x, y, d);
}

inline float
dota_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
    return dot_bf16_bf16(x, y, d);
}

inline int32_t
dota_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    return dot_s8_s8(x, y, d);
}

}  // namespace helpa
