#pragma once

#include "helpa/common.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline float
dot_fp32_fp32(const float* x, const float* y, const int32_t d);
inline float
dot_fp32_fp16(const float* x, const fp16* y, const int32_t d);
inline float
dot_fp16_fp16(const fp16* x, const fp16* y, const int32_t d);
inline float
dot_fp32_bf16(const float* x, const bf16* y, const int32_t d);
inline float
dot_bf16_bf16(const bf16* x, const bf16* y, const int32_t d);
inline int32_t
dot_s8_s8(const int8_t* x, const int8_t* y, const int32_t d);

inline float
dota_fp32_fp32(const float* x, const float* y, const int32_t d);
inline float
dota_fp32_fp16(const float* x, const fp16* y, const int32_t d);
inline float
dota_fp16_fp16(const fp16* x, const fp16* y, const int32_t d);
inline float
dota_fp32_bf16(const float* x, const bf16* y, const int32_t d);
inline float
dota_bf16_bf16(const bf16* x, const bf16* y, const int32_t d);
inline int32_t
dota_s8_s8(const int8_t* x, const int8_t* y, const int32_t d);

}  // namespace helpa

#if defined(USE_X86)
#include "helpa/platform/x86/dot_impl.hpp"
#elif defined(USE_ARM)
#include "helpa/platform/arm/dot_impl.hpp"
#else
#include "helpa/platform/scalar/dot_impl.hpp"
#endif
