#pragma once

#include "helpa/common.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline float
l2_fp32_fp32(const float* x, const float* y, const int32_t d);
inline float
l2_fp32_fp16(const float* x, const fp16* y, const int32_t d);
inline float
l2_fp16_fp16(const fp16* x, const fp16* y, const int32_t d);
inline float
l2_fp32_bf16(const float* x, const bf16* y, const int32_t d);
inline float
l2_bf16_bf16(const bf16* x, const bf16* y, const int32_t d);
inline int32_t
l2_s7_s7(const int8_t* x, const int8_t* y, const int32_t d);
inline float
l2a_fp32_fp32(const float* x, const float* y, const int32_t d);
inline float
l2a_fp32_fp16(const float* x, const fp16* y, const int32_t d);
inline float
l2a_fp16_fp16(const fp16* x, const fp16* y, const int32_t d);
inline float
l2a_fp32_bf16(const float* x, const bf16* y, const int32_t d);
inline float
l2a_bf16_bf16(const bf16* x, const bf16* y, const int32_t d);
inline int32_t
l2a_s7_s7(const int8_t* x, const int8_t* y, const int32_t d);
inline int32_t
l2a_u4_u4(const uint8_t* x, const uint8_t* y, const int32_t d);
inline int32_t
l2a_u2_u2(const uint8_t* x, const uint8_t* y, const int32_t d);

}  // namespace helpa

#if defined(USE_X86)
#include "helpa/platform/x86/l2_impl.hpp"
#elif defined(USE_ARM)
#include "helpa/platform/arm/l2_impl.hpp"
#else
#include "helpa/platform/scalar/l2_impl.hpp"
#endif
