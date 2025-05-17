#pragma once

#include "helpa/common.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline void
fp32_to_fp16(const float* from, fp16* to, size_t size);

inline void
fp32_to_fp16_align(const float* from, fp16* to, size_t size);

}  // namespace helpa

#if defined(USE_X86)
#include "helpa/platform/x86/data_convert.hpp"
#else
#include "helpa/platform/scalar/data_convert.hpp"
#endif
