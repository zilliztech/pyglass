#pragma once

#include "helpa/common.hpp"

#if defined(USE_ARM)

#include <arm_neon.h>

namespace helpa {

HELPA_INLINE inline float32x4_t
bf16_to_fp32(uint16x4_t x) {
    return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x), 16));
}

HELPA_INLINE inline float
reduce_f32x4x4(float32x4x4_t x) {
    x.val[0] = vaddq_f32(x.val[0], x.val[1]);
    x.val[2] = vaddq_f32(x.val[2], x.val[3]);
    x.val[0] = vaddq_f32(x.val[0], x.val[2]);
    return vaddvq_f32(x.val[0]);
}

}  // namespace helpa

#endif
