#pragma once

#include "helpa/common.hpp"
#include "helpa/platform/arm/arm_utils.hpp"

#if defined(USE_ARM)

#include <arm_neon.h>

#if defined(USE_SVE)
#include <arm_sve.h>
#endif

#include "helpa/dot.hpp"
#include "helpa/ref/dot_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline float
dot_fp32_fp32(const float* x, const float* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_fp32(x, y, da) + dot_fp32_fp32_ref(x + da, y + da, d - da);
}

inline float
dot_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_fp16(x, y, da) + dot_fp32_fp16_ref(x + da, y + da, d - da);
}

inline float
dot_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return dota_fp16_fp16(x, y, da) + dot_fp16_fp16_ref(x + da, y + da, d - da);
}

inline float
dot_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_bf16(x, y, da) + dot_fp32_bf16_ref(x + da, y + da, d - da);
}

inline float
dot_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return dota_bf16_bf16(x, y, da) + dot_bf16_bf16_ref(x + da, y + da, d - da);
}

inline int32_t
dot_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    int32_t da = d / 64 * 64;
    return dota_s8_s8(x, y, da) + dot_s8_s8_ref(x + da, y + da, d - da);
}

inline float
dota_fp32_fp32(const float* x, const float* y, const int32_t d) {
#if defined(USE_SVE)
    auto wid = svcntw();
    auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svld1_f32(b32_all, x + i);
        auto yy = svld1_f32(b32_all, y + i);
        sum = svmad_f32_x(b32_all, xx, yy, sum);
    }
    return -svaddv_f32(b32_all, sum);
#else
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto xx0 = vld1q_f32(x + i);
        auto xx1 = vld1q_f32(x + i + 4);
        auto xx2 = vld1q_f32(x + i + 8);
        auto xx3 = vld1q_f32(x + i + 12);
        auto yy0 = vld1q_f32(y + i);
        auto yy1 = vld1q_f32(y + i + 4);
        auto yy2 = vld1q_f32(y + i + 8);
        auto yy3 = vld1q_f32(y + i + 12);
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
#endif
}

inline float
dota_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
#if defined(USE_SVE)
    auto wid = svcntw();
    auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svld1_f32(b32_all, x + i);
        auto yy = svcvt_f32_f16_x(b32_all, svreinterpret_f16_u32(svldff1uh_u32(b32_all, (const uint16_t*)(y + i))));
        sum = svmad_f32_x(b32_all, xx, yy, sum);
    }
    return -svaddv_f32(b32_all, sum);
#else
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto xx0 = vld1q_f32(x + i);
        auto xx1 = vld1q_f32(x + i + 4);
        auto xx2 = vld1q_f32(x + i + 8);
        auto xx3 = vld1q_f32(x + i + 12);
        auto zz0 = vld1q_f16((const __fp16*)(y + i));
        auto zz1 = vld1q_f16((const __fp16*)(y + i + 8));
        auto yy0 = vcvt_f32_f16(vget_low_f16(zz0));
        auto yy1 = vcvt_f32_f16(vget_high_f16(zz0));
        auto yy2 = vcvt_f32_f16(vget_low_f16(zz1));
        auto yy3 = vcvt_f32_f16(vget_high_f16(zz1));
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
#endif
}

inline float
dota_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
#if defined(USE_SVE)
    auto wid = svcntw();
    auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svcvt_f32_f16_x(b32_all, svreinterpret_f16_u32(svldff1uh_u32(b32_all, (const uint16_t*)(x + i))));
        auto yy = svcvt_f32_f16_x(b32_all, svreinterpret_f16_u32(svldff1uh_u32(b32_all, (const uint16_t*)(y + i))));
        sum = svmad_f32_x(b32_all, xx, yy, sum);
    }
    return -svaddv_f32(b32_all, sum);
#else
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto uu0 = vld1q_f16((const __fp16*)(x + i));
        auto uu1 = vld1q_f16((const __fp16*)(x + i + 8));
        auto zz0 = vld1q_f16((const __fp16*)(y + i));
        auto zz1 = vld1q_f16((const __fp16*)(y + i + 8));
        auto xx0 = vcvt_f32_f16(vget_low_f16(uu0));
        auto xx1 = vcvt_f32_f16(vget_high_f16(uu0));
        auto xx2 = vcvt_f32_f16(vget_low_f16(uu1));
        auto xx3 = vcvt_f32_f16(vget_high_f16(uu1));
        auto yy0 = vcvt_f32_f16(vget_low_f16(zz0));
        auto yy1 = vcvt_f32_f16(vget_high_f16(zz0));
        auto yy2 = vcvt_f32_f16(vget_low_f16(zz1));
        auto yy3 = vcvt_f32_f16(vget_high_f16(zz1));
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
#endif
}

inline float
dota_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
#if defined(USE_SVE)
    auto wid = svcntw();
    auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svld1_f32(b32_all, x + i);
        auto yy = svldff1uh_u32(b32_all, (const uint16_t*)(y + i));
        yy = svlsl_n_u32_x(b32_all, yy, 16);
        auto zz = svreinterpret_f32_u32(yy);
        sum = svmad_f32_x(b32_all, xx, zz, sum);
    }
    return -svaddv_f32(b32_all, sum);
#else
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto xx0 = vld1q_f32(x + i);
        auto xx1 = vld1q_f32(x + i + 4);
        auto xx2 = vld1q_f32(x + i + 8);
        auto xx3 = vld1q_f32(x + i + 12);
        auto zz0 = vld1q_u16((const uint16_t*)(y + i));
        auto zz1 = vld1q_u16((const uint16_t*)(y + i + 8));
        auto yy0 = bf16_to_fp32(vget_low_u16(zz0));
        auto yy1 = bf16_to_fp32(vget_high_u16(zz0));
        auto yy2 = bf16_to_fp32(vget_low_u16(zz1));
        auto yy3 = bf16_to_fp32(vget_high_u16(zz1));
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
#endif
}

inline float
dota_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
#if defined(USE_SVE_BF16)
    auto wid = svcnth();
    auto b32_all = svptrue_b32();
    auto b16_all = svptrue_b16();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svld1_bf16(b16_all, (const __bf16*)(x + i));
        auto yy = svld1_bf16(b16_all, (const __bf16*)(y + i));
        sum = svbfdot_f32(sum, xx, yy);
    }
    return -svaddv_f32(b32_all, sum);
#elif defined(USE_SVE)
    auto wid = svcntw();
    auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto uu = svldff1uh_u32(b32_all, (const uint16_t*)(x + i));
        uu = svlsl_n_u32_x(b32_all, uu, 16);
        auto xx = svreinterpret_f32_u32(uu);
        auto zz = svldff1uh_u32(b32_all, (const uint16_t*)(y + i));
        zz = svlsl_n_u32_x(b32_all, zz, 16);
        auto yy = svreinterpret_f32_u32(zz);
        sum = svmad_f32_x(b32_all, xx, yy, sum);
    }
    return -svaddv_f32(b32_all, sum);
#else
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto uu0 = vld1q_u16((const uint16_t*)(x + i));
        auto uu1 = vld1q_u16((const uint16_t*)(x + i + 8));
        auto zz0 = vld1q_u16((const uint16_t*)(y + i));
        auto zz1 = vld1q_u16((const uint16_t*)(y + i + 8));
        auto xx0 = bf16_to_fp32(vget_low_u16(uu0));
        auto xx1 = bf16_to_fp32(vget_high_u16(uu0));
        auto xx2 = bf16_to_fp32(vget_low_u16(uu1));
        auto xx3 = bf16_to_fp32(vget_high_u16(uu1));
        auto yy0 = bf16_to_fp32(vget_low_u16(zz0));
        auto yy1 = bf16_to_fp32(vget_high_u16(zz0));
        auto yy2 = bf16_to_fp32(vget_low_u16(zz1));
        auto yy3 = bf16_to_fp32(vget_high_u16(zz1));
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
#endif
}

inline int32_t
dota_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
#if defined(USE_SVE)
    auto wid = svcntb();
    auto b8_all = svptrue_b8();
    auto sum = svdup_s32(0);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svld1_s8(b8_all, x + i);
        auto yy = svld1_s8(b8_all, y + i);
        sum = svdot_s32(sum, xx, yy);
    }
    return -svaddv_s32(svptrue_b32(), sum);
#else
    // int32x4_t sum = vdupq_n_s32(0);
    // for (int32_t i = 0; i < d; i += 8) {
    //   auto xx = vld1_s8(x + i);
    //   auto yy = vld1_s8(y + i);
    //   auto xxx = vreinterpretq_s16_u16(vmovl_s8(xx));
    //   auto yyy = vreinterpretq_s16_u16(vmovl_s8(yy));
    //   auto t = vsubq_s16(xxx, yyy);
    //   sum = vaddw_s16(sum, vget_low_s16(t));
    //   sum = vaddw_s16(sum, vget_high_s16(t));
    // }
    // return -vaddvq_s32(sum);
    int32_t sum = 0;
    for (int32_t i = 0; i < d; ++i) {
        sum += int32_t(x[i]) * int32_t(y[i]);
    }
    return -sum;
#endif
}

}  // namespace helpa

#endif
