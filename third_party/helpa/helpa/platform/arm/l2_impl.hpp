#pragma once

#include "helpa/platform/arm/arm_utils.hpp"

#if defined(USE_ARM)

#include <arm_neon.h>

#if defined(USE_SVE)
#include <arm_sve.h>
#endif

#include "helpa/l2.hpp"
#include "helpa/ref/l2_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline float
l2_fp32_fp32(const float* x, const float* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return l2a_fp32_fp32(x, y, da) + l2_fp32_fp32_ref(x + da, y + da, d - da);
}

inline float
l2_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return l2a_fp32_fp16(x, y, da) + l2_fp32_fp16_ref(x + da, y + da, d - da);
}

inline float
l2_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return l2a_fp16_fp16(x, y, da) + l2_fp16_fp16_ref(x + da, y + da, d - da);
}

inline float
l2_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return l2a_fp32_bf16(x, y, da) + l2_fp32_bf16_ref(x + da, y + da, d - da);
}

inline float
l2_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return l2a_bf16_bf16(x, y, da) + l2_bf16_bf16_ref(x + da, y + da, d - da);
}

inline int32_t
l2_s7_s7(const int8_t* x, const int8_t* y, const int32_t d) {
    int32_t da = d / 64 * 64;
    return l2a_s7_s7(x, y, da) + l2_s7_s7_ref(x + da, y + da, d - da);
}

inline float
l2a_fp32_fp32(const float* x, const float* y, const int32_t d) {
#if defined(USE_SVE)
    auto wid = svcntw();
    auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svld1_f32(b32_all, x + i);
        auto yy = svld1_f32(b32_all, y + i);
        auto t = svsub_f32_x(b32_all, xx, yy);
        sum = svmad_f32_x(b32_all, t, t, sum);
    }
    return svaddv_f32(b32_all, sum);
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
        auto t0 = vsubq_f32(xx0, yy0);
        auto t1 = vsubq_f32(xx1, yy1);
        auto t2 = vsubq_f32(xx2, yy2);
        auto t3 = vsubq_f32(xx3, yy3);
        sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
        sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
        sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
        sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
    }
    return reduce_f32x4x4(sum);
#endif
}

inline float
l2a_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
#if defined(USE_SVE)
    auto wid = svcntw();
    auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svld1_f32(b32_all, x + i);
        auto yy = svcvt_f32_f16_x(b32_all, svreinterpret_f16_u32(svldff1uh_u32(b32_all, (const uint16_t*)(y + i))));
        auto t = svsub_f32_x(b32_all, xx, yy);
        sum = svmad_f32_x(b32_all, t, t, sum);
    }
    return svaddv_f32(b32_all, sum);
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
        auto t0 = vsubq_f32(xx0, yy0);
        auto t1 = vsubq_f32(xx1, yy1);
        auto t2 = vsubq_f32(xx2, yy2);
        auto t3 = vsubq_f32(xx3, yy3);
        sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
        sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
        sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
        sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
    }
    return reduce_f32x4x4(sum);
#endif
}

inline float
l2a_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
#if defined(USE_SVE)
    const auto wid = svcntw();
    const auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svcvt_f32_f16_x(b32_all, svreinterpret_f16_u32(svldff1uh_u32(b32_all, (const uint16_t*)(x + i))));
        auto yy = svcvt_f32_f16_x(b32_all, svreinterpret_f16_u32(svldff1uh_u32(b32_all, (const uint16_t*)(y + i))));
        auto t = svsub_f32_x(b32_all, xx, yy);
        sum = svmad_f32_x(b32_all, t, t, sum);
    }
    return svaddv_f32(b32_all, sum);
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
        auto t0 = vsubq_f32(xx0, yy0);
        auto t1 = vsubq_f32(xx1, yy1);
        auto t2 = vsubq_f32(xx2, yy2);
        auto t3 = vsubq_f32(xx3, yy3);
        sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
        sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
        sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
        sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
    }
    return reduce_f32x4x4(sum);
#endif
}

inline float
l2a_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
#if defined(USE_SVE)
    const auto wid = svcntw();
    const auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svld1_f32(b32_all, x + i);
        auto yy = svldff1uh_u32(b32_all, (const uint16_t*)(y + i));
        yy = svlsl_n_u32_x(b32_all, yy, 16);
        auto zz = svreinterpret_f32_u32(yy);
        auto t = svsub_f32_x(b32_all, xx, zz);
        sum = svmad_f32_x(b32_all, t, t, sum);
    }
    return svaddv_f32(b32_all, sum);
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
        auto t0 = vsubq_f32(xx0, yy0);
        auto t1 = vsubq_f32(xx1, yy1);
        auto t2 = vsubq_f32(xx2, yy2);
        auto t3 = vsubq_f32(xx3, yy3);
        sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
        sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
        sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
        sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
    }
    return reduce_f32x4x4(sum);
#endif
}

inline float
l2a_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
#if defined(USE_SVE)
    const auto wid = svcntw();
    const auto b32_all = svptrue_b32();
    auto sum = svdup_f32(0.0f);
    for (int32_t i = 0; i < d; i += wid) {
        auto uu = svldff1uh_u32(b32_all, (const uint16_t*)(x + i));
        uu = svlsl_n_u32_x(b32_all, uu, 16);
        auto xx = svreinterpret_f32_u32(uu);
        auto zz = svldff1uh_u32(b32_all, (const uint16_t*)(y + i));
        zz = svlsl_n_u32_x(b32_all, zz, 16);
        auto yy = svreinterpret_f32_u32(zz);
        auto t = svsub_f32_x(b32_all, xx, yy);
        sum = svmad_f32_x(b32_all, t, t, sum);
    }
    return svaddv_f32(b32_all, sum);
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
        auto t0 = vsubq_f32(xx0, yy0);
        auto t1 = vsubq_f32(xx1, yy1);
        auto t2 = vsubq_f32(xx2, yy2);
        auto t3 = vsubq_f32(xx3, yy3);
        sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
        sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
        sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
        sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
    }
    return reduce_f32x4x4(sum);
#endif
}

inline int32_t
l2a_s7_s7(const int8_t* x, const int8_t* y, const int32_t d) {
#if defined(USE_SVE)
    auto wid = svcntb();
    auto b8_all = svptrue_b8();
    auto sum = svdup_s32(0);
    for (int32_t i = 0; i < d; i += wid) {
        auto xx = svld1_s8(b8_all, x + i);
        auto yy = svld1_s8(b8_all, y + i);
        auto t = svsub_s8_x(b8_all, xx, yy);
        sum = svdot_s32(sum, t, t);
    }
    return svaddv_s32(svptrue_b32(), sum);
#else
    // int32x4_t sum = vdupq_n_s32(0);
    // for (int32_t i = 0; i < d; i += 16) {
    //   auto uu = vld1q_s8(x + i);
    //   auto zz = vld1q_s8(y + i);
    //   auto xx0 = vreinterpretq_s16_u16(vmovl_s8(vget_low_s8(uu)));
    //   auto xx1 = vreinterpretq_s16_u16(vmovl_s8(vget_high_s8(uu)));
    //   auto yy0 = vreinterpretq_s16_u16(vmovl_s8(vget_low_s8(zz)));
    //   auto yy1 = vreinterpretq_s16_u16(vmovl_s8(vget_high_s8(zz)));
    //   auto t0 = vsubq_s16(xx0, yy0);
    //   auto t1 = vsubq_s16(xx1, yy1);
    //   t0 = vmulq_s16(t0, t0);
    //   t1 = vmulq_s16(t1, t1);
    //   sum = vaddw_s16(sum, vget_low_s16(t0));
    //   sum = vaddw_s16(sum, vget_high_s16(t0));
    //   sum = vaddw_s16(sum, vget_low_s16(t1));
    //   sum = vaddw_s16(sum, vget_high_s16(t1));
    // }
    // return vaddvq_s32(sum);
    int32_t sum = 0;
    for (int32_t i = 0; i < d; ++i) {
        auto d = int32_t(x[i]) - int32_t(y[i]);
        sum += d * d;
    }
    return sum;
#endif
}

inline int32_t
l2a_u4_u4(const uint8_t* x, const uint8_t* y, const int32_t d) {
#if defined(USE_SVE)
    auto wid = svcntb();
    auto b8_all = svptrue_b8();
    auto sum0 = svdup_s32(0), sum1 = svdup_s32(0);
    for (int32_t i = 0; i < d / 2; i += wid) {
        auto xx = svld1_u8(b8_all, x + i);
        auto yy = svld1_u8(b8_all, y + i);
        auto xx0 = svand_n_u8_x(b8_all, xx, 0x0f);
        auto xx1 = svand_n_u8_x(b8_all, svlsr_n_u8_x(b8_all, xx, 4), 0x0f);
        auto yy0 = svand_n_u8_x(b8_all, yy, 0x0f);
        auto yy1 = svand_n_u8_x(b8_all, svlsr_n_u8_x(b8_all, yy, 4), 0x0f);
        auto t0 = svsub_s8_x(b8_all, svreinterpret_s8_u8(xx0), svreinterpret_s8_u8(yy0));
        auto t1 = svsub_s8_x(b8_all, svreinterpret_s8_u8(xx1), svreinterpret_s8_u8(yy1));
        t0 = svabs_s8_x(b8_all, t0);
        t1 = svabs_s8_x(b8_all, t1);
        sum0 = svdot_s32(sum0, t0, t0);
        sum1 = svdot_s32(sum1, t1, t1);
    }
    sum0 = svadd_s32_x(svptrue_b32(), sum0, sum1);
    return svaddv_s32(svptrue_b32(), sum0);
#else
    // TODO
    return 0;
#endif
}

}  // namespace helpa

#endif
