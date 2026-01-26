#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "helpa/common.hpp"
#if defined(USE_F16C)
#include <immintrin.h>
#endif
#if defined(USE_SVE)
#include <arm_bf16.h>
#include <arm_fp16.h>
#include <arm_sve.h>
#endif

namespace helpa {

#define ROUND_MODE_TO_NEAREST

struct bf16 {
    uint16_t x = 0;

    bf16() = default;

    bf16(float f)
        : x(
#if defined(USE_SVE)
              [&] {
                  auto tmp = vcvth_bf16_f32(f);
                  return *(uint16_t*)&tmp;
              }()
#elif defined(ROUND_MODE_TO_NEAREST)
              round_to_nearest(f)
#elif defined(ROUND_MODE_TO_NEAREST_EVEN)
              round_to_nearest_even(f)
#elif defined(ROUND_MODE_TRUNCATE)
              truncate(f)
#else
#error "ROUNDING_MODE must be one of ROUND_MODE_TO_NEAREST, ROUND_MODE_TO_NEAREST_EVEN, or ROUND_MODE_TRUNCATE"
#endif
          ) {
    }

    template <typename F, std::enable_if_t<std::is_convertible_v<F, float>>* = nullptr>
    bf16(F f) : bf16(float(f)) {
    }

    operator float() const {
#if defined(USE_SVE)
        return vcvtah_f32_bf16(*(bfloat16_t*)&x);
#else
        uint32_t buf = 0;
        std::memcpy(reinterpret_cast<char*>(&buf) + 2, &x, 2);
        auto ptr = reinterpret_cast<void*>(&buf);
        return *reinterpret_cast<float*>(ptr);
#endif
    }

    static uint32_t
    getbits(float x) {
        auto ptr = reinterpret_cast<void*>(&x);
        return *(reinterpret_cast<uint32_t*>(ptr));
    }

    static uint16_t
    round_to_nearest_even(float x) {
        return static_cast<uint16_t>((getbits(x) + ((getbits(x) & 0x00010000) >> 1)) >> 16);
    }

    static uint16_t
    round_to_nearest(float x) {
        return static_cast<uint16_t>((getbits(x) + 0x8000) >> 16);
    }

    static uint16_t
    truncate(float x) {
        return static_cast<uint16_t>((getbits(x)) >> 16);
    }
};

struct fp16 {
    uint16_t x = 0;

    fp16() = default;

    fp16(float f) {
#if defined(USE_SVE)
        float16_t tmp = f;
        x = *(uint16_t*)&tmp;
#elif defined(USE_F16C)
        __m128 xf = _mm_set1_ps(f);
        __m128i xi = _mm_cvtps_ph(xf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        x = _mm_cvtsi128_si32(xi) & 0xffff;
#else
        uint32_t sign_mask = 0x80000000u;
        int32_t o;
        uint32_t fint = intbits(f);
        uint32_t sign = fint & sign_mask;
        fint ^= sign;
        uint32_t f32infty = 255u << 23;
        o = (fint > f32infty) ? 0x7e00u : 0x7c00u;
        const uint32_t round_mask = ~0xfffu;
        const uint32_t magic = 15u << 23;
        float fscale = floatbits(fint & round_mask) * floatbits(magic);
        fscale = std::min(fscale, floatbits((31u << 23) - 0x1000u));
        int32_t fint2 = intbits(fscale) - round_mask;
        if (fint < f32infty) {
            o = fint2 >> 13;
        }
        x = o | (sign >> 16);
#endif
    }

    template <typename F, std::enable_if_t<std::is_convertible_v<F, float>>* = nullptr>
    fp16(F f) : fp16(float(f)) {
    }

    operator float() const {
#if defined(USE_SVE)
        return (float)*(float16_t*)&x;
#elif defined(USE_F16C)
        __m128i xi = _mm_set1_epi16(x);
        __m128 xf = _mm_cvtph_ps(xi);
        return _mm_cvtss_f32(xf);
#else
        const uint32_t shifted_exp = 0x7c00u << 13;
        int32_t o = ((int32_t)(x & 0x7fffu)) << 13;
        int32_t exp = shifted_exp & o;
        o += (int32_t)(127 - 15) << 23;
        int32_t infnan_val = o + ((int32_t)(128 - 16) << 23);
        int32_t zerodenorm_val = intbits(floatbits(o + (1u << 23)) - floatbits(113u << 23));
        int32_t reg_val = (exp == 0) ? zerodenorm_val : o;
        int32_t sign_bit = ((int32_t)(x & 0x8000u)) << 16;
        return floatbits(((exp == shifted_exp) ? infnan_val : reg_val) | sign_bit);
#endif
    }

    inline static float
    floatbits(uint32_t x) {
        void* xptr = &x;
        return *(float*)xptr;
    }

    inline static uint32_t
    intbits(float f) {
        void* fptr = &f;
        return *(uint32_t*)fptr;
    }
};

struct e5m2 {
    uint8_t x = 0;

    e5m2() = default;

    e5m2(fp16 f) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        std::memcpy(&x, &f, 1);
#else
        std::memcpy(&x, reinterpret_cast<const char*>(&f) + 1, 1);
#endif
    }

    e5m2(float f) : e5m2(fp16(f)) {
    }

    template <typename F, std::enable_if_t<std::is_convertible_v<F, float>>* = nullptr>
    e5m2(F f) : e5m2(float(f)) {
    }

    operator fp16() const {
        uint16_t buf = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        std::memcpy(reinterpret_cast<char*>(&buf), &x, 1);
#else
        std::memcpy(reinterpret_cast<char*>(&buf) + 1, &x, 1);
#endif
        auto ptr = reinterpret_cast<void*>(&buf);
        return *reinterpret_cast<fp16*>(ptr);
    }

    operator float() const {
        return float(fp16(*this));
    }
};

}  // namespace helpa
