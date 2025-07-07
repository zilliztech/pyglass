#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "helpa/types.hpp"
#if defined(__SSE2__)
#include <immintrin.h>
#endif

namespace glass {

using bf16 = helpa::bf16;

using fp16 = helpa::fp16;

struct e5m2 {
    uint8_t x = 0;

    e5m2() = default;

    explicit e5m2(fp16 f) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        std::memcpy(&x, &f, 1);
#else
        std::memcpy(&x, reinterpret_cast<const char *>(&f) + 1, 1);
#endif
    }

    explicit e5m2(float f) : e5m2(fp16(f)) {}

    explicit operator fp16() const {
        uint16_t buf = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        std::memcpy(reinterpret_cast<char *>(&buf), &x, 1);
#else
        std::memcpy(reinterpret_cast<char *>(&buf) + 1, &x, 1);
#endif
        auto ptr = reinterpret_cast<void *>(&buf);
        return *reinterpret_cast<fp16 *>(ptr);
    }

    explicit operator float() const { return float(fp16(*this)); }
};

enum class DataType { fp32 = 0, fp16 = 1, bf16 = 2, e5m2 = 3, uint8 = 4, ph = 5 };

template <DataType T>
struct TypeTraits {};

template <>
struct TypeTraits<DataType::fp32> {
    constexpr static int32_t nbits = 32;
};

template <>
struct TypeTraits<DataType::fp16> {
    constexpr static int32_t nbits = 16;
};

template <>
struct TypeTraits<DataType::bf16> {
    constexpr static int32_t nbits = 16;
};

template <>
struct TypeTraits<DataType::e5m2> {
    constexpr static int32_t nbits = 8;
};

template <>
struct TypeTraits<DataType::uint8> {
    constexpr static int32_t nbits = 8;
};

inline void convert_vector(DataType src_type, const char *src, DataType dst_type, char *dst, int32_t d) {
    if (src_type == DataType::fp32 && dst_type == DataType::bf16) {
        const float *x = (const float *)src;
        bf16 *y = (bf16 *)dst;
        for (int32_t i = 0; i < d; ++i) {
            y[i] = bf16(x[i]);
        }
    }
}

}  // namespace glass
