#pragma once

#include "helpa/types.hpp"

namespace helpa {

inline void
fp32_to_fp16_ref(const float* from, fp16* to, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        to[i] = fp16(from[i]);
    }
}

}  // namespace helpa