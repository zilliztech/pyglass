#pragma once

#include "helpa/ref/data_convert_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline void
fp32_to_fp16(const float* from, fp16* to, size_t size) {
    return fp32_to_fp16_ref(from, to, size);
}

}  // namespace helpa
