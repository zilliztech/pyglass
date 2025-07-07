#pragma once

#include <cstdint>
#include <cstring>

#include "glass/memory.hpp"
#include "glass/simd/common.hpp"
#include "glass/types.hpp"

namespace glass {

template <typename Storage>
concept StorageConcept = requires(Storage storage, int32_t u) {
    { storage.get(u) } -> std::same_as<char *>;
};

struct Tensor {
    int32_t nb = 0;
    int32_t d = 0;
    int32_t nbits = 0;
    int32_t align_width = 0;
    int32_t dalign = 0;
    int32_t csize = 0;
    char *codes = nullptr;

    DataType dtype = DataType::ph;

    Tensor() = default;

    Tensor(int32_t dim, int32_t nbits, int32_t align_width)
        : d(dim),
          nbits(nbits),
          align_width(align_width),
          dalign((dim + align_width - 1) / align_width * align_width),
          csize(dalign * nbits / 8) {}

    ~Tensor() { free(codes); }

    Tensor(int32_t n, int32_t dim, int32_t nbits, int32_t align_width) : Tensor(dim, nbits, align_width) { init(n); }

    Tensor(int32_t n, int32_t dim, DataType dtype = DataType::fp32) {
        align_width = 1;
        d = dalign = dim;
        if (dtype == DataType::fp32) {
            constexpr DataType dt = DataType::fp32;
            nbits = TypeTraits<dt>::nbits;
        }
        if (dtype == DataType::fp16) {
            constexpr DataType dt = DataType::fp16;
            nbits = TypeTraits<dt>::nbits;
        }
        if (dtype == DataType::fp16) {
            constexpr DataType dt = DataType::fp16;
            nbits = TypeTraits<dt>::nbits;
        }
        if (dtype == DataType::uint8) {
            constexpr DataType dt = DataType::uint8;
            nbits = TypeTraits<dt>::nbits;
        }
        csize = dalign * nbits / 8;
        init(n);
    }

    Tensor(const Tensor &rhs) = delete;

    Tensor(Tensor &&rhs) { swap(*this, rhs); }

    Tensor &operator=(const Tensor &rhs) = delete;

    Tensor &operator=(Tensor &&rhs) {
        swap(*this, rhs);
        return *this;
    }

    friend void swap(Tensor &lhs, Tensor &rhs) {
        using std::swap;
        swap(lhs.nb, rhs.nb);
        swap(lhs.d, rhs.d);
        swap(lhs.nbits, rhs.nbits);
        swap(lhs.align_width, rhs.align_width);
        swap(lhs.dalign, rhs.dalign);
        swap(lhs.csize, rhs.csize);
        swap(lhs.codes, rhs.codes);
    }

    void init(int32_t n) {
        nb = n;
        this->codes = (char *)align_alloc((int64_t)n * csize);
    }

    char *get(int32_t u) const { return codes + (int64_t)u * csize; }

    void add(int32_t u, const char *x) { memcpy(get(u), x, csize); }

    void prefetch(int32_t u, int32_t num) const { mem_prefetch(get(u), num); }

    int32_t size() const { return nb; }
    int32_t dim() const { return d; }
    int32_t dim_align() const { return dalign; }
    int32_t code_size() const { return csize; }

    Tensor copy() {
        Tensor ret(nb, d, nbits, align_width);
        for (int32_t i = 0; i < nb; ++i) {
            ret.add(i, get(i));
        }
        return ret;
    }

    Tensor astype(DataType type_dst) {
        Tensor ret(nb, d, type_dst);
        for (int32_t i = 0; i < nb; ++i) {
            convert_vector(type_dst, ret.get(i), dtype, this->get(i), d);
        }
        return ret;
    }
};

}  // namespace glass
