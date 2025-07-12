#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>
#include <functional>
#include <tuple>
#include <type_traits>

#include "glass/memory.hpp"
#include "glass/storage/tensor.hpp"

namespace glass {

template <typename Computer>
concept ComputerBaseConcept = requires(Computer computer, int32_t u, int32_t lines) {
    { computer.prefetch(u, lines) };
};

template <typename Computer>
concept ComputerConcept = ComputerBaseConcept<Computer> && requires(Computer computer, int32_t u) {
    { computer.operator()(u) } -> std::same_as<typename Computer::dist_type>;
};

template <typename Computer>
concept SymComputerConcept = ComputerBaseConcept<Computer> && requires(Computer computer, int32_t u, int32_t v) {
    { computer.operator()(u, v) } -> std::same_as<typename Computer::dist_type>;
};

template <StorageConcept Storage>
struct Computer {
    const Storage &tensor;

    explicit Computer(const Tensor &tensor) : tensor(tensor) {}

    void prefetch(int32_t u, int32_t lines) const { tensor.prefetch(u, lines); }
};

struct MemCpyTag {};

template <StorageConcept Storage, auto dist_func, typename U, typename T, typename T1, typename T2, typename... Args>
struct ComputerImpl : Computer<Storage> {
    using dist_type = U;
    using S = T;
    using X = T1;
    using Y = T2;
    static_assert(
        std::is_convertible_v<decltype(dist_func), std::function<dist_type(const X *, const Y *, int32_t, Args...)>>);
    X *q = nullptr;
    std::tuple<Args...> args;
    mutable int64_t dist_cmps_{};

    ComputerImpl(const Storage &tensor, const S *query, const auto &encoder, Args &&...args)
        : Computer<Storage>(tensor), args(std::forward<Args>(args)...) {
        if constexpr (std::is_same_v<std::decay_t<decltype(encoder)>, MemCpyTag>) {
            static_assert(std::is_same_v<S, X>);
            q = (X *)align_alloc(this->tensor.dim_align() * sizeof(X));
            memcpy(q, query, this->tensor.dim() * sizeof(X));
        } else {
            encoder((const S *)query, q);
        }
    }

    ~ComputerImpl() { free(q); }

    GLASS_INLINE dist_type operator()(const Y *p) const {
        dist_cmps_++;
        return std::apply([&](auto &&...args) { return dist_func(q, p, this->tensor.dim_align(), args...); }, args);
    }

    GLASS_INLINE dist_type operator()(int32_t u) const { return operator()((const Y *)this->tensor.get(u)); }

    GLASS_INLINE size_t dist_cmps() const { return dist_cmps_; }
};

template <StorageConcept Storage, auto dist_func, typename U, typename T, typename... Args>
struct SymComputerImpl : Computer<Storage> {
    using dist_type = U;
    using X = T;
    static_assert(
        std::is_convertible_v<decltype(dist_func), std::function<dist_type(const X *, const X *, int32_t, Args...)>>);

    std::tuple<Args...> args;

    SymComputerImpl(const Storage &tensor, Args &&...args)
        : Computer<Storage>(tensor), args(std::forward<Args>(args)...) {}

    GLASS_INLINE dist_type operator()(const X *x, const X *y) const {
        return std::apply([&](auto &&...args) { return dist_func(x, y, this->tensor.dim_align(), args...); }, args);
    }

    GLASS_INLINE dist_type operator()(int32_t u, int32_t v) const {
        return operator()((const X *)this->tensor.get(u), (const X *)this->tensor.get(v));
    }
};

}  // namespace glass
