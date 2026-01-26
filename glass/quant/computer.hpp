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
    const Storage &storage;

    explicit Computer(const Storage &storage) : storage(storage) {}

    void prefetch(int32_t u, int32_t lines) const { storage.prefetch(u, lines); }
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
    mutable int64_t mem_read_bytes_{};

    ComputerImpl(const Storage &storage, const S *query, const auto &encoder, Args &&...args)
        : Computer<Storage>(storage), args(std::forward<Args>(args)...) {
        if constexpr (std::is_same_v<std::decay_t<decltype(encoder)>, MemCpyTag>) {
            static_assert(std::is_same_v<S, X>);
            q = (X *)align_alloc(this->storage.dim_align() * sizeof(X));
            memcpy(q, query, this->storage.dim() * sizeof(X));
        } else {
            encoder((const S *)query, q);
        }
    }

    ~ComputerImpl() { free(q); }

    GLASS_INLINE dist_type operator()(const Y *p) const {
        dist_cmps_++;
        return std::apply([&](auto &&...args) { return dist_func(q, p, this->storage.dim_align(), args...); }, args);
    }

    GLASS_INLINE dist_type operator()(int32_t u) const { return operator()((const Y *)this->storage.get(u)); }

    GLASS_INLINE size_t dist_cmps() const { return dist_cmps_; }

    GLASS_INLINE size_t mem_read_bytes() const { return mem_read_bytes_; }
};

template <StorageConcept Storage, auto dist_func, typename U, typename T, typename... Args>
struct SymComputerImpl : Computer<Storage> {
    using dist_type = U;
    using X = T;
    static_assert(
        std::is_convertible_v<decltype(dist_func), std::function<dist_type(const X *, const X *, int32_t, Args...)>>);

    std::tuple<Args...> args;

    SymComputerImpl(const Storage &storage, Args &&...args)
        : Computer<Storage>(storage), args(std::forward<Args>(args)...) {}

    GLASS_INLINE dist_type operator()(const X *x, const X *y) const {
        return std::apply([&](auto &&...args) { return dist_func(x, y, this->storage.dim_align(), args...); }, args);
    }

    GLASS_INLINE dist_type operator()(int32_t u, int32_t v) const {
        return operator()((const X *)this->storage.get(u), (const X *)this->storage.get(v));
    }
};

}  // namespace glass
