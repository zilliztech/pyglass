#pragma once

#include "glass/neighbor.hpp"
#include "glass/storage/tensor.hpp"

#include <concepts>

namespace glass {

template <typename Quant>
concept QuantBaseConcept = requires(Quant quant) {
  { quant.dim() } -> std::same_as<int32_t>;
  { quant.dim_align() } -> std::same_as<int32_t>;
  { quant.code_size() } -> std::same_as<int32_t>;
}
&&requires(Quant quant, int32_t u) {
  { quant.get_code(u) } -> std::same_as<char *>;
};

template <typename Quant>
concept QuantConcept = QuantBaseConcept<Quant> &&
    requires(Quant quant, const float *x, int32_t n) {
  {quant.train(x, n)};
} && requires(Quant quant, const float *x, int32_t n) {
  {quant.add(x, n)};
} && requires(Quant quant, const float *from, typename Quant::data_type *to) {
  {quant.encode(from, to)};
} && requires(Quant quant, const float *q) {
  { quant.get_computer(q) } -> std::same_as<typename Quant::ComputerType>;
};

template <typename Quant>
concept SymComputableQuantConcept = QuantConcept<Quant> &&
    requires(Quant quant) {
  { quant.get_sym_computer() } -> std::same_as<typename Quant::SymComputerType>;
};

template <Metric METRIC, int32_t AlignWidth, int32_t NBits> struct Quantizer {
  static_assert(AlignWidth * NBits % 8 == 0);

  constexpr static Metric metric = METRIC;
  constexpr static int32_t align_width = AlignWidth;
  constexpr static int32_t nbits = NBits;

  Tensor storage;

  Quantizer() = default;

  explicit Quantizer(int32_t dim) : storage(dim, nbits, align_width) {}

  Quantizer(const Quantizer &) = delete;

  Quantizer &operator=(Quantizer &&rhs) {
    std::swap(storage, rhs.storage);
    return *this;
  }

  virtual ~Quantizer() = default;

  auto size() const { return storage.size(); }

  auto dim() const { return storage.dim(); }

  auto dim_align() const { return storage.dim_align(); }

  auto code_size() const { return storage.code_size(); }

  auto get_code(int32_t u) const { return storage.get(u); }
};

} // namespace glass
