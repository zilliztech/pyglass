#pragma once

#include "glass/builder.hpp"
#include "glass/common.hpp"
#include "glass/graph.hpp"
#include "glass/graph_statistic.hpp"
#include "glass/hnsw/HNSWInitializer.hpp"
#include "glass/hnswlib/hnswalg.h"
#include "glass/memory.hpp"
#include "glass/quant/quant.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/utils.hpp"
#include <chrono>
#include <memory>

namespace glass {

template <SymComputableQuantConcept QuantType> struct HNSW : public Builder {
  int32_t R, efConstruction;

  double construction_time;

  HNSW(int32_t R = 32, int32_t L = 200) : R(R), efConstruction(L) {}

  Graph<int32_t> Build(const float *data, int32_t N, int32_t dim) override {
    QuantType quant(dim);
    quant.train(data, N);
    quant.add(data, N);
    HierarchicalNSW hnsw(quant.get_sym_computer(), N, R / 2, efConstruction);
    running_stats_printer_t printer(N, "Indexing");
    hnsw.addPoint(0);
#pragma omp parallel for schedule(dynamic)
    for (int32_t i = 1; i < N; ++i) {
      hnsw.addPoint(i);
      printer.progress += 1;
      printer.refresh();
    }
    auto time = std::chrono::high_resolution_clock::now();
    construction_time =
        std::chrono::duration<double>(time - printer.start_time).count();
    Graph<int32_t> final_graph(N, R);
#pragma omp parallel for schedule(static, 64)
    for (int64_t i = 0; i < N; ++i) {
      int32_t *edges = (int32_t *)hnsw.get_linklist0(i);
      for (int j = 1; j <= edges[0]; ++j) {
        final_graph.at(i, j - 1) = edges[j];
      }
    }
    auto initializer = std::make_unique<HNSWInitializer>(N, R / 2);
    initializer->ep = hnsw.enterpoint_node_;
    final_graph.eps = {(int)hnsw.enterpoint_node_};
    for (int64_t i = 0; i < N; ++i) {
      int32_t level = hnsw.element_levels_[i];
      initializer->levels[i] = level;
      if (level > 0) {
        initializer->lists[i] = (int *)align_alloc(level * R * 2, true, -1);
        for (int32_t j = 1; j <= level; ++j) {
          int32_t *edges = (int32_t *)hnsw.get_linklist(i, j);
          for (int32_t k = 1; k <= edges[0]; ++k) {
            initializer->at(j, i, k - 1) = edges[k];
          }
        }
      }
    }
    final_graph.initializer = std::move(initializer);
    print_degree_statistic(final_graph);
    return final_graph;
  }

  double GetConstructionTime() const override { return construction_time; }
};

inline std::unique_ptr<Builder>
create_hnsw(const std::string &metric, const std::string &quantizer = "BF16",
            int32_t R = 32, int32_t L = 200) {
  auto m = metric_map[metric];
  auto qua = quantizer_map[quantizer];
  if (qua == QuantizerType::FP32) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<FP32Quantizer<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<FP32Quantizer<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::BF16) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<BF16Quantizer<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<BF16Quantizer<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::FP16) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<FP16Quantizer<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<FP16Quantizer<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::SQ8U) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<SQ8QuantizerUniform<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<SQ8QuantizerUniform<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::SQ4U) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<SQ4QuantizerUniform<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<SQ4QuantizerUniform<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::SQ2U) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<SQ2QuantizerUniform<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<SQ2QuantizerUniform<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::SQ1) {
    if (m == Metric::IP) {
      return std::make_unique<HNSW<SQ1Quantizer<Metric::IP>>>(R, L);
    }
  }
  printf("Quantizer type %s not supported\n", quantizer.c_str());
  return nullptr;
}

struct SparseHNSW {
  int32_t R, efConstruction;

  SparseHNSW(int32_t R = 32, int32_t L = 200) : R(R), efConstruction(L) {}

  Graph<int32_t> Build(const std::string &filename) {
    SparseQuant quant;
    quant.add(filename);
    int32_t nb = quant.n;
    HierarchicalNSW hnsw(quant.get_sym_computer(), nb, R / 2, efConstruction);
    running_stats_printer_t printer(nb, "Indexing");
    hnsw.addPoint(0);
#pragma omp parallel for schedule(dynamic)
    for (int32_t i = 1; i < nb; ++i) {
      hnsw.addPoint(i);
      printer.progress += 1;
      printer.refresh();
    }

    Graph<int32_t> final_graph(nb, R);
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < nb; ++i) {
      int32_t *edges = (int32_t *)hnsw.get_linklist0(i);
      for (int j = 1; j <= edges[0]; ++j) {
        final_graph.at(i, j - 1) = edges[j];
      }
    }
    auto initializer = std::make_unique<HNSWInitializer>(nb, R / 2);
    initializer->ep = hnsw.enterpoint_node_;
    for (int64_t i = 0; i < nb; ++i) {
      int32_t level = hnsw.element_levels_[i];
      initializer->levels[i] = level;
      if (level > 0) {
        initializer->lists[i] = (int *)align_alloc(level * R * 2, true, -1);
        for (int32_t j = 1; j <= level; ++j) {
          int32_t *edges = (int32_t *)hnsw.get_linklist(i, j);
          for (int32_t k = 1; k <= edges[0]; ++k) {
            initializer->at(j, i, k - 1) = edges[k];
          }
        }
      }
    }
    final_graph.initializer = std::move(initializer);
    return final_graph;
  }
};

} // namespace glass
