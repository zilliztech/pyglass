#pragma once

#include <omp.h>

#include <thread>

#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/product_quant.hpp"
#include "glass/quant/quant.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/searcher/refiner.hpp"
#include "glass/searcher/searcher_base.hpp"
#include "glass/utils.hpp"

namespace glass {

template <QuantConcept Quant>
struct GraphSearcher : public SearcherBase {
    int32_t d;
    int32_t nb;
    Graph<int32_t> graph;
    Quant quant;

    // Search parameters
    int32_t ef = 32;
    bool stats_enabled = false;
    mutable SearchStats stats;

    // Memory prefetch parameters
    int32_t po = 1;
    int32_t pl = 1;
    int32_t graph_po = 1;

    // Optimization parameters
    constexpr static int32_t kOptimizePoints = 1000;
    constexpr static int32_t kTryPos = 20;
    constexpr static int32_t kTryPls = 20;
    constexpr static int32_t kTryK = 10;
    int32_t sample_points_num;
    std::vector<float> optimize_queries;

    mutable std::vector<LinearPool<typename Quant::ComputerType::dist_type, Bitset<>>> pools;

    GraphSearcher(Graph<int32_t> g)
        : graph(std::move(g)), graph_po(graph.K / 16), pools(std::thread::hardware_concurrency()) {}

    GraphSearcher(const GraphSearcher &) = delete;
    GraphSearcher(GraphSearcher &&) = delete;
    GraphSearcher &operator=(const GraphSearcher &) = delete;
    GraphSearcher &operator=(GraphSearcher &&) = delete;

    void SetData(const float *data, int32_t n, int32_t dim, int32_t *ivf_map = nullptr,
                 const float *centroids = nullptr, int32_t ncentroids = 0) override {
        this->nb = n;
        this->d = dim;
        quant = Quant(d);
        printf("Starting quantizer training\n");
        auto t1 = std::chrono::high_resolution_clock::now();
        quant.train(data, n);
        quant.add(data, n);
        auto t2 = std::chrono::high_resolution_clock::now();
        printf("Done quantizer training, cost %.2lfs\n", std::chrono::duration<double>(t2 - t1).count());

        sample_points_num = std::min(kOptimizePoints, nb - 1);
        std::vector<int32_t> sample_points(sample_points_num);
        std::mt19937 rng;
        GenRandom(rng, sample_points.data(), sample_points_num, nb);
        optimize_queries.resize((int64_t)sample_points_num * d);
        for (int32_t i = 0; i < sample_points_num; ++i) {
            memcpy(optimize_queries.data() + (int64_t)i * d, data + (int64_t)sample_points[i] * d, d * sizeof(float));
        }
    }

    void SetEf(int32_t ef) override { this->ef = ef; }

    int32_t GetEf() const override { return ef; }

    void EnableStats(bool val) override { stats_enabled = val; }

    SearchStats GetStats() const override { return stats; }

    void Optimize(int32_t num_threads = 0) override {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        std::vector<int32_t> try_pos(std::min(kTryPos, graph.K));
        std::vector<int32_t> try_pls(std::min(kTryPls, (int32_t)upper_div(quant.code_size(), 64)));
        std::iota(try_pos.begin(), try_pos.end(), 1);
        std::iota(try_pls.begin(), try_pls.end(), 1);
        std::vector<int32_t> dummy_dst(sample_points_num * kTryK);
        printf("=============Start optimization=============\n");
        auto timeit = [&] {
            auto st = std::chrono::high_resolution_clock::now();
            SearchBatch(optimize_queries.data(), sample_points_num, kTryK, dummy_dst.data(), nullptr);
            auto ed = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double>(ed - st).count();
        };
        timeit(), timeit(), timeit();  // warmup
        float min_ela = std::numeric_limits<float>::max();
        int32_t best_po = 0, best_pl = 0;
        for (auto try_po : try_pos) {
            for (auto try_pl : try_pls) {
                this->po = try_po;
                this->pl = try_pl;
                auto ela = timeit();
                if (ela < min_ela) {
                    min_ela = ela;
                    best_po = try_po;
                    best_pl = try_pl;
                }
            }
        }
        float baseline_ela;
        {
            this->po = 1;
            this->pl = 1;
            baseline_ela = timeit();
        }
        float slow_ela;
        {
            this->po = 0;
            this->pl = 0;
            slow_ela = timeit();
        }

        printf(
            "settint best po = %d, best pl = %d\n"
            "gaining %6.2f%% performance improvement wrt baseline\ngaining "
            "%6.2f%% performance improvement wrt slow\n============="
            "Done optimization=============\n",
            best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1), 100.0 * (slow_ela / min_ela - 1));
        this->po = best_po;
        this->pl = best_pl;
        std::vector<float>().swap(optimize_queries);
    }

    void Search(const float *q, int32_t k, int32_t *dst, float *scores = nullptr) const override {
        auto computer = quant.get_computer(q);
        auto &pool = pools[omp_get_thread_num()];
        pool.reset(nb, std::max(k, ef), std::max(k, ef));
        graph.initialize_search(pool, computer);
        SearchImpl1(pool, computer);
        pool.to_sorted(dst, scores, k);
    }

    void SearchBatch(const float *qs, int32_t nq, int32_t k, int32_t *dst, float *scores = nullptr) const override {
        std::vector<float> latencies;
        if (stats_enabled) {
            latencies.resize(nq);
        }
        std::atomic<int64_t> total_dist_cmps{0};
#pragma omp parallel for schedule(dynamic)
        for (int32_t i = 0; i < nq; ++i) {
            std::chrono::high_resolution_clock::time_point start;
            if (stats_enabled) {
                start = std::chrono::high_resolution_clock::now();
            }
            const float *cur_q = qs + i * d;
            int32_t *cur_dst = dst + i * k;
            float *cur_scores = scores ? scores + i * k : nullptr;
            auto computer = quant.get_computer(cur_q);
            auto &pool = pools[omp_get_thread_num()];
            pool.reset(nb, ef, std::max(k, ef));
            graph.initialize_search(pool, computer);
            SearchImpl2(pool, computer);
            pool.to_sorted(cur_dst, cur_scores, k);
            if (stats_enabled) {
                auto end = std::chrono::high_resolution_clock::now();
                latencies[i] = std::chrono::duration<float, std::milli>(end - start).count();
                total_dist_cmps.fetch_add(computer.dist_cmps());
            }
        }
        if (stats_enabled) {
            std::sort(latencies.begin(), latencies.end());
            stats.p99_latency_ms = latencies.empty() ? 0.0f : latencies[static_cast<size_t>(0.99 * nq)];
            stats.avg_dist_comps = (double)total_dist_cmps.load() / nq;
        }
    }

    void SearchImpl1(NeighborPoolConcept auto &pool, const ComputerConcept auto &computer) const {
        while (pool.has_next()) {
            auto u = pool.pop();
            for (int32_t i = 0; i < po; ++i) {
                int32_t to = graph.at(u, i);
                if (to == -1) {
                    break;
                }
                if (!pool.check_visited(to)) {
                    computer.prefetch(to, pl);
                }
            }
            for (int32_t i = 0; i < graph.K; ++i) {
                int32_t v = graph.at(u, i);
                if (v == -1) {
                    break;
                }
                if (i + po < graph.K && graph.at(u, i + po) != -1) {
                    int32_t to = graph.at(u, i + po);
                    if (!pool.check_visited(to)) {
                        computer.prefetch(to, pl);
                    }
                }
                if (pool.check_visited(v)) {
                    continue;
                }
                pool.set_visited(v);
                auto cur_dist = computer(v);
                if (pool.insert(v, cur_dist)) {
                    graph.prefetch(v, graph_po);
                }
            }
        }
    }

    void SearchImpl2(NeighborPoolConcept auto &pool, const ComputerConcept auto &computer) const {
        alignas(64) int32_t edge_buf[graph.K];
        while (pool.has_next()) {
            auto u = pool.pop();
            int32_t edge_size = 0;
            for (int32_t i = 0; i < graph.K; ++i) {
                int32_t v = graph.at(u, i);
                if (v == -1) {
                    break;
                }
                if (pool.check_visited(v)) {
                    continue;
                }
                pool.set_visited(v);
                edge_buf[edge_size++] = v;
            }
            for (int i = 0; i < std::min(po, edge_size); ++i) {
                computer.prefetch(edge_buf[i], pl);
            }
            for (int i = 0; i < edge_size; ++i) {
                if (i + po < edge_size) {
                    computer.prefetch(edge_buf[i + po], pl);
                }
                auto v = edge_buf[i];
                auto cur_dist = computer(v);
                pool.insert(v, cur_dist);
            }
        }
    }
};

inline std::unique_ptr<SearcherBase> create_searcher(Graph<int32_t> graph, const std::string &metric,
                                                     const std::string &quantizer = "FP16",
                                                     const std::string &refine_quant = "") {
    using RType = std::unique_ptr<SearcherBase>;
    auto m = metric_map[metric];
    auto qua = quantizer_map[quantizer];
    RType ret = nullptr;
    if (qua == QuantizerType::FP32) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<FP32Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<FP32Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::FP16) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<FP16Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<FP16Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::BF16) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<BF16Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<BF16Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::E5M2) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<E5M2Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<E5M2Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ8U) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<SQ8QuantizerUniform<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<SQ8QuantizerUniform<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ8) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<SQ8Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<SQ8Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ6) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<SQ6Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<SQ6Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ4U) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<SQ4QuantizerUniform<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<SQ4QuantizerUniform<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ4UA) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<SQ4QuantizerUniformAsym<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<SQ4QuantizerUniformAsym<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ4) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<SQ4Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<SQ4Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ2U) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<SQ2QuantizerUniform<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<SQ2QuantizerUniform<Metric::L2>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ1) {
        if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<SQ1Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::PQ8) {
        if (m == Metric::L2) {
            ret = std::make_unique<GraphSearcher<ProductQuant<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            ret = std::make_unique<GraphSearcher<ProductQuant<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else {
        printf("Quantizer type not supported\n");
        return nullptr;
    }
    if (ret && !refine_quant.empty()) {
        float factor = get_refine_factor(quantizer);
        if (m == Metric::L2) {
            ret = make_refiner<Metric::L2>(std::move(ret), refine_quant, factor);
        } else {
            ret = make_refiner<Metric::IP>(std::move(ret), refine_quant, factor);
        }
    }
    return ret;
}

}  // namespace glass
