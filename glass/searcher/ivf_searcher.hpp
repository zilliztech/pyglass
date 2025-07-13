#pragma once

#include <omp.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "glass/neighbor.hpp"
#include "glass/quant/quant.hpp"
#include "glass/searcher/refiner.hpp"
#include "glass/searcher/searcher_base.hpp"

namespace glass {

template <QuantConcept Quant>
struct CoarseSearcher : public SearcherBase {
    int32_t d;
    int32_t nb;
    Quant quant;

    CoarseSearcher() = default;

    CoarseSearcher(const CoarseSearcher &) = delete;
    CoarseSearcher(CoarseSearcher &&) = delete;
    CoarseSearcher &operator=(const CoarseSearcher &) = delete;
    CoarseSearcher &operator=(CoarseSearcher &&) = delete;

    void SetData(const float *data, int32_t n, int32_t dim, int32_t *ivf_map = nullptr,
                 const float *centroids = nullptr, int32_t ncentroids = 0) override {
        this->nb = n;
        this->d = dim;
        quant = Quant(d);
        quant.train(data, n);
        quant.add(data, n);
    }

    void Search(const float *q, int32_t k, int32_t *dst, float *scores = nullptr) const override {
        auto computer = quant.get_computer(q);
        MaxHeap<Neighbor<float>> pool(k);
        for (int32_t i = 0; i < nb; ++i) {
            auto dist = computer(i);
            pool.push(Neighbor<float>(i, dist));
        }
        for (int32_t i = 0; i < k; ++i) {
            auto nbr = pool.pop();
            dst[k - i - 1] = nbr.id;
            if (scores) {
                scores[k - i - 1] = nbr.distance;
            }
        }
    }

    void SearchBatch(const float *qs, int32_t nq, int32_t k, int32_t *dst, float *scores = nullptr) const {
#pragma omp parallel for schedule(dynamic)
        for (int32_t i = 0; i < nq; ++i) {
            Search(qs + i * d, k, dst + i * k, scores ? scores + i * k : nullptr);
        }
    }
};

template <QuantConcept Quant>
struct IVFSearcher : public SearcherBase {
    int32_t d;
    int32_t nb;
    Quant quant;

    // Coarse searcher
    int32_t ncentroids;
    std::unique_ptr<SearcherBase> coarse_searcher = nullptr;

    // Ivf
    std::vector<int32_t> bucket_range;
    std::vector<int32_t> o2i_map;
    std::vector<int32_t> i2o_map;

    // Search parameters
    int32_t nprobe = 32;

    mutable std::vector<LinearPool<typename Quant::ComputerType::dist_type, Bitset<>>> pools;

    IVFSearcher() : pools(std::thread::hardware_concurrency()) {}

    void SetData(const float *data, int32_t n, int32_t dim, int32_t *ivf_map, const float *centroids,
                 int32_t ncentroids) override {
        this->nb = n;
        this->d = dim;

        this->ncentroids = ncentroids;
        std::vector<std::vector<int32_t>> ivf_lists(ncentroids);
        i2o_map.resize(n);
        o2i_map.resize(n);

        for (int32_t i = 0; i < n; ++i) {
            ivf_lists[ivf_map[i]].push_back(i);
        }

        bucket_range.resize(ncentroids + 1);
        for (size_t i = 0; i < ncentroids; ++i) {
            bucket_range[i + 1] = bucket_range[i] + ivf_lists[i].size();
        }

        std::vector<float> reordered_data((int64_t)n * d);

        int32_t cur_idx = 0;
        for (const auto &ivf_list : ivf_lists) {
            for (int32_t idx : ivf_list) {
                memcpy(reordered_data.data() + (int64_t)cur_idx * d, data + (int64_t)idx * d, d * sizeof(float));
                o2i_map[idx] = cur_idx;
                i2o_map[cur_idx] = idx;
                cur_idx++;
            }
        }

        quant = Quant(d);
        quant.train(reordered_data.data(), n);
        quant.add(reordered_data.data(), n);

        coarse_searcher = std::make_unique<CoarseSearcher<Quant>>();
        coarse_searcher->SetData(centroids, ncentroids, d);
    }

    void SetEf(int32_t ef) override { nprobe = ef; }

    int32_t GetEf() const override { return nprobe; }

    void Search(const float *q, int32_t k, int32_t *dst, float *scores = nullptr) const override {
        std::vector<int32_t> coarse_ids(nprobe);
        coarse_searcher->Search(q, nprobe, coarse_ids.data(), nullptr);

        auto computer = quant.get_computer(q);
        MaxHeap<Neighbor<float>> pool(k);

        for (int32_t i = 0; i < nprobe; ++i) {
            int32_t coarse_id = coarse_ids[i];
            int32_t start = bucket_range[coarse_id];
            int32_t end = bucket_range[coarse_id + 1];

            for (int32_t j = start; j < end; ++j) {
                auto dist = computer(j);
                pool.push(Neighbor<float>(j, dist));
            }
        }

        for (int32_t i = 0; i < k; ++i) {
            auto nbr = pool.pop();
            dst[k - i - 1] = i2o_map[nbr.id];
            if (scores) {
                scores[k - i - 1] = nbr.distance;
            }
        }
    }

    void SearchBatch(const float *qs, int32_t nq, int32_t k, int32_t *dst, float *scores = nullptr) const override {
#pragma omp parallel for schedule(dynamic)
        for (int32_t i = 0; i < nq; ++i) {
            Search(qs + i * d, k, dst + i * k, scores ? scores + i * k : nullptr);
        }
    }
};

inline std::unique_ptr<SearcherBase> create_ivf_searcher(const std::string &metric, const std::string &quantizer,
                                                         const std::string &refine_quant = "") {
    auto m = metric_map[metric];
    auto qua = quantizer_map[quantizer];

    using RType = std::unique_ptr<SearcherBase>;
    RType ret = nullptr;

    if (qua == QuantizerType::FP32) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<FP32Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<FP32Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::FP16) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<FP16Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<FP16Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::BF16) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<BF16Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<BF16Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::E5M2) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<E5M2Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<E5M2Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ8U) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<SQ8QuantizerUniform<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<SQ8QuantizerUniform<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ8) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<SQ8Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<SQ8Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ6) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<SQ6Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<SQ6Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ4U) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<SQ4QuantizerUniform<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<SQ4QuantizerUniform<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ4UA) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<SQ4QuantizerUniformAsym<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<SQ4QuantizerUniformAsym<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ4) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<SQ4Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<SQ4Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ2U) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<SQ2QuantizerUniform<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<SQ2QuantizerUniform<Metric::L2>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::SQ1) {
        if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<SQ1Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
            return nullptr;
        }
    } else if (qua == QuantizerType::PQ8) {
        if (m == Metric::L2) {
            ret = std::make_unique<IVFSearcher<ProductQuant<Metric::L2>>>();
        } else if (m == Metric::IP) {
            ret = std::make_unique<IVFSearcher<ProductQuant<Metric::IP>>>();
        } else {
            printf("Metric not supported\n");
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
