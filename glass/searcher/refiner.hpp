#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "glass/neighbor.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant.hpp"
#include "glass/searcher/searcher_base.hpp"

namespace glass {

template <QuantConcept QuantType>
struct Refiner : SearcherBase {
    int32_t dim;
    std::unique_ptr<SearcherBase> inner;
    QuantType quant;

    float reorder_mul = 1.0f;

    Refiner(std::unique_ptr<SearcherBase> inner, float reorder_mul = 1.0f)
        : inner(std::move(inner)), reorder_mul(reorder_mul) {}

    void SetData(const float *data, int32_t n, int32_t dim, int32_t *ivf_map = nullptr,
                 const float *centroids = nullptr, int32_t ncentroids = 0) override {
        this->dim = dim;
        quant = QuantType(dim);

        printf("Starting refiner quantizer training\n");
        auto t1 = std::chrono::high_resolution_clock::now();
        quant.train(data, n);
        quant.add(data, n);
        auto t2 = std::chrono::high_resolution_clock::now();
        printf("Done refiner quantizer training, cost %.2lfs\n", std::chrono::duration<double>(t2 - t1).count());
        inner->SetData(data, n, dim, ivf_map, centroids, ncentroids);
    }

    void SetEf(int32_t ef) override { inner->SetEf(ef); }

    void Optimize(int32_t num_threads = 0) override { inner->Optimize(num_threads); }

    void EnableStats(bool val) override { inner->EnableStats(val); }

    SearchStats GetStats() const override { return inner->GetStats(); }

    void Search(const float *q, int32_t k, int32_t *dst, float *scores = nullptr) const override {
        int32_t reorder_k = (int32_t)(k * reorder_mul);
        if (reorder_k == k) {
            inner->Search(q, k, dst, scores);
            return;
        }
        std::vector<int32_t> ret(reorder_k);
        inner->Search(q, reorder_k, ret.data(), nullptr);
        auto computer = quant.get_computer(q);
        MaxHeap<Neighbor<typename decltype(computer)::dist_type>> heap(k);
        for (int i = 0; i < reorder_k; ++i) {
            if (i + 1 < reorder_k) {
                computer.prefetch(ret[i + 1], 1);
            }
            int id = ret[i];
            float dist = computer(id);
            heap.push({id, dist});
        }
        for (int i = heap.size() - 1; i >= 0; --i) {
            auto top = heap.pop();
            dst[i] = top.id;
            if (scores) {
                scores[i] = top.distance;
            }
        }
    }

    void SearchBatch(const float *qs, int32_t nq, int32_t k, int32_t *dst, float *scores = nullptr) const override {
        int32_t reorder_k = (int32_t)(k * reorder_mul);
        if (reorder_k == k) {
            inner->SearchBatch(qs, nq, k, dst, scores);
            return;
        }
        std::vector<int32_t> ret(nq * reorder_k);
        inner->SearchBatch(qs, nq, reorder_k, ret.data(), nullptr);

#pragma omp parallel for schedule(dynamic)
        for (int32_t i = 0; i < nq; ++i) {
            const float *cur_q = qs + i * dim;
            const int32_t *cur_ret = &ret[i * reorder_k];
            int32_t *cur_dst = dst + i * k;
            float *cur_scores = scores + i * k;
            auto computer = quant.get_computer(cur_q);
            RefineImpl(computer, cur_ret, reorder_k, cur_dst, cur_scores, k);
        }
    }

    void RefineImpl(const ComputerConcept auto &computer, const int32_t *from, int32_t from_len, int32_t *to,
                    float *scores, int32_t to_len) const {
        MaxHeap<Neighbor<typename std::decay_t<decltype(computer)>::dist_type>> heap(to_len);
        for (int32_t j = 0; j < from_len; ++j) {
            if (j + 1 < from_len) {
                computer.prefetch(from[j + 1], 1);
            }
            int id = from[j];
            float dist = computer(id);
            heap.push({id, dist});
        }
        for (int j = heap.size() - 1; j >= 0; --j) {
            auto top = heap.pop();
            to[j] = top.id;
            if (scores) {
                scores[j] = top.distance;
            }
        }
    }
};

inline float get_refine_factor(const std::string &quantizer) {
    auto qua = quantizer_map[quantizer];
    if (qua == QuantizerType::SQ8U) return 1.5f;
    if (qua == QuantizerType::SQ8) return 1.5f;
    if (qua == QuantizerType::SQ8P) return 1.5f;
    if (qua == QuantizerType::SQ4U) return 1.5f;
    if (qua == QuantizerType::SQ4UA) return 1.5f;
    if (qua == QuantizerType::SQ2U) return 3.0f;
    if (qua == QuantizerType::SQ1) return 3.0f;
    if (qua == QuantizerType::PQ8) return 1.5f;
    return 1.0f;
}

template <Metric metric, typename... Args>
std::unique_ptr<SearcherBase> make_refiner(std::unique_ptr<SearcherBase> inner, const std::string &refine_quant,
                                           float factor, Args... args) {
    auto refine_qua = quantizer_map[refine_quant];
    if (refine_qua == QuantizerType::FP16) {
        return std::make_unique<Refiner<FP16Quantizer<metric>>>(std::move(inner), factor, args...);
    } else if (refine_qua == QuantizerType::FP32) {
        return std::make_unique<Refiner<FP32Quantizer<metric>>>(std::move(inner), factor, args...);
    } else if (refine_qua == QuantizerType::SQ8U) {
        return std::make_unique<Refiner<SQ8QuantizerUniform<metric>>>(std::move(inner), factor, args...);
    } else if (refine_qua == QuantizerType::SQ8) {
        return std::make_unique<Refiner<SQ8Quantizer<metric>>>(std::move(inner), factor, args...);
    }
    return inner;
}

}  // namespace glass
