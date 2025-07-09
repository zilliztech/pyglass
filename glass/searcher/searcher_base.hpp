#pragma once

#include <cstdint>

namespace glass {

struct SearchStats {
    float p99_latency_ms = 0.0f;
    double avg_dist_comps = 0.0;
};

struct SearcherBase {
    virtual void SetData(const float *data, int32_t n, int32_t dim, int32_t *ivf_map = nullptr,
                         const float *centroids = nullptr, int32_t ncentroids = 0) = 0;
    virtual void Search(const float *q, int32_t k, int32_t *dst, float *scores = nullptr) const = 0;
    virtual void SearchBatch(const float *qs, int32_t nq, int32_t k, int32_t *dst, float *scores = nullptr) const = 0;
    virtual void Optimize(int32_t num_threads = 0) {}
    virtual void SetEf(int32_t ef) {}
    virtual int32_t GetEf() const { return 0; }
    virtual void EnableStats(bool val) {}
    virtual SearchStats GetStats() const { return SearchStats(); }
    virtual ~SearcherBase() = default;
};

}  // namespace glass
