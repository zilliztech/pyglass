#pragma once

#include <limits>

#include "glass/algorithms/clustering.hpp"
#include "glass/quant/quant_base.hpp"

namespace glass {

template <Metric metric, typename Template = Quantizer<metric, 1, 8>>
struct ProductQuant : Template {
    using type = ProductQuant;
    using data_type = uint8_t;

    constexpr static int32_t K = 256;
    int32_t d;
    int32_t sd;
    int32_t nsq;
    std::vector<Tensor> codebook;

    ProductQuant() = default;

    explicit ProductQuant(int32_t dim, int32_t sd = 4)
        : Template(dim / sd), d(dim), sd(sd), nsq(dim / sd), codebook(nsq) {}

    void train(const float *data, int32_t n) {
#pragma omp parallel for schedule(dynamic)
        for (int32_t sq = 0; sq < nsq; ++sq) {
            Tensor tensor(n, sd);
            for (int32_t i = 0; i < n; ++i) {
                tensor.add(i, (const char *)(data + (int64_t)i * d + sq * sd));
            }
            Clustering cluster(K);
            // if constexpr (metric == Metric::IP) {
            //   cluster.spherical = true;
            // }
            cluster.fit(tensor);
            codebook[sq] = std::move(cluster.medoids);
        }
    }

    void add(const float *data, int32_t n) {
        this->storage.init(n);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            encode(data + (int64_t)i * d, (data_type *)this->get_code(i));
        }
    }

    void encode(const float *from, data_type *to) const {
        for (int sq = 0; sq < nsq; ++sq) {
            const float *cur = from + sq * sd;
            float dist_min = std::numeric_limits<float>::max();
            for (int k = 0; k < K; ++k) {
                float dist = helpa::l2_fp32_fp32(cur, (const float *)codebook[sq].get(k), sd);
                if (dist < dist_min) {
                    dist_min = dist;
                    to[sq] = k;
                }
            }
        }
    }

    void decode(const data_type *from, float *to) const {
        for (int sq = 0; sq < nsq; ++sq) {
            memcpy(to + sq * sd, codebook[sq].get(from[sq]), sd * sizeof(float));
        }
    }

    void get_lut(const float *query, float *lut) const {
        for (int sq = 0; sq < nsq; ++sq) {
            for (int k = 0; k < K; ++k) {
                if constexpr (metric == Metric::L2) {
                    lut[sq * K + k] = helpa::l2_fp32_fp32(query + sq * sd, (const float *)codebook[sq].get(k), sd);
                } else {
                    lut[sq * K + k] = helpa::dot_fp32_fp32(query + sq * sd, (const float *)codebook[sq].get(k), sd);
                }
            }
        }
    }

    static float dist_func(const float *lut, const uint8_t *y, int32_t nsq) {
        float sum = 0.0f;
        for (int sq = 0; sq < nsq; ++sq) {
            sum += lut[sq * K + y[sq]];
        }
        return sum;
    }

    using ComputerType = ComputerImpl<Tensor, dist_func, float, float, float, uint8_t>;

    auto get_computer(const float *query) const {
        return ComputerType(this->storage, query, [this](const float *from, float *&to) {
            to = (float *)align_alloc(nsq * K * sizeof(float));
            get_lut(from, to);
        });
    }
};

}  // namespace glass
