#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

#include "glass/quant/computer.hpp"
#include "glass/storage/tensor.hpp"
#include "glass/utils.hpp"
#include "helpa/core.hpp"

namespace glass {

struct Clustering {
    int32_t d;
    int32_t n_cluster;
    int32_t epochs;
    Tensor medoids;
    using SymComputerType = SymComputerImpl<Tensor, helpa::l2_fp32_fp32, float, float>;
    using ComputerType = ComputerImpl<Tensor, helpa::l2_fp32_fp32, float, float, float, float>;

    constexpr static int32_t kSampleNum = 100000;

    enum class InitType {
        RANDOM = 0,
        KMEANSPP = 1,
    };
    InitType init_type;

    inline const static auto init_map = [] {
        std::unordered_map<std::string, InitType> ret;
        ret["random"] = InitType::RANDOM;
        ret["kmeans++"] = InitType::KMEANSPP;
        return ret;
    }();

    bool spherical = false;
    bool verbose = false;

    Clustering() = default;

    Clustering(int32_t n_cluster, int32_t epochs = 10, const std::string &init = "kmeans++")
        : n_cluster(n_cluster), epochs(epochs), init_type(init_map.at(init)) {}

    void fit(const Tensor &tensor) {
        d = tensor.dim();
        medoids = Tensor(n_cluster, d);
        int32_t n = tensor.size();
        int32_t sample_num = std::min(n - 1, kSampleNum);
        std::vector<int32_t> samples(sample_num);
        std::mt19937 rng;
        GenRandom(rng, samples.data(), sample_num, n);
        Tensor train_data(sample_num, d);
        for (int i = 0; i < sample_num; ++i) {
            train_data.add(i, tensor.get(samples[i]));
        }
        if (init_type == InitType::RANDOM) {
            init_random(train_data);
        } else if (init_type == InitType::KMEANSPP) {
            init_kmeanspp(train_data);
        }
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            auto dists = compute_dist(train_data);
            auto [idx, loss] = find_index(train_data, dists);
            update_mediods(train_data, idx);
            post_process_centroids(medoids);
            if (verbose) {
                printf("Done epoch [%d/%d], loss = %.2lf\n", epoch, epochs, loss);
            }
        }
    }

    void fit(const float *data, int n, int dim) {
        d = dim;
        medoids = Tensor(n_cluster, d);
        int32_t sample_num = std::min(n - 1, kSampleNum);
        std::vector<int32_t> samples(sample_num);
        std::mt19937 rng;
        GenRandom(rng, samples.data(), sample_num, n);
        Tensor train_data(sample_num, d);
        for (int i = 0; i < sample_num; ++i) {
            train_data.add(i, (const char *)(data + samples[i] * d));
        }
        if (init_type == InitType::RANDOM) {
            init_random(train_data);
        } else if (init_type == InitType::KMEANSPP) {
            init_kmeanspp(train_data);
        }
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            auto dists = compute_dist(train_data);
            auto [idx, loss] = find_index(train_data, dists);
            update_mediods(train_data, idx);
            post_process_centroids(medoids);
            if (verbose) {
                printf("Done epoch [%d/%d], loss = %.2lf\n", epoch, epochs, loss);
            }
        }
    }

    std::vector<int32_t> predict(const Tensor &tensor) const {
        auto dists = compute_dist(tensor);
        auto [idx, _] = find_index(tensor, dists);
        return idx;
    }

    std::vector<int32_t> predict(const float *data, int n, int dim) const {
        Tensor predict_data(n, dim);
        for (int i = 0; i < n; ++i) {
            predict_data.add(i, (const char *)(data + i * dim));
        }
        return predict(predict_data);
    }

    void transform(float *data, int n, int dim) {
        Tensor predict_data(n, dim);
        for (int i = 0; i < n; ++i) {
            predict_data.add(i, (const char *)(data + i * dim));
        }
        auto dists = compute_dist(predict_data);
        auto [idx, _] = find_index(predict_data, dists);
        for (int32_t i = 0; i < n; ++i) {
            auto center = (const float *)medoids.get(idx[i]);
            for (int32_t j = 0; j < dim; ++j) {
                data[i * dim + j] -= center[j];
            }
        }
    }

private:
    void init_random(const Tensor &tensor) {
        int32_t n = tensor.size();
        std::mt19937 rng;
        std::vector<int32_t> idx(n_cluster);
        GenRandom(rng, idx.data(), n_cluster, n);
        for (int i = 0; i < n_cluster; ++i) {
            medoids.add(i, tensor.get(idx[i]));
        }
    }

    void init_kmeanspp(const Tensor &tensor) {
        int32_t n = tensor.size();
        std::vector<size_t> idx;
        std::mt19937 rng;
        std::uniform_real_distribution<> distribution(0, 1);
        std::uniform_int_distribution<int32_t> int_dist(0, n - 1);
        int32_t init_id = int_dist(rng);
        int32_t num_picked = 1;

        std::vector<float> dist(n);
        idx.push_back(init_id);
        medoids.add(0, tensor.get(init_id));
        SymComputerType computer(tensor);
        for (int32_t i = 0; i < n; i++) {
            dist[i] = computer(init_id, i);
        }

        double dart_val;
        size_t tmp_pivot;
        bool sum_flag = false;

        while (num_picked < n_cluster) {
            dart_val = distribution(rng);

            double sum = 0;
            for (int32_t i = 0; i < n; i++) {
                sum = sum + dist[i];
            }
            if (sum < 1e-6) {
                sum_flag = true;
            }

            dart_val *= sum;

            double prefix_sum = 0;
            for (int32_t i = 0; i < n; i++) {
                tmp_pivot = i;
                if (dart_val >= prefix_sum && dart_val < prefix_sum + dist[i]) {
                    break;
                }

                prefix_sum += dist[i];
            }

            if (std::find(idx.begin(), idx.end(), tmp_pivot) != idx.end() && sum_flag == false) {
                continue;
            }
            idx.push_back(tmp_pivot);
            medoids.add(num_picked, tensor.get(tmp_pivot));
            SymComputerType computer(tensor);
            for (int32_t i = 0; i < n; i++) {
                dist[i] = std::min(dist[i], computer(tmp_pivot, i));
            }
            num_picked++;
        }
    }

    std::vector<float> compute_dist(const Tensor &tensor) const {
        int n = tensor.size();
        std::vector<float> dists(n * n_cluster);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            ComputerType computer(medoids, (const float *)tensor.get(i), MemCpyTag{});
            for (int j = 0; j < n_cluster; ++j) {
                dists[i * n_cluster + j] = computer(j);
            }
        }
        return dists;
    }

    std::pair<std::vector<int32_t>, double> find_index(const Tensor &tensor, const std::vector<float> &dists) const {
        double loss = 0.0;
        int32_t n = tensor.size();
        std::vector<int32_t> idx(n);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            float min_dist = HUGE_VALF;
            int32_t min_id = -1;
            for (int j = 0; j < n_cluster; ++j) {
                float cur_dist = dists[i * n_cluster + j];
                if (cur_dist < min_dist) {
                    min_dist = cur_dist;
                    min_id = j;
                }
            }
            idx[i] = min_id;
        }
        for (int i = 0; i < n; ++i) {
            loss += dists[i * n_cluster + idx[i]];
        }
        return std::make_pair(idx, loss);
    };

    void update_mediods(const Tensor &tensor, const std::vector<int32_t> &idx) {
        int n = tensor.size();
        Tensor new_medoids(n_cluster, d);
        std::vector<int32_t> cnt(n_cluster);
        for (int i = 0; i < n; ++i) {
            auto id = idx[i];
            cnt[id]++;
            auto from = (const float *)tensor.get(i);
            auto to = (float *)new_medoids.get(id);
            for (int j = 0; j < d; ++j) {
                to[j] += from[j];
            }
        }
        for (int i = 0; i < n_cluster; ++i) {
            if (cnt[i] == 0) {
                continue;
            }
            auto from = (float *)new_medoids.get(i);
            for (int j = 0; j < d; ++j) {
                from[j] /= float(cnt[i]);
            }
        }
        medoids = std::move(new_medoids);
    };

    void post_process_centroids(Tensor &centroids) {
        if (spherical) {
            int32_t n = centroids.size();
            int32_t d = centroids.dim();
            for (int i = 0; i < n; ++i) {
                float *x = (float *)centroids.get(i);
                float norm = helpa::dot_fp32_fp32(x, x, d);
                float div = 1.0f / sqrtf(norm);
                for (int j = 0; j < d; ++j) {
                    x[j] *= div;
                }
            }
        }
    }
};

}  // namespace glass
