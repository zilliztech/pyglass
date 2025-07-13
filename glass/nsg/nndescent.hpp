#pragma once

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/nsg/nhood.hpp"

namespace glass {

template <SymComputerConcept ComputerType>
struct NNDescent {
    std::vector<Nhood> graph;
    Graph<int> final_graph;
    int32_t nb;
    const ComputerType &computer;
    int K;
    int S = 10;
    int R = 100;
    int iters = 10;
    int random_seed = 347;
    int L;

    NNDescent(int32_t nb, const ComputerType &computer) : nb(nb), computer(computer) {}

    void Build(int K) {
        printf("Start building NN-Descent with params K=%d S=%d R=%d L=%d niter=%d\n", K, S, R, L, iters);
        this->K = K;
        this->L = K + 50;
        srand(random_seed);
        Init();
        Descent();
        final_graph.init(nb, K);
        for (int i = 0; i < nb; i++) {
            std::sort(graph[i].pool.begin(), graph[i].pool.end());
            for (int j = 0; j < K; j++) {
                final_graph.at(i, j) = graph[i].pool[j].id;
            }
        }
        std::vector<Nhood>().swap(graph);
    }

    void Init() {
        graph.reserve(nb);
        for (int i = 0; i < nb; ++i) {
            graph.emplace_back(S, nb);
        }
#pragma omp parallel for
        for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < S; j++) {
                int id = rand() % nb;
                if (id == i) {
                    continue;
                }
                auto dist = computer(i, id);
                graph[i].pool.emplace_back(id, dist, true);
            }
            std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
            graph[i].pool.reserve(L);
        }
    }

    void Descent() {
        int num_eval = std::min(100, nb);
        std::vector<int> eval_points(num_eval);
        std::vector<std::vector<int>> eval_gt(num_eval);
        for (auto &u : eval_points) {
            u = rand() % nb;
        }
        GenEvalGt(eval_points, eval_gt);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int iter = 1; iter <= iters; ++iter) {
            auto t11 = std::chrono::high_resolution_clock::now();
            Join();
            Update();
            float recall = EvalRecall(eval_points, eval_gt);
            auto t22 = std::chrono::high_resolution_clock::now();
            auto ella = std::chrono::duration<double>(t22 - t11).count();
            printf("NNDescent iter: [%d/%d], recall: %f, cost: %.2fs\n", iter, iters, recall, ella);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ela = std::chrono::duration<double>(t2 - t1).count();
        printf("NNDescent cost: %.2lfs\n", ela);
    }

    void Join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (int u = 0; u < nb; u++) {
            graph[u].join(computer, [&](int i, int j) {
                if (i != j) {
                    auto dist = computer(i, j);
                    graph[i].insert(j, dist);
                    graph[j].insert(i, dist);
                }
            });
        }
    }

    void Update() {
#pragma omp parallel for
        for (int i = 0; i < nb; i++) {
            std::vector<int>().swap(graph[i].nn_new);
            std::vector<int>().swap(graph[i].nn_old);
        }
#pragma omp parallel for
        for (int n = 0; n < nb; ++n) {
            auto &nn = graph[n];
            std::sort(nn.pool.begin(), nn.pool.end());
            if ((int)nn.pool.size() > L) {
                nn.pool.resize(L);
            }
            nn.pool.reserve(L);
            int maxl = std::min(nn.M + S, (int)nn.pool.size());
            int c = 0;
            int l = 0;
            while ((l < maxl) && (c < S)) {
                if (nn.pool[l].flag) {
                    ++c;
                }
                ++l;
            }
            nn.M = l;
        }
#pragma omp parallel for
        for (int n = 0; n < nb; ++n) {
            auto &node = graph[n];
            auto &nn_new = node.nn_new;
            auto &nn_old = node.nn_old;
            for (int l = 0; l < node.M; ++l) {
                auto &nn = node.pool[l];
                auto &other = graph[nn.id];
                if (nn.flag) {
                    nn_new.push_back(nn.id);
                    if (nn.distance > other.pool.back().distance) {
                        std::scoped_lock guard(other.lock);
                        if ((int)other.rnn_new.size() < R) {
                            other.rnn_new.push_back(n);
                        } else {
                            int pos = rand() % R;
                            other.rnn_new[pos] = n;
                        }
                    }
                    nn.flag = false;
                } else {
                    nn_old.push_back(nn.id);
                    if (nn.distance > other.pool.back().distance) {
                        std::scoped_lock guard(other.lock);
                        if ((int)other.rnn_old.size() < R) {
                            other.rnn_old.push_back(n);
                        } else {
                            int pos = rand() % R;
                            other.rnn_old[pos] = n;
                        }
                    }
                }
            }
            std::make_heap(node.pool.begin(), node.pool.end());
        }
#pragma omp parallel for
        for (int i = 0; i < nb; ++i) {
            auto &nn_new = graph[i].nn_new;
            auto &nn_old = graph[i].nn_old;
            auto &rnn_new = graph[i].rnn_new;
            auto &rnn_old = graph[i].rnn_old;
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            if ((int)nn_old.size() > R * 2) {
                nn_old.resize(R * 2);
                nn_old.reserve(R * 2);
            }
            std::vector<int>().swap(graph[i].rnn_new);
            std::vector<int>().swap(graph[i].rnn_old);
        }
    }

    void GenEvalGt(const std::vector<int> &eval_set, std::vector<std::vector<int>> &eval_gt) {
#pragma omp parallel for
        for (int i = 0; i < (int)eval_set.size(); i++) {
            eval_gt[i].resize(K);
            auto u = eval_set[i];
            MaxHeap<Neighbor<typename ComputerType::dist_type>> heap(K);
            for (int j = 0; j < nb; j++) {
                if (u == j) {
                    continue;
                }
                auto dist = computer(u, j);
                heap.push({j, dist});
            }
            auto tmp = heap.pool;
            std::partial_sort(tmp.begin(), tmp.begin() + K, tmp.end());
            for (int j = 0; j < K; j++) {
                eval_gt[i][j] = tmp[j].id;
            }
        }
    }

    float EvalRecall(const std::vector<int> &eval_set, const std::vector<std::vector<int>> &eval_gt) {
        float mean_acc = 0.0f;
        for (int i = 0; i < (int)eval_set.size(); i++) {
            float acc = 0;
            std::vector<FlagNeighbor> &g = graph[eval_set[i]].pool;
            const std::vector<int> &v = eval_gt[i];
            for (int j = 0; j < (int)g.size(); j++) {
                for (int k = 0; k < (int)v.size(); k++) {
                    if (g[j].id == v[k]) {
                        acc++;
                        break;
                    }
                }
            }
            mean_acc += acc / v.size();
        }
        return mean_acc / eval_set.size();
    }
};

}  // namespace glass
