#pragma once

#include <atomic>
#include <chrono>
#include <limits>
#include <random>
#include <stack>

#include "glass/builder.hpp"
#include "glass/graph.hpp"
#include "glass/graph_statistic.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/fp32_quant.hpp"
#include "glass/quant/quant.hpp"
#include "glass/quant/sq4u_quant.hpp"
#include "glass/utils.hpp"
#include "nndescent.hpp"

namespace glass {

template <SymComputableQuantConcept QuantType>
struct NSG : public Builder {
    int32_t d;
    int R;
    int L;
    int C;
    int nb;
    int ep;

    // NNDescent params
    int GK;
    int nndescent_S;
    int nndescent_R;
    int nndescent_L;
    int nndescent_iter;

    // SSG params
    bool use_ssg = false;
    constexpr static double kPi = 3.14159265358979323846264;
    float threshold = (float)std::cos(60.0 / 180.0 * kPi);

    // prefetch params
    int32_t po = 5;
    int32_t pl = 1;

    double construction_time;

    explicit NSG(int R = 32, int L = 0, bool use_ssg = true) : R(R), L(L ? L : R + 32), C(R + 100), use_ssg(use_ssg) {
        this->GK = 64;
        this->nndescent_S = 10;
        this->nndescent_R = 100;
        this->nndescent_L = this->GK + 50;
        this->nndescent_iter = 10;
    }

    Graph<int32_t> Build(const float *data, int32_t n, int32_t d) override {
        printf("Start building NSG with params R=%d L=%d C=%d\n", R, L, C);
        Graph<int32_t> final_graph(n, R);
        this->nb = n;
        this->d = d;
        QuantType quant(d);
        quant.train(data, n);
        quant.add(data, n);
        auto computer = quant.get_sym_computer();
        pl = quant.code_size() / 64;
        NNDescent nnd(n, computer);
        nnd.S = nndescent_S;
        nnd.R = nndescent_R;
        nnd.L = nndescent_L;
        nnd.iters = nndescent_iter;
        auto st = std::chrono::high_resolution_clock::now();
        nnd.Build(GK);
        const auto &knng = nnd.final_graph;

        srand(347);
        Init(quant, data);
        std::vector<int> degrees(n, 0);
        {
            Graph<Node> tmp_graph(n, R);
            link(computer, knng, tmp_graph);
            final_graph.init(n, R);
            std::fill_n(final_graph.data, n * R, final_graph.EMPTY_ID);
            final_graph.eps = {ep};
#pragma omp parallel for schedule(static, 64)
            for (int i = 0; i < n; i++) {
                int cnt = 0;
                for (int j = 0; j < R; j++) {
                    int id = tmp_graph.at(i, j).id;
                    if (id != final_graph.EMPTY_ID) {
                        final_graph.at(i, cnt) = id;
                        cnt += 1;
                    }
                    degrees[i] = cnt;
                }
            }
            auto num_attached = tree_grow(final_graph, computer);
            printf("%d of points are attached\n", num_attached);
        }
        auto ed = std::chrono::high_resolution_clock::now();
        construction_time = std::chrono::duration<double>(ed - st).count();
        printf("NSG building cost: %.2lfs\n", construction_time);
        print_degree_statistic(final_graph);
        return final_graph;
    }

    void Init(const auto &quant, const float *data) {
        std::vector<float> center(d);
        for (int32_t i = 0; i < nb; ++i) {
            for (int32_t j = 0; j < d; ++j) {
                center[j] += data[(int64_t)i * d + j];
            }
        }
        for (int32_t j = 0; j < d; ++j) {
            center[j] /= nb;
        }
        auto computer = quant.get_computer(center.data());
        auto min_dist = computer(0);
        auto min_idx = 0;
        for (int32_t i = 1; i < nb; ++i) {
            auto dist = computer(i);
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = i;
            }
        }
        this->ep = min_idx;
    }

    void search_on_graph(const auto &computer, int32_t p, const Graph<int> &graph, int ep, auto &pool,
                         std::vector<Node> &fullset) const {
        constexpr int32_t po = 5, pl = 1;
        auto ep_dis = computer(p, ep);
        pool.insert(ep, ep_dis);
        pool.set_visited(ep);
        fullset.push_back({ep, static_cast<float>(ep_dis)});
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
                auto cur_dist = computer(p, v);
                pool.insert(v, cur_dist);
                fullset.push_back({v, static_cast<float>(cur_dist)});
            }
        }
    }

    void link(const auto &computer, const Graph<int> &knng, Graph<Node> &graph) {
        running_stats_printer_t printer(nb, "Indexing");
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nb; i++) {
            std::vector<Node> fullset;
            LinearPool<float> pool(nb, L, L);
            search_on_graph(computer, i, knng, ep, pool, fullset);
            sync_prune(computer, i, pool, fullset, knng, graph);
            printer.progress += 1;
            printer.refresh();
        }

        std::vector<std::mutex> locks(nb);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nb; ++i) {
            add_reverse_links(computer, i, locks, graph);
        }
    }

    void sync_prune(const auto &computer, int q, auto &pool, std::vector<Node> &fullset, const Graph<int> &knng,
                    Graph<Node> &graph) {
        for (int i = 0; i < knng.K; i++) {
            int id = knng.at(q, i);
            if (id == -1) {
                break;
            }
            if (pool.check_visited(id)) {
                continue;
            }

            auto dist = computer(q, id);
            fullset.emplace_back(id, dist);
        }

        std::sort(fullset.begin(), fullset.end());

        std::vector<Node> result;

        int start = 0;
        if (fullset[start].id == q) {
            start++;
        }
        result.push_back(fullset[start]);

        while ((int)result.size() < R && (++start) < (int)fullset.size() && start < C) {
            auto &p = fullset[start];
            bool occlude = false;
            for (int t = 0; t < (int)result.size(); t++) {
                if (p.id == result[t].id) {
                    occlude = true;
                    break;
                }

                auto djk = computer(result[t].id, p.id);

                if (use_ssg) {
                    float cos_ij = (p.distance + result[t].distance - djk) / 2 / sqrt(p.distance * result[t].distance);
                    if (cos_ij > threshold) {
                        occlude = true;
                        break;
                    }
                } else {
                    if (djk < p.distance /* dik */) {
                        occlude = true;
                        break;
                    }
                }
            }
            if (!occlude) {
                result.push_back(p);
            }
        }

        for (int i = 0; i < R; i++) {
            if (i < (int)result.size()) {
                graph.at(q, i).id = result[i].id;
                graph.at(q, i).distance = result[i].distance;
            } else {
                graph.at(q, i).id = graph.EMPTY_ID;
            }
        }
    }

    void add_reverse_links(const auto &computer, int q, std::vector<std::mutex> &locks, Graph<Node> &graph) {
        for (int i = 0; i < R; i++) {
            if (graph.at(q, i).id == graph.EMPTY_ID) {
                break;
            }

            Node sn(q, graph.at(q, i).distance);
            int des = graph.at(q, i).id;

            std::vector<Node> tmp_pool;
            int dup = 0;
            {
                LockGuard guard(locks[des]);
                for (int j = 0; j < R; j++) {
                    if (graph.at(des, j).id == graph.EMPTY_ID) {
                        break;
                    }
                    if (q == graph.at(des, j).id) {
                        dup = 1;
                        break;
                    }
                    tmp_pool.push_back(graph.at(des, j));
                }
            }

            if (dup) {
                continue;
            }

            tmp_pool.push_back(sn);
            if ((int)tmp_pool.size() > R) {
                std::vector<Node> result;
                int start = 0;
                std::sort(tmp_pool.begin(), tmp_pool.end());
                result.push_back(tmp_pool[start]);

                while ((int)result.size() < R && (++start) < (int)tmp_pool.size()) {
                    auto &p = tmp_pool[start];
                    bool occlude = false;
                    for (int t = 0; t < (int)result.size(); t++) {
                        if (p.id == result[t].id) {
                            occlude = true;
                            break;
                        }
                        auto djk = computer(result[t].id, p.id);
                        if (djk < p.distance /* dik */) {
                            occlude = true;
                            break;
                        }
                    }
                    if (!occlude) {
                        result.push_back(p);
                    }
                }

                {
                    LockGuard guard(locks[des]);
                    for (int t = 0; t < (int)result.size(); t++) {
                        graph.at(des, t) = result[t];
                    }
                }

            } else {
                LockGuard guard(locks[des]);
                for (int t = 0; t < R; t++) {
                    if (graph.at(des, t).id == graph.EMPTY_ID) {
                        graph.at(des, t) = sn;
                        break;
                    }
                }
            }
        }
    }

    int32_t tree_grow(Graph<int32_t> &graph, const auto &computer) {
        int32_t n = graph.size();
        std::vector<int32_t> degrees(n);
        for (int32_t i = 0; i < n; ++i) {
            degrees[i] = graph.degree(i);
        }
        int32_t root = graph.eps[0];
        Bitset<> vis(n);
        int32_t num_attached = 0, cnt = 0;
        while (true) {
            cnt = bfs(graph, vis, root, cnt);
            if (cnt >= n) {
                break;
            }
            root = attach_unlinked(graph, computer, vis, degrees);
            num_attached += 1;
        }
        return num_attached;
    }

    int bfs(const Graph<int32_t> &graph, auto &vis, int32_t root, int32_t cnt) {
        std::queue<int> q;
        q.push(root);
        if (!vis.get(root)) {
            cnt++;
            vis.set(root);
        }
        while (q.size()) {
            int u = q.front();
            q.pop();
            for (int i = 0; i < graph.range(); ++i) {
                int v = graph.at(u, i);
                if (v == graph.EMPTY_ID) {
                    break;
                }
                if (vis.get(v)) {
                    continue;
                }
                vis.set(v);
                cnt++;
                q.push(v);
            }
        }
        return cnt;
    }

    int attach_unlinked(Graph<int32_t> &graph, const auto &computer, auto &vis, std::vector<int32_t> &degrees) {
        int id = graph.EMPTY_ID;
        for (int i = 0; i < graph.size(); ++i) {
            if (!vis.get(i)) {
                id = i;
                break;
            }
        }
        if (id == graph.EMPTY_ID) {
            return id;
        }

        LinearPool<float> pool(nb, L, L);
        std::vector<Node> fullset;
        search_on_graph(computer, id, graph, graph.eps[0], pool, fullset);
        std::sort(fullset.begin(), fullset.end());
        int node;
        bool found = false;
        for (int i = 0; i < fullset.size(); i++) {
            node = fullset[i].id;
            if (degrees[node] < R && node != id) {
                found = true;
                break;
            }
        }
        if (!found) {
            do {
                node = rand() % graph.size();
                if (vis.get(node) && degrees[node] < graph.range() && node != id) {
                    found = true;
                }
            } while (!found);
        }
        int pos = degrees[node];
        graph.at(node, pos) = id;
        degrees[node]++;
        return node;
    }

    double GetConstructionTime() const override { return construction_time; }
};

inline std::unique_ptr<Builder> create_nsg(const std::string &quantizer = "SQ4U", int32_t R = 32, int32_t L = 0,
                                           bool use_ssg = true) {
    auto qua = quantizer_map[quantizer];
    if (qua == QuantizerType::FP32) {
        return std::make_unique<NSG<FP32Quantizer<Metric::L2>>>(R, L, use_ssg);
    }
    if (qua == QuantizerType::BF16) {
        return std::make_unique<NSG<BF16Quantizer<Metric::L2>>>(R, L, use_ssg);
    }
    if (qua == QuantizerType::FP16) {
        return std::make_unique<NSG<FP16Quantizer<Metric::L2>>>(R, L, use_ssg);
    }
    if (qua == QuantizerType::SQ8U) {
        return std::make_unique<NSG<SQ8QuantizerUniform<Metric::L2>>>(R, L, use_ssg);
    }
    if (qua == QuantizerType::SQ4U) {
        return std::make_unique<NSG<SQ4QuantizerUniform<Metric::L2>>>(R, L, use_ssg);
    }
    if (qua == QuantizerType::SQ2U) {
        return std::make_unique<NSG<SQ2QuantizerUniform<Metric::L2>>>(R, L, use_ssg);
    }
    printf("Quantizer type %s not supported\n", quantizer.c_str());
    return nullptr;
}

}  // namespace glass