//
// Created by weijian on 10/8/24.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <string>
#include <cstring>
#include <set>
#include <format>
#include <filesystem>

#include "Eigen/Dense"

#include "glass/searcher.hpp"
#include "glass/quant/fp32_quant.hpp"
#include "glass/quant/sq8_quant.hpp"
#include "glass/quant/sq4_quant.hpp"
#include "glass/quant/to_quant.hpp"
#include "glass/nsg/nsg.hpp"
#include "glass/hnsw/hnsw.hpp"

#include "io_util.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
namespace fs = std::filesystem;
using namespace Eigen;


int main() {
    cout << "Explore whether it is feasible to use sq4 for image search and then rerank?" << endl;
    string dataset = "sift1m";
    vector<float> data{};
    vector<float> queries{};
    vector<vector<int>> GT{};
    int R = 64, L = 128;
    int nb = 0, d = 0;
    int nq = 0;
    int k = 10;
    int nthr = 1;
    string index_path = std::format("/data/raid0/{}/{}_R{}_L{}", dataset, dataset, R, L);
    bool update_index = !fs::exists(index_path);
    read_dataset(dataset, data, nb, queries, nq, d, GT);

    glass::HNSW index(d, "L2", R, L);
    if (update_index) {
        index.Build(data.data(), nb);
        index.final_graph.save(index_path);
    } else {
        index.final_graph.load(index_path);
    }

    glass::Searcher<glass::FP32Quantizer<glass::Metric::L2>> searcher(index.final_graph);
    searcher.SetData(data.data(), nb, d);
    glass::FP32Quantizer<glass::Metric::L2> raw_data(d);
    raw_data.train(data.data(), nb);

    std::vector<int> efs{10, 20, 30, 40, 50, 100, 150, 200, 250, 300};
    for (auto ef : efs) {
        searcher.SetEf(ef);
//        searcher.Optimize(nthr);
        vector<vector<int>> output(nq, vector<int>(k, 0));

        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(nthr)
        for (int i = 0; i < nq; i++) {
            searcher.Search(queries.data() + i * d, k, output[i].data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double mean_latency = static_cast<double>(duration_microseconds) / nq;
        double duration_seconds = static_cast<double>(duration_microseconds) / 1'000'000;
        double qps = static_cast<double>(nq) / duration_seconds;
        cout << "mean latency: " << mean_latency << "us, qps: " << qps << endl;

        std::atomic<int> total_coselection{0};
        std::atomic<int> total_num{0};
        std::vector<double> recalls{};
//#pragma omp parallel for
        for (int i = 0; i < nq; i++) {
            int cur_coselection = 0;
            std::set gt(GT[i].begin(), GT[i].begin() + k);
            std::set res(output[i].begin(), output[i].begin() + k);
            for (auto item: res) {
                if (gt.find(static_cast<int64_t>(item)) != gt.end()) {
                    cur_coselection++;
                }
            }
            total_num += 1;
            total_coselection += cur_coselection;
            recalls.push_back((double) cur_coselection * 100 / k);
        }

        std::cout << "R = " << R << ", L = " << L << ", ef = " << ef << ", recall = "
                  << (double) total_coselection * 100 / (total_num * k) << " %" << std::endl;
        cout << "-----------------------------" << endl;
    }
}