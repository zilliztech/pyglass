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
#include <map>
#include <unordered_map>
#include <iomanip> // for std::setw

#include "glass/searcher.hpp"
#include "glass/quant/fp32_quant.hpp"
#include "glass/hnsw/hnsw.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
template<class T>
void ReadBin(const std::string &file_path,
             std::vector<std::vector<T>> &data) {
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    unsigned N;  // num of points
    unsigned num_dimensions;
    ifs.read((char *)&N, sizeof(unsigned));
    ifs.read((char *)&num_dimensions, sizeof(unsigned));
    data.resize(N);
    std::cout << "# of points: " << N << std::endl;
    std::cout << "# of dimensions: " << num_dimensions << std::endl;
    std::vector<T> buff(num_dimensions);
    int counter = 0;
    for(int i = 0; i < N; i++) {
        ifs.read((char *)buff.data(), num_dimensions * sizeof(T));
        std::vector<T> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++) {
            row[d] = static_cast<T>(buff[d]);
        }
        data[counter++] = std::move(row);
    }
    ifs.close();
    std::cout << "Finish Reading Data" << endl;
}

void printDistribution(const std::vector<int>& in_degree) {
    // 统计频率
    std::map<int, int> frequency_map;
    for (int value : in_degree) {
        frequency_map[value]++;
    }

    // 打印分布
    std::cout << "Value Distribution:" << std::endl;
    for (const auto& pair : frequency_map) {
        std::cout << "Value: " << std::setw(5) << pair.first
                  << " | Frequency: " << std::setw(5) << pair.second << std::endl;
    }
}

int main() {
    cout << "Explore the distribution of neighbors of each vector in buckets" << endl;
    string data_file = "/data/raid0/sift1m/sift1m_base.fbin";
    float ratio = 0.001;  // posting list len = 500
    vector<vector<float>> data{}, sub_data{};
    ReadBin(data_file, data);
    int n = data.size(), d = data.front().size();
    int m = n * ratio;
    vector<int> bucket_ids(n, 0);
    vector<int> in_degree(n, 0);
    std::map<int, int> bucket_length{};

    {
        vector<int> subset_ids(n);
        std::iota(subset_ids.begin(), subset_ids.end(), 0);
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(subset_ids.begin(), subset_ids.end(), std::default_random_engine(seed));
        subset_ids.resize(m);

        sub_data.reserve(m);
        for(auto id : subset_ids) {
            sub_data.emplace_back(data[id]);
        }
    }

    {
        vector<float> data_buf(m * d);
#pragma omp parallel for
        for (int i = 0; i < m; i++) {
            std::memcpy(data_buf.data() + i * d, sub_data[i].data(), d * 4);
        }
        glass::HNSW index(d, "L2", 64, 128);
        index.Build(data_buf.data(), m);
        glass::Searcher<glass::FP32Quantizer<glass::Metric::L2>> searcher(index.final_graph);
        searcher.SetData(data_buf.data(), m, d);
        searcher.SetEf(128);
#pragma omp parallel for schedule(dynamic) num_threads(8)
        for (int i = 0; i < n; i++) {
            searcher.Search(data[i].data(), 1, bucket_ids.data() + i);
        }
        for(auto bucked_id : bucket_ids) {
            bucket_length[bucked_id]++;
        }
    }

    vector<int> nc_bucket_num(n, 0);

    {
        vector<float> data_buf(n * d);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            std::memcpy(data_buf.data() + i * d, data[i].data(), d * 4);
        }
        glass::HNSW index(d, "L2");
        index.Build(data_buf.data(), n);
//#pragma omp parallel for
        for(int i = 0; i < n; i++) {
            std::set<int> bucket{};
            int *nbr = index.final_graph.edges(i);
            for(int j = 0; j < index.final_graph.K; j++) {
                bucket.insert(bucket_ids[nbr[j]]);
                in_degree[nbr[j]]++;
            }
            nc_bucket_num[i] = bucket.size();
        }
        cout << "K = " << index.final_graph.K << endl;
    }

    printDistribution(in_degree);

//    {
//        std::set<int> bucket{};
//        for(auto bucket_id : bucket_ids) {
//            bucket.insert(bucket_id);
//        }
//        cout << "Bucket num = " << bucket.size() << endl;
//    }
//
//
//    for(int i = 0; i < 100; i++) {
//        cout << nc_bucket_num[i] << ",";
//    }
//    cout << endl;
//
//    for(auto [_, len] : bucket_length) {
//        cout << len << ",";
//    }

//    {
//        vector<int> distribute(33, 0);
//        for(auto idx : nc_bucket_num) {
//            distribute[idx]++;
//        }
//        for(int i = 0; i <= 32; i++) {
//            cout << "bucket num: " << i << ", vector num: " << distribute[i] << endl;
//        }
//    }


}