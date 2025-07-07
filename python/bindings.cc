#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thread>

#include <omp.h>

#include "glass/algorithms/clustering.hpp"
#include "glass/builder.hpp"
#include "glass/hnsw/hnsw.hpp"
#include "glass/nsg/nsg.hpp"
#include "glass/nsg/rnndescent.hpp"
#include "glass/searcher/graph_searcher.hpp"

namespace py = pybind11;
using namespace pybind11::literals; // needed to bring in _a literal

inline void get_input_array_shapes(const py::buffer_info &buffer, size_t *rows,
                                   size_t *features) {
  if (buffer.ndim != 2 && buffer.ndim != 1) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "Input vector data wrong shape. Number of dimensions %d. Data "
             "must be a 1D or 2D array.",
             buffer.ndim);
  }
  if (buffer.ndim == 2) {
    *rows = buffer.shape[0];
    *features = buffer.shape[1];
  } else {
    *rows = 1;
    *features = buffer.shape[0];
  }
}

void set_num_threads(int num_threads) { omp_set_num_threads(num_threads); }

struct Graph {
  glass::Graph<int> graph;

  Graph() = default;

  explicit Graph(Graph &&rhs) : graph(std::move(rhs.graph)) {}

  Graph(const std::string &filename, const std::string &format = "glass") {
    graph.load(filename, format);
  }

  explicit Graph(glass::Graph<int> graph) : graph(std::move(graph)) {}

  void save(const std::string &filename) { graph.save(filename); }

  void load(const std::string &filename, const std::string &format = "glass") {
    graph.load(filename, format);
  }
};

struct Index {
  std::unique_ptr<glass::Builder> index = nullptr;

  Index(const std::string &index_type, const std::string &metric,
        const std::string &quant = "FP32", int R = 32, int L = 200) {
    if (index_type == "HNSW") {
      index = glass::create_hnsw(metric, quant, R, L);
    } else if (index_type == "NSG") {
      index = glass::create_nsg(quant, R, L, false);
    } else if (index_type == "SSG") {
      index = glass::create_nsg(quant, R, L, true);
    } else if (index_type == "RNNDESCENT") {
      index = glass::create_rnndescent(quant, R);
    } else {
      printf("Unknown index type\n");
    }
  }

  Graph build(py::object input) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();
    size_t rows, features;
    get_input_array_shapes(buffer, &rows, &features);
    float *vector_data = (float *)items.data(0);
    return Graph(index->Build(vector_data, rows, features));
  }

  double get_construction_time() const { return index->GetConstructionTime(); }
};

struct Searcher {

  std::unique_ptr<glass::GraphSearcherBase> searcher;

  Searcher(Graph &graph, py::object input, const std::string &metric,
           const std::string &quantizer, const std::string &refine_quant = "")
      : searcher(
            std::unique_ptr<glass::GraphSearcherBase>(glass::create_searcher(
                std::move(graph.graph), metric, quantizer, refine_quant))) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();
    size_t rows, features;
    get_input_array_shapes(buffer, &rows, &features);
    float *vector_data = (float *)items.data(0);
    searcher->SetData(vector_data, rows, features);
  }

  py::object search(py::object query, int k) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    int *ids;
    float *dis;
    {
      py::gil_scoped_release l;
      ids = new int[k];
      dis = new float[k];
      searcher->Search(items.data(0), k, ids, dis);
    }
    py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    py::capsule free_when_done_dis(dis, [](void *f) { delete[] f; });
    return py::make_tuple(
        py::array_t<int32_t>({(size_t)k}, {sizeof(int32_t)}, ids, free_when_done),
        py::array_t<float>({(size_t)k}, {sizeof(float)}, dis, free_when_done_dis));
  }

  py::object batch_search(py::object query, int k, int num_threads = 0) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    auto buffer = items.request();
    int32_t *ids;
    float *dis;
    size_t nq, dim;
    {
      py::gil_scoped_release l;
      get_input_array_shapes(buffer, &nq, &dim);
      ids = new int[nq * k];
      dis = new float[nq * k];
      if (num_threads != 0) {
        omp_set_num_threads(num_threads);
      }
      searcher->SearchBatch(items.data(0), nq, k, ids, dis);
    }
    py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    py::capsule free_when_done_dis(dis, [](void *f) { delete[] f; });
    return py::make_tuple(
        py::array_t<int32_t>({(size_t)nq, (size_t)k}, {k * sizeof(int32_t), sizeof(int32_t)}, ids, free_when_done),
        py::array_t<float>({(size_t)nq, (size_t)k}, {k * sizeof(float), sizeof(float)}, dis, free_when_done_dis));
  }

  void set_ef(int ef) { searcher->SetEf(ef); }

  void optimize(int num_threads = 0) { searcher->Optimize(num_threads); }

  double get_last_search_avg_dist_cmps() const {
    return searcher->GetLastSearchAvgDistCmps();
  }
};

struct Clustering {

  glass::Clustering internal;

  Clustering(int32_t n_cluster, int32_t epochs = 10,
             const std::string &init = "random")
      : internal(n_cluster, epochs, init) {}

  void fit(py::object data) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(data);
    auto buffer = items.request();
    size_t n, dim;
    {
      py::gil_scoped_release l;
      get_input_array_shapes(buffer, &n, &dim);
      internal.fit(items.data(0), n, dim);
    }
  }

  py::object predict(py::object data) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(data);
    auto buffer = items.request();
    std::vector<int32_t> ids;
    size_t n, dim;
    {
      py::gil_scoped_release l;
      get_input_array_shapes(buffer, &n, &dim);
      ids = internal.predict(items.data(0), n, dim);
    }
    return py::array(ids.size(), ids.data());
  }

  py::object transform(py::object data, bool inplace = false) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(data);
    auto buffer = items.request();
    size_t n, dim;
    {
      py::gil_scoped_release l;
      get_input_array_shapes(buffer, &n, &dim);
      if (inplace) {
        internal.transform((float *)items.data(0), n, dim);
        return py::none();
      } else {
        // TODO
        return py::none();
      }
    }
  }
};

#define DECLARE_QUANT(py_name, cxx_name)                                       \
  struct py_name {                                                             \
    glass::cxx_name<glass::Metric::L2> quant;                                  \
    py_name(int32_t dim) : quant(dim) {}                                       \
                                                                               \
    void train(py::object data) {                                              \
      py::array_t<float, py::array::c_style | py::array::forcecast> items(     \
          data);                                                               \
      auto buffer = items.request();                                           \
      size_t n, dim;                                                           \
      {                                                                        \
        py::gil_scoped_release l;                                              \
        get_input_array_shapes(buffer, &n, &dim);                              \
        quant.train(items.data(0), n);                                         \
      }                                                                        \
    }                                                                          \
  };

DECLARE_QUANT(quant_fp32, FP32Quantizer)
DECLARE_QUANT(quant_bf16, BF16Quantizer)
DECLARE_QUANT(quant_fp16, FP16Quantizer)
DECLARE_QUANT(quant_e5m2, E5M2Quantizer)
DECLARE_QUANT(quant_sq8, SQ8Quantizer)
DECLARE_QUANT(quant_sq8u, SQ8QuantizerUniform)

double build_graph(const std::string &index_type, py::object input,
                   const std::string &graph_file, const std::string &metric,
                   const std::string &quant = "FP32", int R = 32, int L = 200) {
  Index index(index_type, metric, quant, R, L);
  index.build(input).save(graph_file);
  return index.get_construction_time();
}

PYBIND11_MODULE(glass, m) {
  m.def("set_num_threads", &set_num_threads, py::arg("num_threads"));

  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("filename"), py::arg("format") = "glass")
      .def("save", &Graph::save, py::arg("filename"))
      .def("load", &Graph::load, py::arg("filename"),
           py::arg("format") = "glass");

  py::class_<Index>(m, "Index")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, int, int>(),
           py::arg("index_type"), py::arg("metric"), py::arg("quant") = "FP32",
           py::arg("R") = 32, py::arg("L") = 0)
      .def("build", &Index::build, py::arg("data"))
      .def("get_construction_time", &Index::get_construction_time);

  py::class_<Searcher>(m, "Searcher")
      .def(py::init<Graph &, py::object, const std::string &,
                    const std::string &, const std::string &>(),
           py::arg("graph"), py::arg("data"), py::arg("metric"),
           py::arg("quantizer"), py::arg("refine_quant") = "")
      .def("set_ef", &Searcher::set_ef, py::arg("ef"))
      .def("search", &Searcher::search, py::arg("query"), py::arg("k"))
      .def("batch_search", &Searcher::batch_search, py::arg("query"),
           py::arg("k"), py::arg("num_threads") = 0)
      .def("optimize", &Searcher::optimize, py::arg("num_threads") = 0)
      .def("get_last_search_dist_cmps",
           &Searcher::get_last_search_avg_dist_cmps);

  py::class_<Clustering>(m, "Clustering")
      .def(py::init<int32_t, int32_t, const std::string &>(),
           py::arg("n_cluster"), py::arg("epochs") = 10,
           py::arg("init") = "random")
      .def("fit", &Clustering::fit, py::arg("data"))
      .def("predict", &Clustering::predict, py::arg("data"))
      .def("transform", &Clustering::transform, py::arg("data"),
           py::arg("inplace") = false);

#define PYDECLARE_QUANT(py_name)                                               \
  py::class_<py_name>(m, #py_name)                                             \
      .def(py::init<int32_t>(), py::arg("dim"))                                \
      .def("train", &py_name::train, py::arg("data"));

  PYDECLARE_QUANT(quant_fp32)
  PYDECLARE_QUANT(quant_bf16)
  PYDECLARE_QUANT(quant_fp16)
  PYDECLARE_QUANT(quant_e5m2)
  PYDECLARE_QUANT(quant_sq8)
  PYDECLARE_QUANT(quant_sq8u)

  m.def("build_graph", &build_graph, py::arg("index_type"), py::arg("input"),
        py::arg("graph_file"), py::arg("metric"), py::arg("quant"),
        py::arg("R"), py::arg("L"));
}
