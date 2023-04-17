#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thread>

#include <omp.h>

#include "glass/builder.hpp"
#include "glass/hnsw/hnsw.hpp"
#include "glass/nsg/nsg.hpp"
#include "glass/searcher.hpp"

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

  explicit Graph(const Graph &rhs) : graph(rhs.graph) {}

  explicit Graph(const std::string &filename) { graph.load(filename); }

  explicit Graph(const glass::Graph<int> &graph) : graph(graph) {}

  void save(const std::string &filename) { graph.save(filename); }

  void load(const std::string &filename) { graph.load(filename); }
};

struct Index {
  std::unique_ptr<glass::Builder> index = nullptr;

  Index(const std::string &index_type, int dim, const std::string &metric,
        int R = 32, int L = 200) {
    if (index_type == "NSG") {
      index = std::unique_ptr<glass::Builder>(
          (glass::Builder *)new glass::NSG(dim, metric, R, L));
    } else if (index_type == "HNSW") {
      index = std::unique_ptr<glass::Builder>(
          (glass::Builder *)new glass::HNSW(dim, metric, R, L));
    } else {
      printf("Index type [%s] not supported\n", index_type.c_str());
    }
  }

  Graph build(py::object input) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();
    size_t rows, features;
    get_input_array_shapes(buffer, &rows, &features);
    float *vector_data = (float *)items.data(0);
    index->Build(vector_data, rows);
    return Graph(index->GetGraph());
  }
};

struct Searcher {

  std::unique_ptr<glass::SearcherBase> searcher;

  Searcher(const Graph &graph, py::object input, const std::string &metric,
           int level)
      : searcher(std::unique_ptr<glass::SearcherBase>(
            glass::create_searcher(graph.graph, metric, level))) {
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
    ids = new int[k];
    searcher->Search(items.data(0), k, ids);
    py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    return py::array_t<int>({k}, {sizeof(int)}, ids, free_when_done);
  }

  py::object batch_search(py::object query, int k, int num_threads = 0) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    auto buffer = items.request();
    int *ids;
    size_t nq, dim;
    {
      py::gil_scoped_release l;
      get_input_array_shapes(buffer, &nq, &dim);
      ids = new int[nq * k];
      if (num_threads != 0) {
        omp_set_num_threads(num_threads);
      }
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < nq; ++i) {
        searcher->Search(items.data(i), k, ids + i * k);
      }
    }
    py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    return py::array_t<int>({nq * k}, {sizeof(int)}, ids, free_when_done);
  }

  void set_ef(int ef) { searcher->SetEf(ef); }

  void optimize(int num_threads = 0) { searcher->Optimize(num_threads); }
};

PYBIND11_PLUGIN(glassppy) {
  py::module m("glassppy");

  m.def("set_num_threads", &set_num_threads, py::arg("num_threads"));

  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("filename"))
      .def("save", &Graph::save, py::arg("filename"))
      .def("load", &Graph::load, py::arg("filename"));

  py::class_<Index>(m, "Index")
      .def(py::init<const std::string &, int, const std::string &, int, int>(),
           py::arg("index_type"), py::arg("dim"), py::arg("metric"),
           py::arg("R") = 32, py::arg("L") = 0)
      .def("build", &Index::build, py::arg("data"));

  py::class_<Searcher>(m, "Searcher")
      .def(py::init<const Graph &, py::object, const std::string &, int>(),
           py::arg("graph"), py::arg("data"), py::arg("metric"),
           py::arg("level"))
      .def("set_ef", &Searcher::set_ef, py::arg("ef"))
      .def("search", &Searcher::search, py::arg("query"), py::arg("k"))
      .def("batch_search", &Searcher::batch_search, py::arg("query"),
           py::arg("k"), py::arg("num_threads") = 0)
      .def("optimize", &Searcher::optimize, py::arg("num_threads") = 0);

  return m.ptr();
}
