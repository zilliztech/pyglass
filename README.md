# Graph Library for Approximate Similarity Search

pyglass is a library for fast inference of graph index for approximate similarity search.

## Features

- Supports multiple graph algorithms, like [**HNSW**](https://github.com/nmslib/hnswlib) and [**NSG**](https://github.com/ZJULearning/nsg).
- Supports multiple hardware platforms, like **X86** and **ARM**. Support for **GPU** is on the way
- No third-party library dependencies, does not rely on OpenBLAS / MKL or any other computing framework.
- Sophisticated memory management and data structure design, very low memory footprint.
- It's high performant.

## Installation

### Installation from Source
It is recommended to use `uv` for setting up the development environment.

1. **Install `uv`**
```bash
pip install uv
```

2. **Create and Activate Virtual Environment**
```bash
# Create the virtual environment in a .venv directory
uv venv --seed --python 3.11
# Activate the environment (on bash/zsh)
source .venv/bin/activate
```
*Note: For other shells like Fish or Powershell, use the corresponding activation script in the `.venv/bin/` directory.*

3. **Install the Project**
```bash
# Install the project and its dependencies in editable mode
uv pip install -v -e "python"
```

## Quick Tour
A runnable demo is at [examples/demo.ipynb](https://github.com/zilliztech/pyglass/blob/master/examples/demo.ipynb). It's highly recommended to try it.

## Usage
**Import library**
```python
>>> import glass
```
**Load Data**
```python
>>> import numpy as np
>>> n, d = 10000, 128
>>> X = np.random.randn(n, d)
>>> Y = np.random.randn(d)
```
**Create Index**
pyglass supports **HNSW** and **NSG** index currently, with different quantization support
```python
>>> index = glass.Index(index_type="HNSW", metric="L2", R=32, L=50)
>>> index = glass.Index(index_type="NSG", metric="L2", R=32, L=50, quant="SQ8U")
```
**Build Graph**
```python
>>> graph = index.build(X)
```
**Create Searcher**
```python
>>> searcher = glass.Searcher(graph=graph, data=X, metric="L2", quantizer="SQ4U")
>>> searcher.set_ef(32)
```
**(Optional) Optimize Searcher**
```python
>>> searcher.optimize()
```
**Searching**
```python
>>> ret = searcher.search(query=Y, k=10)
>>> print(ret)
```

## Supported Quantization Methods

- FP8_E5M2
- PQ8
- SQ8
- SQ8U
- SQ6
- SQ4
- SQ4U
- SQ4UA
- SQ2U
- BinaryQuant

**Rule of Thumb**: Use SQ8U for indexing, and SQ4U for searching is almost always a good choice.

## Performance

Glass is among one of the top performant ann algorithms on [ann-benchmarks](https://ann-benchmarks.com/)

### fashion-mnist-784-euclidean
![](docs/figures/fashion-mnist-784-euclidean_10_euclidean.png)
### gist-960-euclidean
![](docs/figures/gist-960-euclidean_10_euclidean.png)
### sift-128-euclidean
![](docs/figures/sift-128-euclidean_10_euclidean.png)

### Quick Benchmark

1. Change configuration file `examples/config.json`
2. Run benchmark
```
python3 examples/main.py
```
If you encounter network issues when downloading datasets, you can try setting the `HF_ENDPOINT` environment variable to `https://hf-mirror.com` or something else.

## Citation

You can cite the PyGlass repo as follows:
```bibtex
@misc{PyGlass,
    author = {Zihao Wang},
    title = {Graph Library for Approximate Similarity Search},
    url = {https://github.com/zilliztech/pyglass},
    year = {2025},
    month = {4},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hhy3/pyglass&type=Date)](https://www.star-history.com/#hhy3/pyglass&Date)