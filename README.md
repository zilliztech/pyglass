# Graph Library for Approximate Nearest Search

pyglass is a library for fast inference of graph index for approximate nearest search.

## Installation
### (Recommanded)Installation from Wheel
pyglass can be installed using pip as follows:
```bash
pip3 install glassppy
```

### Installation from Source
``` bash
sudo apt-get update && sudo apt-get install -y build-essential git python3 python3-distutils python3-venv
```
``` bash
pip3 install pybind11
```
``` bash
bash build.sh
```

## Quick Tour

```python
>>> import glassppy as glass
>>> import numpy as np
>>> n, d = 10000, 128
>>> X = np.random.randn(n, d)
>>> Y = np.random.randn(d)
>>> index = glass.Index("HNSW", dim=d, metric="L2", R=32, L=50)
>>> graph = index.build(X)
>>> searcher = glass.Searcher(graph, X, "L2", 0)
>>> searcher.optimize()
>>> searcher.set_ef(32)
>>> print(searcher.search(Y, 10))
```