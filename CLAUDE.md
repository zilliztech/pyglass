# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyGlass is a high-performance C++ library with Python bindings for approximate similarity search using graph-based algorithms (HNSW, NSG, SSG, RNNDESCENT, IVF). The core is implemented in C++ with SIMD optimizations, and Python bindings are provided via pybind11.

## Essential Commands

### Build and Installation
```bash
# Development installation (editable)
pip install -v -e "python"

# Using uv (recommended for development)
uv venv
source .venv/bin/activate
uv pip install -v -e "python"

# Generate C++ compile_commands.json
bear -- uv pip install -v -e "python"
```

### Testing
```bash
pytest "python"
```

### Code Quality
```bash
# Python formatting
black .

# Python linting
ruff check .

# C++ formatting
clang-format -i $(find glass -name "*.hpp")
```

### Benchmarking
```bash
# Run with default configuration
python3 examples/main.py

# Run with custom configuration
python3 examples/main.py path/to/config.yaml
```

## Architecture

### Core Components

1. **Graph Algorithms** (`glass/hnsw/`, `glass/nsg/`): HNSW and NSG graph implementations for similarity search
2. **Quantization** (`glass/quant/`): Multiple quantization methods (FP16, SQ8, PQ, etc.) for memory efficiency
3. **SIMD Optimizations** (`glass/simd/`): Platform-specific optimizations for AVX2, AVX512, and ARM NEON
4. **Search Layer** (`glass/searcher/`): Unified search interface over different index types

### Python Integration

- **Bindings**: `python/bindings.cc` uses pybind11 to expose C++ classes
- **Builder Pattern**: Python API uses builder pattern for index construction
- **Dataset Utils**: `python/ann_dataset/` handles standard ANN benchmark datasets

### Key Design Patterns

1. **Template-Heavy C++**: Extensive use of C++ templates for performance and flexibility
2. **Memory Management**: Custom memory allocators and storage classes for efficient data handling
3. **Parallel Processing**: OpenMP for multi-threaded operations
4. **Builder Pattern**: Index construction uses fluent builder API

## Development Notes

- C++20 standard required
- No external C++ dependencies (self-contained)
- Performance is critical - always benchmark changes
- Use `examples/config.yaml` to configure benchmark parameters
- Dataset downloads may require `export HF_ENDPOINT=https://hf-mirror.com` for network issues