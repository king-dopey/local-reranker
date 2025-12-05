# local-reranker

A local reranker service with a Jina compatible API.

## Overview

This project provides a FastAPI-based web service that implements a reranking API endpoint (`/v1/rerank`) compatible with the [Jina AI Rerank API](https://jina.ai/rerank/). It allows you to host a reranking model locally for enhanced privacy and performance.

## Features

*   **Jina Compatible API**: Implements `/v1/rerank` endpoint structure.
*   **Local Hosting**: Run reranker model entirely on your own infrastructure.
*   **Multiple Backends**: Supports both PyTorch and MLX backends for optimal performance.
*   **Apple Silicon Optimization**: MLX backend provides optimized performance for M1/M2/M3 chips.
*   **Sentence Transformers**: Uses powerful `sentence-transformers` library for PyTorch backend.
*   **Configurable Model**: Easily switch between different reranker models and backends.
*   **Modern FastAPI**: Built using modern FastAPI features like `lifespan` for resource management.
*   **Async Support**: Leverages asynchronous processing for potentially better concurrency.

## Requirements

*   Python 3.9+
*   [uv](https://github.com/astral-sh/uv) (for installation and package management - recommended)
*   Sufficient RAM and compute resources (CPU or GPU) depending on the chosen reranker model.

### Backend-Specific Requirements

**PyTorch Backend:**
*   PyTorch 2.0+ (automatically installed)
*   CUDA/MPS support for GPU acceleration (optional)

**MLX Backend (Apple Silicon only):**
*   Apple Silicon (M1/M2/M3) Mac
*   MLX and MLX-LM libraries (automatically installed)
*   Optimized for memory efficiency and performance on Apple chips

## Installation

```bash
# Install with MLX support (for Apple Silicon)
uv add mlx mlx-lm safetensors

# Or install all dependencies
uv pip install -e ".[dev]"
```

## Usage

### Command Line Options

```bash
cli serve --reranker <backend> [options]
```

**Available Backends:**
*   `--reranker pytorch`: PyTorch-based reranker (default, cross-platform)
*   `--reranker mlx`: MLX-based reranker (Apple Silicon optimized)

**Other Options:**
*   `--host 0.0.0.0`: Makes the server accessible on your network. Default 127.0.0.1
*   `--port 8010`: Specifies the port (adjust if needed). Default 8010
*   `--reload`: Automatically restarts the server when code changes are detected (useful for development).

### Examples

**PyTorch Backend (default):**
```bash
cli serve --reranker pytorch --model jinaai/jina-reranker-v2-base-multilingual
```

**MLX Backend (Apple Silicon):**
```bash
cli serve --reranker mlx --model jinaai/jina-reranker-v3-mlx
```

**Configuration Management:**
```bash
# Show current configuration and available backends
cli config show
```

## Development

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd local-reranker
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # Using uv
    uv venv
    source .venv/bin/activate 
    
    # Or using standard venv
    # python -m venv .venv
    # source .venv/bin/activate 
    ```

3.  **Install the package and dependencies:**
    ```bash
    # Using uv (installs base + dev dependencies)
    uv pip install -e ".[dev]"
    ```

## Running the Server

You can run the server using `uvicorn` directly or via `uv run`:

**Method 1: Using `uvicorn`**

```bash
# Ensure your virtual environment is active
uvicorn local_reranker.api:app --host 0.0.0.0 --port 8010 --reload
```

*   `--host 0.0.0.0`: Makes the server accessible on your network.
*   `--port 8010`: Specifies the port (adjust if needed).
*   `--reload`: Automatically restarts the server when code changes are detected (useful for development).

**Method 2: Using `uv run` (handles environment implicitly)**

```bash
# From the project root directory
uv run uvicorn local_reranker.api:app --host 0.0.0.0 --port 8000 --reload
```

The server will start, and the first time it runs, it will download the default reranker model:
*   **PyTorch**: `jina-reranker-v2-base-multilingual` (~1.4GB)
*   **MLX**: `jina-reranker-v3-mlx` (~1.2GB, Apple Silicon optimized)

Model download may take some time depending on your internet connection.

## Usage

Once the server is running, you can send requests to the `/v1/rerank` endpoint. Here's an example using `curl`:

```bash
curl -X POST "http://localhost:8010/v1/rerank" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "jina-reranker-v2-base-multilingual", 
           "query": "What are the benefits of using FastAPI?", 
           "documents": [
             "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.",
             "Django is a high-level Python Web framework that encourages rapid development and clean, pragmatic design.",
             "The key features are: Fast, Fast to code, Fewer bugs, Intuitive, Easy, Short, Robust, Standards-based.",
             "Flask is a micro web framework written in Python."
           ],
           "top_n": 3,
           "return_documents": true
         }'
```

**Parameters:**

*   `model`: (Currently ignored by the API, uses the default) The name of the reranker model.
*   `query`: The search query string.
*   `documents`: A list of strings or dictionaries (`{"text": "..."}`) to be reranked against the query.
*   `top_n`: (Optional) The maximum number of results to return.
*   `return_documents`: (Optional, default `False`) Whether to include the document text in the results.

## Testing

Tests are implemented using `pytest`. To run the tests:

1.  Make sure you have installed the development dependencies (`uv pip install -e ".[dev]"`).
2.  Ensure your virtual environment is active or use `uv run`.

```bash
# Ensure venv is active
python -m pytest

# Or using uv run
uv run pytest
```

## MLX Backend Troubleshooting

### Common Issues

**MLX not found:**
```bash
# Ensure you're on Apple Silicon
uname -m  # Should show arm64

# Install MLX dependencies
uv add mlx mlx-lm safetensors
```

**Model download fails:**
```bash
# Check internet connection
# Try manual download
huggingface-cli download jinaai/jina-reranker-v3-mlx
```

**Performance issues:**
```bash
# Check MLX is using GPU (if available)
python -c "import mlx; print(mlx.metal.is_available())"

# Monitor memory usage
top -o mem | grep python
```

### Environment Variables

```bash
# Force MLX backend
export RERANKER_RERANKER_TYPE=mlx

# Custom model path
export RERANKER_MODEL_NAME=custom-mlx-model
```
