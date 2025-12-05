# -*- coding: utf-8 -*-
"""Tests for MLX reranker implementation."""

import pytest
from unittest.mock import Mock, patch

from local_reranker.reranker import Reranker as RerankerProtocol
from local_reranker.models import RerankRequest

# Skip MLX tests if MLX is not installed
mlx_available = True
try:
    import mlx.core
    import mlx.nn
    import mlx_lm
    import numpy
    import safetensors
except ImportError:
    mlx_available = False

if mlx_available:
    from local_reranker.reranker_mlx import Reranker as MLXReranker


@pytest.mark.skipif(not mlx_available, reason="MLX dependencies not installed")
class TestMLXReranker:
    """Test MLX implementation of reranker protocol."""

    def test_mlx_reranker_implements_protocol(self):
        """Test that MLX reranker implements the protocol."""
        # This is a simple protocol compliance test without complex mocking
        assert hasattr(MLXReranker, "__annotations__")
        assert hasattr(MLXReranker, "rerank")
        assert hasattr(MLXReranker, "__init__")

    @patch("huggingface_hub.snapshot_download")
    def test_initialization_runtime_error(self, mock_snapshot_download):
        """Test handling of runtime errors during initialization."""
        mock_snapshot_download.side_effect = Exception("Download failed")

        with pytest.raises(RuntimeError, match="Failed to load MLX model"):
            MLXReranker()
