# -*- coding: utf-8 -*-
"""Tests for MLX reranker implementation."""

import pytest
from unittest.mock import Mock, patch

from local_reranker.reranker import Reranker as RerankerProtocol
from local_reranker.reranker_mlx import Reranker as MLXReranker
from local_reranker.models import RerankRequest


class TestMLXReranker:
    """Test MLX implementation of reranker protocol."""

    @patch("local_reranker.reranker_mlx.snapshot_download")
    @patch("local_reranker.reranker_mlx.importlib.util")
    def test_initialization_with_default_params(
        self, mock_importlib, mock_snapshot_download
    ):
        """Test MLX reranker initialization with default parameters."""
        mock_snapshot_download.return_value = "/mock/model/path"

        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_importlib.spec_from_file_location.return_value = mock_spec
        mock_importlib.module_from_spec.return_value = Mock()

        mock_mlx_reranker_class = Mock()
        mock_module = Mock()
        mock_module.MLXReranker = mock_mlx_reranker_class
        mock_importlib.module_from_spec.return_value = mock_module

        with (
            patch("local_reranker.reranker_mlx.mx"),
            patch("local_reranker.reranker_mlx.nn"),
            patch("local_reranker.reranker_mlx.load"),
            patch("local_reranker.reranker_mlx.np"),
            patch("local_reranker.reranker_mlx.safe_open"),
        ):
            reranker = MLXReranker()

            assert reranker.model_name == "jinaai/jina-reranker-v3-mlx"
            assert reranker.device is None
            mock_snapshot_download.assert_called_once_with(
                repo_id="jinaai/jina-reranker-v3-mlx",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "rerank.py"],
            )
            mock_mlx_reranker_class.assert_called_once()

    @patch("local_reranker.reranker_mlx.snapshot_download")
    @patch("local_reranker.reranker_mlx.importlib.util")
    def test_initialization_with_custom_params(
        self, mock_importlib, mock_snapshot_download
    ):
        """Test MLX reranker initialization with custom parameters."""
        mock_snapshot_download.return_value = "/custom/model/path"

        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_importlib.spec_from_file_location.return_value = mock_spec
        mock_importlib.module_from_spec.return_value = Mock()

        mock_mlx_reranker_class = Mock()
        mock_module = Mock()
        mock_module.MLXReranker = mock_mlx_reranker_class
        mock_importlib.module_from_spec.return_value = mock_module

        with (
            patch("local_reranker.reranker_mlx.mx"),
            patch("local_reranker.reranker_mlx.nn"),
            patch("local_reranker.reranker_mlx.load"),
            patch("local_reranker.reranker_mlx.np"),
            patch("local_reranker.reranker_mlx.safe_open"),
        ):
            reranker = MLXReranker(model_name="custom-mlx-model", device="ignored")

            assert reranker.model_name == "custom-mlx-model"
            assert reranker.device == "ignored"
            mock_snapshot_download.assert_called_once_with(
                repo_id="custom-mlx-model",
                allow_patterns=["*.safetensors", "*.json", "*.txt", "rerank.py"],
            )

    @patch("local_reranker.reranker_mlx.snapshot_download")
    def test_initialization_import_error(self, mock_snapshot_download):
        """Test handling of MLX import errors."""
        mock_snapshot_download.return_value = "/mock/model/path"

        with patch("local_reranker.reranker_mlx.importlib.util"):
            with pytest.raises(ImportError, match="MLX dependencies not found"):
                MLXReranker()

    @patch("local_reranker.reranker_mlx.snapshot_download")
    def test_initialization_runtime_error(self, mock_snapshot_download):
        """Test handling of runtime errors during initialization."""
        mock_snapshot_download.side_effect = Exception("Download failed")

        with (
            patch("local_reranker.reranker_mlx.mx"),
            patch("local_reranker.reranker_mlx.nn"),
            patch("local_reranker.reranker_mlx.load"),
            patch("local_reranker.reranker_mlx.np"),
            patch("local_reranker.reranker_mlx.safe_open"),
        ):
            with pytest.raises(RuntimeError, match="Failed to load MLX model"):
                MLXReranker()

    @patch("local_reranker.reranker_mlx.snapshot_download")
    @patch("local_reranker.reranker_mlx.importlib.util")
    def test_rerank_with_string_documents(self, mock_importlib, mock_snapshot_download):
        """Test MLX reranking with string documents."""
        mock_snapshot_download.return_value = "/mock/model/path"

        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_importlib.spec_from_file_location.return_value = mock_spec
        mock_importlib.module_from_spec.return_value = Mock()

        mock_mlx_results = [
            {"document": "doc1", "relevance_score": 0.9, "index": 0, "embedding": None},
            {"document": "doc2", "relevance_score": 0.7, "index": 1, "embedding": None},
            {"document": "doc3", "relevance_score": 0.8, "index": 2, "embedding": None},
        ]

        mock_mlx_reranker_instance = Mock()
        mock_mlx_reranker_instance.rerank.return_value = mock_mlx_results

        mock_mlx_reranker_class = Mock()
        mock_mlx_reranker_class.return_value = mock_mlx_reranker_instance

        mock_module = Mock()
        mock_module.MLXReranker = mock_mlx_reranker_class
        mock_importlib.module_from_spec.return_value = mock_module

        with (
            patch("local_reranker.reranker_mlx.mx"),
            patch("local_reranker.reranker_mlx.nn"),
            patch("local_reranker.reranker_mlx.load"),
            patch("local_reranker.reranker_mlx.np"),
            patch("local_reranker.reranker_mlx.safe_open"),
        ):
            reranker = MLXReranker()
            request = RerankRequest(
                query="test query", documents=["doc1", "doc2", "doc3"]
            )

            results = reranker.rerank(request)

            assert len(results) == 3
            assert results[0].index == 0 and results[0].relevance_score == 0.9
            assert results[1].index == 2 and results[1].relevance_score == 0.8
            assert results[2].index == 1 and results[2].relevance_score == 0.7
            mock_mlx_reranker_instance.rerank.assert_called_once_with(
                query="test query",
                documents=["doc1", "doc2", "doc3"],
                top_n=None,
                return_embeddings=False,
            )

    @patch("local_reranker.reranker_mlx.snapshot_download")
    @patch("local_reranker.reranker_mlx.importlib.util")
    def test_rerank_with_dict_documents(self, mock_importlib, mock_snapshot_download):
        """Test MLX reranking with dictionary documents."""
        mock_snapshot_download.return_value = "/mock/model/path"

        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_importlib.spec_from_file_location.return_value = mock_spec
        mock_importlib.module_from_spec.return_value = Mock()

        mock_mlx_results = [
            {"document": "doc1", "relevance_score": 0.9, "index": 0, "embedding": None},
            {"document": "doc2", "relevance_score": 0.7, "index": 1, "embedding": None},
        ]

        mock_mlx_reranker_instance = Mock()
        mock_mlx_reranker_instance.rerank.return_value = mock_mlx_results

        mock_mlx_reranker_class = Mock()
        mock_mlx_reranker_class.return_value = mock_mlx_reranker_instance

        mock_module = Mock()
        mock_module.MLXReranker = mock_mlx_reranker_class
        mock_importlib.module_from_spec.return_value = mock_module

        with (
            patch("local_reranker.reranker_mlx.mx"),
            patch("local_reranker.reranker_mlx.nn"),
            patch("local_reranker.reranker_mlx.load"),
            patch("local_reranker.reranker_mlx.np"),
            patch("local_reranker.reranker_mlx.safe_open"),
        ):
            reranker = MLXReranker()
            request = RerankRequest(
                query="test query",
                documents=[
                    {"text": "doc1", "metadata": "meta1"},
                    {"text": "doc2", "metadata": "meta2"},
                ],
            )

            results = reranker.rerank(request)

            assert len(results) == 2
            assert results[0].index == 0 and results[0].relevance_score == 0.9
            assert results[1].index == 1 and results[1].relevance_score == 0.7
            mock_mlx_reranker_instance.rerank.assert_called_once_with(
                query="test query",
                documents=["doc1", "doc2"],
                top_n=None,
                return_embeddings=False,
            )

    @patch("local_reranker.reranker_mlx.snapshot_download")
    @patch("local_reranker.reranker_mlx.importlib.util")
    def test_mlx_reranker_implements_protocol(
        self, mock_importlib, mock_snapshot_download
    ):
        """Test that MLX reranker implements the protocol."""
        mock_snapshot_download.return_value = "/mock/model/path"

        mock_spec = Mock()
        mock_spec.loader = Mock()
        mock_importlib.spec_from_file_location.return_value = mock_spec
        mock_importlib.module_from_spec.return_value = Mock()

        mock_mlx_reranker_class = Mock()
        mock_mlx_reranker_class.return_value = Mock()

        mock_module = Mock()
        mock_module.MLXReranker = mock_mlx_reranker_class
        mock_importlib.module_from_spec.return_value = mock_module

        with (
            patch("local_reranker.reranker_mlx.mx"),
            patch("local_reranker.reranker_mlx.nn"),
            patch("local_reranker.reranker_mlx.load"),
            patch("local_reranker.reranker_mlx.np"),
            patch("local_reranker.reranker_mlx.safe_open"),
        ):
            assert isinstance(MLXReranker, type(RerankerProtocol))
            reranker = MLXReranker()
            assert isinstance(reranker, RerankerProtocol)
