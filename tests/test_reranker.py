# -*- coding: utf-8 -*-
"""Tests for reranker protocol compliance and PyTorch implementation."""

import pytest
from unittest.mock import Mock, patch

from local_reranker.reranker import Reranker as RerankerProtocol
from local_reranker.reranker_pytorch import Reranker as PyTorchReranker
from local_reranker.models import RerankRequest


class TestRerankerProtocol:
    """Test the reranker protocol definition and compliance."""

    def test_protocol_is_runtime_checkable(self):
        """Test that the protocol is runtime checkable."""
        assert hasattr(RerankerProtocol, "__instancecheck__")
        assert hasattr(RerankerProtocol, "__subclasscheck__")

    def test_pytorch_reranker_implements_protocol(self):
        """Test that PyTorch reranker implements the protocol."""
        assert isinstance(PyTorchReranker, type(RerankerProtocol))
        # Test that an instance would satisfy the protocol
        with patch("local_reranker.reranker_pytorch.CrossEncoder"):
            reranker = PyTorchReranker()
            assert isinstance(reranker, RerankerProtocol)


class TestPyTorchReranker:
    """Test the PyTorch implementation of the reranker protocol."""

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    @patch("local_reranker.reranker_pytorch.torch")
    def test_initialization_with_default_params(self, mock_torch, mock_cross_encoder):
        """Test reranker initialization with default parameters."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        reranker = PyTorchReranker()

        assert reranker.model_name == "jinaai/jina-reranker-v2-base-multilingual"
        assert reranker.device == "cpu"
        mock_cross_encoder.assert_called_once()

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    @patch("local_reranker.reranker_pytorch.torch")
    def test_initialization_with_custom_params(self, mock_torch, mock_cross_encoder):
        """Test reranker initialization with custom parameters."""
        mock_torch.cuda.is_available.return_value = True

        reranker = PyTorchReranker(model_name="custom-model", device="cuda")

        assert reranker.model_name == "custom-model"
        assert reranker.device == "cuda"
        mock_cross_encoder.assert_called_once_with(
            model_name_or_path="custom-model", device="cuda", trust_remote_code=True
        )

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    @patch("local_reranker.reranker_pytorch.torch")
    def test_device_auto_detection_cuda(self, mock_torch, mock_cross_encoder):
        """Test CUDA device auto-detection."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        reranker = PyTorchReranker()
        assert reranker.device == "cuda"

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    @patch("local_reranker.reranker_pytorch.torch")
    def test_device_auto_detection_mps(self, mock_torch, mock_cross_encoder):
        """Test MPS device auto-detection."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.backends.mps.is_built.return_value = True

        reranker = PyTorchReranker()
        assert reranker.device == "mps"

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    @patch("local_reranker.reranker_pytorch.torch")
    def test_device_auto_detection_cpu(self, mock_torch, mock_cross_encoder):
        """Test CPU fallback when no GPU available."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        reranker = PyTorchReranker()
        assert reranker.device == "cpu"

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    def test_rerank_with_string_documents(self, mock_cross_encoder):
        """Test reranking with string documents."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.8]
        mock_cross_encoder.return_value = mock_model

        reranker = PyTorchReranker()
        request = RerankRequest(query="test query", documents=["doc1", "doc2", "doc3"])

        results = reranker.rerank(request)

        assert len(results) == 3
        assert results[0].index == 0 and results[0].relevance_score == 0.9
        assert results[1].index == 2 and results[1].relevance_score == 0.8
        assert results[2].index == 1 and results[2].relevance_score == 0.7
        mock_model.predict.assert_called_once()

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    def test_rerank_with_dict_documents(self, mock_cross_encoder):
        """Test reranking with dictionary documents."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7]
        mock_cross_encoder.return_value = mock_model

        reranker = PyTorchReranker()
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

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    def test_rerank_with_empty_documents(self, mock_cross_encoder):
        """Test reranking with empty documents list."""
        mock_cross_encoder.return_value = Mock()

        reranker = PyTorchReranker()
        request = RerankRequest(query="test query", documents=[])

        results = reranker.rerank(request)

        assert results == []

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    def test_rerank_with_empty_document_content(self, mock_cross_encoder):
        """Test reranking with documents that have empty content."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8]  # Two valid documents
        mock_cross_encoder.return_value = mock_model

        reranker = PyTorchReranker()
        request = RerankRequest(
            query="test query",
            documents=["valid doc", "", {"text": ""}, "another valid"],
        )

        results = reranker.rerank(request)

        # Should skip empty documents and only process valid ones
        assert len(results) == 2
        assert results[0].index == 0 and results[0].relevance_score == 0.9
        assert results[1].index == 3 and results[1].relevance_score == 0.8

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    def test_rerank_with_top_n(self, mock_cross_encoder):
        """Test reranking with top_n limit."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.8, 0.6]
        mock_cross_encoder.return_value = mock_model

        reranker = PyTorchReranker()
        request = RerankRequest(
            query="test query", documents=["doc1", "doc2", "doc3", "doc4"], top_n=2
        )

        results = reranker.rerank(request)

        # Should return only top 2 results
        assert len(results) == 2
        assert results[0].index == 0 and results[0].relevance_score == 0.9
        assert results[1].index == 2 and results[1].relevance_score == 0.8

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    def test_rerank_with_return_documents(self, mock_cross_encoder):
        """Test reranking with return_documents=True."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7]
        mock_cross_encoder.return_value = mock_model

        reranker = PyTorchReranker()
        request = RerankRequest(
            query="test query", documents=["doc1", "doc2"], return_documents=True
        )

        results = reranker.rerank(request)

        assert len(results) == 2
        assert results[0].document is not None
        assert results[0].document.text == "doc1"
        assert results[1].document is not None
        assert results[1].document.text == "doc2"

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    def test_compute_scores_model_loading_failure(self, mock_cross_encoder):
        """Test handling of model loading failures."""
        mock_cross_encoder.side_effect = Exception("Model loading failed")

        with pytest.raises(RuntimeError, match="Could not load model"):
            PyTorchReranker()


class TestRerankerErrorHandling:
    """Test error handling in reranker implementations."""

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    def test_mismatched_scores_and_indices(self, mock_cross_encoder):
        """Test handling of mismatched scores and indices."""
        mock_model = Mock()
        # Return different number of scores than expected
        mock_model.predict.return_value = [0.9, 0.7]  # Only 2 scores for 3 docs
        mock_cross_encoder.return_value = mock_model

        reranker = PyTorchReranker()
        request = RerankRequest(query="test query", documents=["doc1", "doc2", "doc3"])

        results = reranker.rerank(request)

        # Should return empty list when mismatch occurs
        assert results == []

    @patch("local_reranker.reranker_pytorch.CrossEncoder")
    def test_no_valid_document_pairs(self, mock_cross_encoder):
        """Test handling when no valid document pairs are found."""
        mock_model = Mock()
        mock_model.predict.return_value = []  # No scores for empty documents
        mock_cross_encoder.return_value = mock_model

        reranker = PyTorchReranker()
        request = RerankRequest(
            query="test query",
            documents=["", {"text": ""}, "   "],  # All empty documents
        )

        results = reranker.rerank(request)

        assert results == []
