# -*- coding: utf-8 -*-
"""MLX-based reranker implementation for Apple Silicon optimization."""

from typing import List, Optional
import logging
import os
import sys
import importlib.util

from .reranker import Reranker as RerankerProtocol
from .models import RerankRequest, RerankResult, RerankDocument

logger = logging.getLogger(__name__)


class Reranker(RerankerProtocol):
    """MLX implementation of reranker protocol for Apple Silicon optimization."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3-mlx",
        device: Optional[str] = None,
    ):
        """Initialize MLX reranker.

        Args:
            model_name: The name of the MLX model to load.
            device: The device to run the model on. MLX auto-detects Apple Silicon GPU/CPU.
                    This parameter is kept for protocol compatibility but ignored.

        Raises:
            ImportError: If MLX dependencies are not installed.
            RuntimeError: If model loading fails.
        """
        self.model_name = model_name
        self.device = device  # Ignored for MLX, kept for compatibility

        try:
            # Import MLX dependencies
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load
            import numpy as np
            from safetensors import safe_open

            # Download and prepare model files
            model_path = self._prepare_model_files(model_name)

            # Load MLX reranker implementation
            self.model = self._load_mlx_reranker(model_path)
            logger.info(f"Successfully loaded MLX reranker: {model_name}")

        except ImportError as e:
            raise ImportError(
                "MLX dependencies not found. Install with: pip install mlx mlx-lm safetensors"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX model '{model_name}': {e}") from e

    def _prepare_model_files(self, model_name: str) -> str:
        """Prepare model files by downloading from HuggingFace if needed."""
        try:
            from huggingface_hub import snapshot_download

            # Download model to cache directory
            model_path = snapshot_download(
                repo_id=model_name,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "rerank.py"],
            )
            return model_path

        except ImportError:
            raise ImportError(
                "huggingface-hub not found. Install with: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model files: {e}") from e

    def _load_mlx_reranker(self, model_path: str):
        """Load MLX reranker from model path."""
        # Load rerank.py module from model directory
        rerank_file = os.path.join(model_path, "rerank.py")
        if not os.path.exists(rerank_file):
            raise RuntimeError(f"rerank.py not found in model directory: {model_path}")

        spec = importlib.util.spec_from_file_location("mlx_reranker", rerank_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {rerank_file}")

        mlx_reranker_module = importlib.util.module_from_spec(spec)
        sys.modules["mlx_reranker"] = mlx_reranker_module
        spec.loader.exec_module(mlx_reranker_module)

        # Initialize the MLX reranker
        MLXRerankerImpl = mlx_reranker_module.MLXReranker
        return MLXRerankerImpl(
            model_path=model_path,
            projector_path=os.path.join(model_path, "projector.safetensors"),
        )

    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        """Rerank documents using the MLX backend.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).

        Raises:
            ValueError: If the request is invalid.
            RuntimeError: If reranking fails.
        """
        try:
            # Extract documents from request (handle both string and dict formats)
            documents = []
            for doc in request.documents:
                if isinstance(doc, str):
                    documents.append(doc)
                elif isinstance(doc, dict) and "text" in doc:
                    documents.append(doc["text"])
                else:
                    raise ValueError(f"Invalid document format: {doc}")

            # Call MLX reranker
            mlx_results = self.model.rerank(
                query=request.query,
                documents=documents,
                top_n=request.top_n,
                return_embeddings=request.return_documents,
            )

            # Convert MLX results to our RerankResult format
            results = []
            for result in mlx_results:
                # Handle document text based on return_documents flag
                document = None
                if request.return_documents and "document" in result:
                    document = RerankDocument(text=result["document"])

                # Extract relevance score and ensure it's a float
                relevance_score = float(result["relevance_score"])

                # Extract index (original position in input list)
                index = int(result["index"])

                rerank_result = RerankResult(
                    document=document, index=index, relevance_score=relevance_score
                )
                results.append(rerank_result)

            return results

        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"MLX reranking failed: {e}") from e
