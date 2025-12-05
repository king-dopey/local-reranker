# -*- coding: utf-8 -*-
"""PyTorch implementation of the reranker protocol."""

import logging
from typing import List, Union, Dict, Any, Optional, Tuple
from typing_extensions import override

import torch
from sentence_transformers import CrossEncoder
from .reranker import Reranker as RerankerProtocol
from .models import RerankRequest, RerankResult, RerankDocument

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Default Model ---
# Default model is now managed in config.py
# This constant is kept for backward compatibility
DEFAULT_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"


class Reranker(RerankerProtocol):
    """PyTorch implementation of the reranker protocol using CrossEncoder."""

    @override
    def __init__(
        self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None
    ):
        """
        Initializes the Reranker.

        Args:
            model_name: The name of the CrossEncoder model to load
                        (e.g., 'jinaai/jina-reranker-v2-base-multilingual').
            device: The device to run the model on ('cpu', 'cuda', 'mps').
                    If None, attempts to auto-detect.
        """
        self.model_name = model_name
        self.device = device or self._get_best_device()  # Auto-detect if not specified
        logger.info(
            f"Initializing Reranker with model '{self.model_name}' on device '{self.device}'"
        )

        try:
            # Basic CrossEncoder initialization
            logger.info(
                f"Attempting basic CrossEncoder initialization for '{self.model_name}'..."
            )
            self.model = CrossEncoder(
                model_name_or_path=self.model_name,  # Renamed argument
                device=self.device,
                trust_remote_code=True,  # Add back based on ValueError
            )
            logger.info(
                f"Successfully loaded model '{self.model_name}' to device '{self.device}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to load model '{self.model_name}': {e}", exc_info=True
            )
            # Fallback to CPU if specific device fails? Or just raise?
            # For now, let's raise to make the issue clear.
            raise RuntimeError(
                f"Could not load model '{self.model_name}'. Ensure it's installed or accessible."
            ) from e

    def _get_best_device(self) -> str:
        """Auto-detects the best available device."""
        if torch.cuda.is_available():
            logger.info("CUDA detected. Using GPU.")
            return "cuda"
        # Check for Apple Silicon MPS (requires PyTorch >= 1.12)
        elif torch.backends.mps.is_available():
            if torch.backends.mps.is_built():  # Extra check for older PyTorch versions
                logger.info("MPS detected. Using Apple Silicon GPU.")
                return "mps"
            else:
                logger.warning("MPS available but not built. Falling back to CPU.")
                return "cpu"
        else:
            logger.info("No GPU detected (CUDA or MPS). Using CPU.")
            return "cpu"

    def _prepare_input_pairs(
        self, query: str, documents: List[Union[str, Dict[str, Any]]]
    ) -> Tuple[List[Tuple[str, str]], List[int]]:
        """Prepares query-document pairs for the CrossEncoder model."""
        pairs = []
        original_indices = []
        for i, doc in enumerate(documents):
            doc_text = (
                doc if isinstance(doc, str) else doc.get("text", "")
            )  # Handle string or dict
            if doc_text:  # Avoid empty documents
                pairs.append((query, doc_text))
                original_indices.append(i)
            else:
                logger.warning(f"Skipping empty document at index {i}.")
        return pairs, original_indices

    @override
    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        """
        Reranks documents based on the request.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).
        """
        if not request.documents:
            return []

        sentence_pairs, original_indices = self._prepare_input_pairs(
            request.query, request.documents
        )

        if not sentence_pairs:
            logger.warning("No valid document pairs found after preparation.")
            return []

        logger.debug(
            f"Computing scores for {len(sentence_pairs)} query-document pairs..."
        )
        # The CrossEncoder model's predict method handles batching internally
        # It expects a list of [query, passage] pairs
        scores = self.model.predict(
            sentence_pairs, show_progress_bar=False
        )  # Progress bar can be noisy
        logger.debug("Score computation finished.")

        if len(scores) != len(original_indices):
            logger.error(
                "Mismatch between number of scores and original indices. This shouldn't happen."
            )
            # Handle error case, maybe return empty or raise? For now, return empty
            return []

        # Combine original indices with scores
        indexed_scores = list(zip(original_indices, scores))

        # Sort by score (descending)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_n limit if specified
        if request.top_n is not None:
            indexed_scores = indexed_scores[: request.top_n]

        # Create RerankResult objects
        results: List[RerankResult] = []
        for index, score in indexed_scores:
            doc_content = None
            if request.return_documents:
                original_doc = request.documents[index]
                doc_text = (
                    original_doc
                    if isinstance(original_doc, str)
                    else original_doc.get("text", "")
                )
                doc_content = RerankDocument(text=doc_text)

            results.append(
                RerankResult(
                    document=doc_content, index=index, relevance_score=float(score)
                )
            )

        return results
