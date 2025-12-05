# -*- coding: utf-8 -*-
"""FastAPI application for the local reranker service."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
import torch

from fastapi import FastAPI, HTTPException, Depends, Request
from .models import RerankRequest, RerankResponse
from .reranker import Reranker as RerankerProtocol
from .reranker_pytorch import Reranker as PyTorchReranker
from .reranker_mlx import Reranker as MLXReranker
from .config import Settings, get_effective_model_name

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Configuration ---
settings = Settings()


# --- App Lifespan Management (Load model on startup, cleanup on shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the reranker model's lifecycle."""
    logger.info("Lifespan startup: Loading reranker model...")
    reranker_instance = None
    try:
        # Get configuration from environment variables
        model_name = get_effective_model_name(settings)
        logger.info(f"Loading reranker type: {settings.reranker_type}")
        logger.info(f"Loading model: {model_name}")

        # Initialize reranker based on type
        if settings.reranker_type == "pytorch":
            reranker_instance = PyTorchReranker(model_name=model_name)
        elif settings.reranker_type == "mlx":
            reranker_instance = MLXReranker(model_name=model_name)
        else:
            raise ValueError(f"Unsupported reranker type: {settings.reranker_type}")

        app.state.reranker = reranker_instance  # Store instance in app state
        app.state.settings = settings  # Store settings for reference
        logger.info("Reranker model loaded successfully and stored in app state.")
    except Exception as e:
        logger.error(
            f"Fatal error: Could not load reranker model during startup: {e}",
            exc_info=True,
        )
        app.state.reranker = None  # Ensure it's None if loading failed

    yield  # Application runs here

    # --- Cleanup logic ---
    logger.info("Lifespan shutdown: Releasing resources...")
    current_reranker = getattr(app.state, "reranker", None)
    if current_reranker and hasattr(
        current_reranker.model, "cpu"
    ):  # Basic check if it's a torch model
        try:
            # Ensure model and tensors are moved to CPU before deletion if applicable
            # Note: sentence-transformers CrossEncoder might not need explicit deletion
            # but clearing cache is good practice.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear CUDA cache if applicable
            logger.info("Model resources released.")
        except Exception as e:
            logger.error(f"Error during model resource cleanup: {e}", exc_info=True)
    app.state.reranker = None


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Local Reranker API",
    description="Provides a local implementation of reranker APIs (starting with Jina).",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Dependency Injection for Reranker ---
def get_reranker(request: Request):
    # Retrieve the instance from app.state managed by lifespan
    reranker = getattr(request.app.state, "reranker", None)
    if reranker is None:
        logger.error(
            "Reranker instance is not available via app.state. Model loading might have failed during startup."
        )
        raise HTTPException(
            status_code=503, detail="Service Unavailable: Reranker model not loaded."
        )
    return reranker


# --- API Endpoints ---
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_endpoint(
    request_body: RerankRequest, reranker: RerankerProtocol = Depends(get_reranker)
):
    """Handles reranking requests, compatible with Jina's /v1/rerank API."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.debug(f"[{request_id}] Received rerank request.")
    logger.info(f"[{request_id}] Reranking query: {request_body.query}")
    # logger.info(f"[{request_id}] Reranking request: {request_body}")
    try:
        # Call the reranker's rerank method
        results = reranker.rerank(request_body)

        # Add top score and first few characters of top document to the log message
        top_doc_preview = ""
        top_score = "N/A"
        if results:
            top_score = results[0].relevance_score
            if request_body.return_documents and results[0].document:
                top_doc_preview = results[0].document.text[:50]
        logger.info(
            f"[{request_id}] Reranking done, top score: {top_score}, preview: {top_doc_preview}"
        )
        # logger.info(f"results: {results}")
        response = RerankResponse(id=request_id, results=results)

        end_time = time.time()
        logger.debug(
            f"[{request_id}] Rerank request processed in {end_time - start_time:.4f} seconds."
        )
        return response
    except Exception as e:
        logger.error(
            f"[{request_id}] Error processing rerank request: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Internal Server Error during reranking."
        )


@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}
