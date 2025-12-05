# -*- coding: utf-8 -*-
"""Configuration management for the local reranker service."""

from typing import Dict, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the reranker service."""

    # Reranker configuration
    reranker_type: str = "pytorch"
    model_name: Optional[str] = None  # None = use reranker default

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8010
    log_level: str = "info"
    reload: bool = False

    class Config:
        env_prefix = "RERANKER_"
        case_sensitive = False


# Default models for each reranker type
RERANKER_DEFAULTS: Dict[str, str] = {
    "pytorch": "jinaai/jina-reranker-v2-base-multilingual",
    # Future implementations
    # "onnx": "some/onnx-model",
    # "tensorflow": "some/tf-model",
}


def get_effective_model_name(settings: Settings) -> str:
    """Get the effective model name based on settings and defaults."""
    if settings.model_name:
        return settings.model_name

    if settings.reranker_type in RERANKER_DEFAULTS:
        return RERANKER_DEFAULTS[settings.reranker_type]

    raise ValueError(
        f"No default model configured for reranker type: {settings.reranker_type}"
    )


def get_available_rerankers() -> Dict[str, str]:
    """Get available reranker types and their descriptions."""
    return {
        "pytorch": "PyTorch-based reranker using sentence-transformers",
        # Future implementations
        # "onnx": "ONNX-based reranker for CPU optimization",
        # "tensorflow": "TensorFlow-based reranker",
    }
