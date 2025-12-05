# -*- coding: utf-8 -*-
"""Command line interface for the local reranker service."""

import argparse
import logging

import uvicorn

logger = logging.getLogger(__name__)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8010,
    log_level: str = "info",
    reload: bool = False,
) -> None:
    """Run the Local Reranker API server.

    Args:
        host: Host to bind the server to.
        port: Port to bind the server to.
        log_level: Uvicorn log level.
        reload: Enable auto-reload for development.
    """
    uvicorn.run(
        "local_reranker.api:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Run the Local Reranker API server.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Port to bind the server to (default: 8010).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Uvicorn log level (default: info).",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development."
    )

    args = parser.parse_args()

    logger.info(f"Starting Local Reranker API server on {args.host}:{args.port}")
    run_server(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
