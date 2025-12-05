# -*- coding: utf-8 -*-
"""Integration tests for CLI and FastAPI application."""

import pytest
import subprocess
import time
import httpx
import sys


class TestCLIAppIntegration:
    """Test CLI integration with FastAPI application."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_cli_starts_fastapi_app(self):
        """Test that CLI properly starts the FastAPI application."""
        # Start server with custom port to avoid conflicts
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "local_reranker.cli",
                "--host",
                "127.0.0.1",
                "--port",
                "8012",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Give server time to start
            time.sleep(8)  # Increased time for model loading

            # Check if process is still running
            assert process.poll() is None, "Server failed to start"

            # Test health endpoint
            with httpx.Client(timeout=10.0) as client:
                response = client.get("http://127.0.0.1:8012/health")
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}

        finally:
            # Clean up
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_cli_with_rerank_endpoint(self):
        """Test that CLI can serve actual rerank requests."""
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "local_reranker.cli",
                "--host",
                "127.0.0.1",
                "--port",
                "8013",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Give server time to start and load model
            time.sleep(10)

            # Check if process is still running
            assert process.poll() is None, "Server failed to start"

            # Test actual rerank endpoint
            with httpx.Client(timeout=30.0) as client:
                # First check health
                health_response = client.get("http://127.0.0.1:8013/health")
                assert health_response.status_code == 200

                # Then test rerank endpoint
                rerank_payload = {
                    "model": "jina-reranker-v1-tiny-en",
                    "query": "What is the capital of France?",
                    "documents": [
                        "Paris is the capital of France.",
                        "Berlin is the capital of Germany.",
                        "The Eiffel Tower is in Paris.",
                    ],
                    "top_n": 2,
                    "return_documents": True,
                }

                response = client.post(
                    "http://127.0.0.1:8013/v1/rerank", json=rerank_payload
                )
                assert response.status_code == 200

                data = response.json()
                assert "id" in data
                assert "results" in data
                assert len(data["results"]) == 2

                # Verify response structure
                for result in data["results"]:
                    assert "index" in result
                    assert "relevance_score" in result
                    assert "document" in result
                    assert result["document"] is not None
                    assert "text" in result["document"]

        finally:
            # Clean up
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    @pytest.mark.integration
    def test_cli_argument_passing_to_uvicorn(self):
        """Test that CLI arguments are correctly passed to uvicorn."""
        # This test uses mocking to verify argument passing without starting a real server
        from unittest.mock import patch

        with patch("local_reranker.cli.uvicorn.run") as mock_uvicorn:
            # Import and run main with custom arguments
            from local_reranker.cli import main

            with patch(
                "sys.argv",
                [
                    "local-reranker",
                    "--host",
                    "192.168.1.100",
                    "--port",
                    "9999",
                    "--log-level",
                    "debug",
                    "--reload",
                ],
            ):
                main()

                # Verify uvicorn.run was called with correct arguments
                mock_uvicorn.assert_called_once_with(
                    "local_reranker.api:app",
                    host="192.168.1.100",
                    port=9999,
                    log_level="debug",
                    reload=True,
                )

    @pytest.mark.integration
    def test_cli_api_module_access(self):
        """Test that CLI can access the API module correctly."""
        # Test that that API module can be imported and has the expected app
        try:
            from local_reranker.api import app

            assert hasattr(app, "title")
            assert app.title == "Local Reranker API"

            # Test that the app has the expected routes
            routes = []
            for route in app.routes:
                if hasattr(route, "path"):
                    routes.append(route.path)

            assert "/health" in routes
            assert "/v1/rerank" in routes

        except ImportError as e:
            pytest.fail(f"Failed to import API module: {e}")

    @pytest.mark.integration
    def test_cli_entry_point_functionality(self):
        """Test that the CLI entry point from pyproject.toml works."""
        # Test that we can import the main function
        try:
            from local_reranker.cli import main

            assert callable(main)

            # Test that main function has expected signature
            import inspect

            sig = inspect.signature(main)
            assert len(sig.parameters) == 0  # main takes no parameters

        except ImportError as e:
            pytest.fail(f"Failed to import CLI main function: {e}")
