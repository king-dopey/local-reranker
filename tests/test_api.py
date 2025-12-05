# -*- coding: utf-8 -*-
"""Tests for the FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock  # Import MagicMock

# Assuming your FastAPI app instance is named 'app' in 'src/local_reranker/api.py'
# Adjust the import path if necessary
from local_reranker.api import app, get_reranker  # Import get_reranker for overriding
from local_reranker.reranker import (
    Reranker,
)  # Import Reranker for type hinting if needed


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient for the FastAPI app."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Tests for /v1/rerank using mocking ---


@pytest.fixture
def mock_reranker_dependency():
    """Fixture to create a mock Reranker and override the dependency."""
    mock_reranker = MagicMock(spec=Reranker)

    # Define a default mock rerank behavior
    def default_mock_rerank(request):
        # Create mock results in descending order of score for simplicity
        from local_reranker.models import RerankResult

        scores = sorted(
            [(i, 1.0 - (i * 0.1)) for i, _ in enumerate(request.documents)],
            key=lambda x: x[1],
            reverse=True,
        )
        return [RerankResult(index=i, relevance_score=score) for i, score in scores]

    mock_reranker.rerank.side_effect = default_mock_rerank

    # Define the override function
    def override_get_reranker():
        return mock_reranker

    # Apply the override
    app.dependency_overrides[get_reranker] = override_get_reranker
    yield mock_reranker  # Provide the mock object to the test if needed
    # Clean up the override after the test
    app.dependency_overrides.clear()


def test_rerank_endpoint_basic(client, mock_reranker_dependency):
    """Test basic reranking functionality with mocked reranker."""
    # Define specific mock behavior for this test
    # Input docs: ["Paris...", "Berlin...", "Eiffel..."]
    # Mock scores: Assign scores so expected sorted order is Paris, Eiffel, Berlin
    from local_reranker.models import RerankResult

    mock_reranker_dependency.rerank.side_effect = lambda request: [
        RerankResult(index=0, relevance_score=0.9),  # Paris
        RerankResult(index=2, relevance_score=0.8),  # Eiffel
        RerankResult(index=1, relevance_score=0.1),  # Berlin
    ]

    payload = {
        "model": "mocked-model",
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "The Eiffel Tower is in Paris.",
        ],
        "top_n": 3,
        "return_documents": False,
    }
    response = client.post("/v1/rerank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "results" in data
    assert len(data["results"]) == 3

    # Check specific sorting based on mocked scores
    result_indices = [result["index"] for result in data["results"]]
    result_scores = [result["relevance_score"] for result in data["results"]]
    assert result_indices == [0, 2, 1]  # Expected order: Paris, Eiffel, Berlin
    assert result_scores == [0.9, 0.8, 0.1]

    for result in data["results"]:
        assert "index" in result
        assert "relevance_score" in result
        assert result["document"] is None


def test_rerank_endpoint_top_n(client, mock_reranker_dependency):
    """Test the top_n parameter with mocked reranker."""
    # Mock scores: [0.9, 0.5, 0.8, 0.1]
    from local_reranker.models import RerankResult

    mock_reranker_dependency.rerank.side_effect = (
        lambda request: [
            RerankResult(index=0, relevance_score=0.9),
            RerankResult(index=2, relevance_score=0.8),
            RerankResult(index=1, relevance_score=0.5),
            RerankResult(index=3, relevance_score=0.1),
        ][: request.top_n]
        if request.top_n
        else [
            RerankResult(index=0, relevance_score=0.9),
            RerankResult(index=2, relevance_score=0.8),
            RerankResult(index=1, relevance_score=0.5),
            RerankResult(index=3, relevance_score=0.1),
        ]
    )

    payload = {
        "query": "Fast cars",
        "documents": [
            "A Ferrari is a fast car.",
            "A Ford Focus is a car.",
            "A Lamborghini is also a fast car.",
            "My bicycle is slow.",
        ],
        "top_n": 2,  # Request only top 2
        "return_documents": False,
    }
    response = client.post("/v1/rerank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2  # Should only return 2 results

    # Check specific sorting based on mocked scores
    result_indices = [result["index"] for result in data["results"]]
    result_scores = [result["relevance_score"] for result in data["results"]]
    assert result_indices == [0, 2]  # Expected order: Ferrari, Lamborghini
    assert result_scores == [0.9, 0.8]


def test_rerank_endpoint_return_documents(client, mock_reranker_dependency):
    """Test the return_documents parameter with mocked reranker."""
    # Mock scores: [0.9, 0.8]
    from local_reranker.models import RerankResult, RerankDocument

    mock_reranker_dependency.rerank.side_effect = lambda request: [
        RerankResult(
            index=0,
            relevance_score=0.9,
            document=RerankDocument(text=request.documents[0])
            if request.return_documents
            else None,
        ),
        RerankResult(
            index=1,
            relevance_score=0.8,
            document=RerankDocument(text=request.documents[1])
            if request.return_documents
            else None,
        ),
    ]

    docs = ["Python is a programming language.", "Java is another language."]
    payload = {
        "query": "programming",
        "documents": docs,
        "top_n": 2,
        "return_documents": True,  # Request documents back
    }
    response = client.post("/v1/rerank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    for result in data["results"]:
        assert result["document"] is not None
        assert "text" in result["document"]
        original_doc_index = result["index"]
        assert result["document"]["text"] == docs[original_doc_index]


def test_rerank_endpoint_empty_documents(client, mock_reranker_dependency):
    """Test reranking with an empty document list with mocked reranker."""
    # Mock should return empty list for empty input
    mock_reranker_dependency.rerank.side_effect = lambda request: []

    payload = {
        "query": "Anything",
        "documents": [],
        "top_n": 3,
        "return_documents": False,
    }
    response = client.post("/v1/rerank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 0  # Expect empty results
