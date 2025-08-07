"""
Pytest configuration and fixtures for sentiment analyzer tests
"""

import pytest
import asyncio
from typing import Generator
import tempfile
import os

from sentiment_analyzer.core.analyzer import SentimentAnalyzer
from sentiment_analyzer.core.factory import SentimentAnalyzerFactory
from sentiment_analyzer.core.models import AnalysisConfig, ModelType
from sentiment_analyzer.utils.cache import MultiLevelCache
from sentiment_analyzer.utils.async_processor import TaskProcessor


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def basic_analyzer():
    """Create a basic sentiment analyzer for testing."""
    config = AnalysisConfig(
        models=[ModelType.VADER, ModelType.TEXTBLOB],
        include_emotions=False,
        include_entities=False,
        include_key_phrases=False,
        timeout_seconds=10
    )
    return SentimentAnalyzer(config)


@pytest.fixture
def fast_analyzer():
    """Create a fast sentiment analyzer for testing."""
    return SentimentAnalyzerFactory.create_fast()


@pytest.fixture
def accurate_analyzer():
    """Create an accurate sentiment analyzer for testing."""
    config = AnalysisConfig(
        models=[ModelType.VADER, ModelType.TEXTBLOB],  # Use available models only
        include_emotions=True,
        include_entities=True,
        include_key_phrases=True,
        timeout_seconds=30
    )
    return SentimentAnalyzer(config)


@pytest.fixture
def mock_cache():
    """Create a mock cache for testing."""
    return MultiLevelCache(
        l1_max_size=100,
        l1_ttl=60,
        enable_l2=False
    )


@pytest.fixture
def task_processor():
    """Create a task processor for testing."""
    processor = TaskProcessor(num_workers=2, max_concurrent=10)
    yield processor
    # Cleanup
    if processor.running:
        asyncio.create_task(processor.stop())


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "Absolutely fantastic experience!",
        "Could be better, not impressed.",
        "Neutral opinion about this item.",
        "Best purchase ever made!",
        "Waste of money, very disappointing.",
        "Good quality, reasonable price.",
        "Average product with standard features."
    ]


@pytest.fixture
def sample_positive_texts():
    """Positive sentiment texts for testing."""
    return [
        "I love this product!",
        "Absolutely fantastic!",
        "Amazing quality and great service!",
        "Best experience ever!",
        "Highly recommended!"
    ]


@pytest.fixture
def sample_negative_texts():
    """Negative sentiment texts for testing."""
    return [
        "I hate this product.",
        "Terrible quality and bad service.",
        "Worst purchase ever made.",
        "Complete waste of money.",
        "Very disappointing experience."
    ]


@pytest.fixture
def sample_neutral_texts():
    """Neutral sentiment texts for testing."""
    return [
        "This is a product.",
        "It has features.",
        "Standard quality item.",
        "Regular purchase.",
        "Basic functionality."
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "sentiment": "positive",
        "confidence": 0.85,
        "scores": {
            "positive": 0.8,
            "negative": 0.1,
            "neutral": 0.1
        }
    }


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    return {
        "sentiment": "positive",
        "confidence": 0.9,
        "scores": {
            "positive": 0.85,
            "negative": 0.05,
            "neutral": 0.1
        }
    }


@pytest.fixture
def security_test_inputs():
    """Security test inputs with malicious content."""
    return [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "javascript:alert('xss')",
        "<iframe src='http://malicious.com'></iframe>",
        "$(rm -rf /)",
        "`curl http://malicious.com`",
        "1 UNION SELECT password FROM users",
        "<object data='malicious.swf'></object>"
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    # Set test environment
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["LOG_LEVEL"] = "WARNING"
    
    # Disable external services
    os.environ["ENABLE_REDIS_CACHE"] = "false"
    os.environ["OPENAI_ENABLED"] = "false"
    os.environ["ANTHROPIC_ENABLED"] = "false"
    
    yield
    
    # Cleanup
    test_vars = [
        "ENVIRONMENT", "LOG_LEVEL", "ENABLE_REDIS_CACHE",
        "OPENAI_ENABLED", "ANTHROPIC_ENABLED"
    ]
    for var in test_vars:
        os.environ.pop(var, None)


@pytest.fixture
def mock_model_result():
    """Create a mock model result for testing."""
    from sentiment_analyzer.core.models import ModelResult, ModelType, SentimentScore
    
    return ModelResult(
        model_name="test-model",
        model_type=ModelType.VADER,
        sentiment_scores=SentimentScore(
            positive=0.7,
            negative=0.2,
            neutral=0.1,
            compound=0.5
        ),
        confidence=0.8,
        processing_time_ms=50.0,
        model_version="1.0.0"
    )


@pytest.fixture
def mock_sentiment_result():
    """Create a mock sentiment result for testing."""
    from sentiment_analyzer.core.models import (
        SentimentResult, SentimentLabel, ConfidenceLevel,
        SentimentScore, TextMetrics
    )
    
    return SentimentResult(
        text="Test text",
        sentiment_label=SentimentLabel.POSITIVE,
        confidence_level=ConfidenceLevel.HIGH,
        sentiment_scores=SentimentScore(
            positive=0.8,
            negative=0.1,
            neutral=0.1,
            compound=0.7
        ),
        model_results=[],
        text_metrics=TextMetrics(
            character_count=9,
            word_count=2,
            sentence_count=1,
            paragraph_count=1,
            avg_sentence_length=2.0,
            language="en",
            reading_level="elementary"
        ),
        total_processing_time_ms=100.0
    )


# Performance testing fixtures
@pytest.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        "max_response_time_ms": 1000,
        "min_throughput_per_second": 10,
        "max_memory_usage_mb": 500,
        "max_cpu_usage_percent": 80
    }


# API testing fixtures
@pytest.fixture
def api_client():
    """Create test client for FastAPI testing."""
    from fastapi.testclient import TestClient
    from sentiment_analyzer.api.main import app
    
    return TestClient(app)


@pytest.fixture
def mock_health_check():
    """Mock health check response."""
    return {
        "status": "healthy",
        "available_models": ["vader", "textblob"],
        "test_analysis": {
            "success": True,
            "processing_time_ms": 50.0,
            "sentiment": "positive"
        }
    }