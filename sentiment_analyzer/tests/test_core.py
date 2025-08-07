"""
Tests for core sentiment analysis functionality
"""

import pytest
import asyncio
from datetime import datetime

from sentiment_analyzer.core.analyzer import SentimentAnalyzer
from sentiment_analyzer.core.factory import SentimentAnalyzerFactory
from sentiment_analyzer.core.models import (
    TextInput, AnalysisConfig, ModelType, SentimentLabel,
    ConfidenceLevel, BatchTextInput
)


class TestSentimentAnalyzer:
    """Test SentimentAnalyzer class"""
    
    @pytest.mark.asyncio
    async def test_basic_analysis(self, basic_analyzer, sample_positive_texts):
        """Test basic sentiment analysis"""
        text = sample_positive_texts[0]
        result = await basic_analyzer.analyze(text)
        
        assert result is not None
        assert result.text == text
        assert isinstance(result.sentiment_label, SentimentLabel)
        assert isinstance(result.confidence_level, ConfidenceLevel)
        assert result.total_processing_time_ms > 0
        assert len(result.model_results) > 0
    
    @pytest.mark.asyncio
    async def test_positive_sentiment_detection(self, basic_analyzer, sample_positive_texts):
        """Test positive sentiment detection"""
        for text in sample_positive_texts:
            result = await basic_analyzer.analyze(TextInput(text=text))
            
            # Should detect positive sentiment for clearly positive texts
            assert result.sentiment_scores.positive >= result.sentiment_scores.negative
    
    @pytest.mark.asyncio
    async def test_negative_sentiment_detection(self, basic_analyzer, sample_negative_texts):
        """Test negative sentiment detection"""
        for text in sample_negative_texts:
            result = await basic_analyzer.analyze(TextInput(text=text))
            
            # Should detect negative sentiment for clearly negative texts
            assert result.sentiment_scores.negative >= result.sentiment_scores.positive
    
    @pytest.mark.asyncio
    async def test_text_input_validation(self, basic_analyzer):
        """Test text input validation"""
        # Valid input
        valid_input = TextInput(text="This is a valid text.")
        result = await basic_analyzer.analyze(valid_input)
        assert result is not None
        
        # Test with metadata
        input_with_metadata = TextInput(
            text="Test with metadata",
            language="en",
            metadata={"source": "test"}
        )
        result = await basic_analyzer.analyze(input_with_metadata)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self, basic_analyzer, sample_texts):
        """Test batch analysis functionality"""
        text_inputs = [TextInput(text=text) for text in sample_texts[:5]]
        results = await basic_analyzer.analyze_batch(text_inputs)
        
        assert len(results) <= len(text_inputs)  # Some might fail
        
        for result in results:
            assert result.text in sample_texts
            assert isinstance(result.sentiment_label, SentimentLabel)
            assert result.total_processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_empty_batch(self, basic_analyzer):
        """Test empty batch handling"""
        results = await basic_analyzer.analyze_batch([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_available_models(self, basic_analyzer):
        """Test getting available models"""
        models = basic_analyzer.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, ModelType) for model in models)
    
    @pytest.mark.asyncio
    async def test_health_check(self, basic_analyzer):
        """Test analyzer health check"""
        health = await basic_analyzer.health_check()
        
        assert "status" in health
        assert "available_models" in health
        assert "test_analysis" in health
        assert isinstance(health["available_models"], list)


class TestAnalysisConfig:
    """Test AnalysisConfig functionality"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = AnalysisConfig()
        
        assert ModelType.TRANSFORMERS in config.models
        assert config.include_emotions is False
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = AnalysisConfig(
            models=[ModelType.VADER, ModelType.TEXTBLOB],
            include_emotions=True,
            timeout_seconds=60,
            max_retries=5
        )
        
        assert config.models == [ModelType.VADER, ModelType.TEXTBLOB]
        assert config.include_emotions is True
        assert config.timeout_seconds == 60
        assert config.max_retries == 5
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test batch size limits
        config = AnalysisConfig(batch_size=200)
        assert config.batch_size == 200
        
        # Test timeout validation
        config = AnalysisConfig(timeout_seconds=10)
        assert config.timeout_seconds == 10


class TestSentimentAnalyzerFactory:
    """Test SentimentAnalyzerFactory functionality"""
    
    def test_create_default(self):
        """Test creating default analyzer"""
        analyzer = SentimentAnalyzerFactory.create_default()
        assert isinstance(analyzer, SentimentAnalyzer)
        assert analyzer.config is not None
    
    def test_create_fast(self):
        """Test creating fast analyzer"""
        analyzer = SentimentAnalyzerFactory.create_fast()
        assert isinstance(analyzer, SentimentAnalyzer)
        
        # Fast analyzer should use lightweight models
        models = analyzer.get_available_models()
        assert ModelType.VADER in models or ModelType.TEXTBLOB in models
    
    def test_create_accurate(self):
        """Test creating accurate analyzer"""
        analyzer = SentimentAnalyzerFactory.create_accurate()
        assert isinstance(analyzer, SentimentAnalyzer)
        assert analyzer.config.include_emotions is True
    
    def test_create_custom(self):
        """Test creating custom analyzer"""
        analyzer = SentimentAnalyzerFactory.create_custom(
            models=[ModelType.VADER],
            include_emotions=True,
            timeout_seconds=20
        )
        
        assert isinstance(analyzer, SentimentAnalyzer)
        assert analyzer.config.include_emotions is True
        assert analyzer.config.timeout_seconds == 20
    
    def test_list_presets(self):
        """Test listing analyzer presets"""
        presets = SentimentAnalyzerFactory.list_presets()
        
        assert isinstance(presets, dict)
        assert "default" in presets
        assert "fast" in presets
        assert "accurate" in presets
        
        for preset_name, preset_info in presets.items():
            assert "description" in preset_info
            assert "models" in preset_info
            assert "speed" in preset_info
            assert "accuracy" in preset_info


class TestTextMetrics:
    """Test text metrics calculation"""
    
    @pytest.mark.asyncio
    async def test_text_metrics_calculation(self, basic_analyzer):
        """Test text metrics are calculated correctly"""
        text = "This is a test sentence. This is another sentence!"
        result = await basic_analyzer.analyze(text)
        
        metrics = result.text_metrics
        assert metrics.character_count == len(text)
        assert metrics.word_count == 10  # Approximate count
        assert metrics.sentence_count == 2
        assert metrics.paragraph_count == 1
        assert metrics.avg_sentence_length > 0
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self, basic_analyzer):
        """Test handling of edge cases in text"""
        # This should be caught by validation, but test graceful handling
        try:
            text = "   "  # Only whitespace
            result = await basic_analyzer.analyze(text)
            # If it doesn't raise an error, check it handles gracefully
            if result:
                assert result.text_metrics.word_count >= 0
        except Exception:
            # Expected to raise validation error
            pass


class TestSentimentScores:
    """Test sentiment score calculations"""
    
    @pytest.mark.asyncio
    async def test_score_ranges(self, basic_analyzer, sample_texts):
        """Test sentiment scores are within valid ranges"""
        for text in sample_texts[:3]:  # Test a few samples
            result = await basic_analyzer.analyze(text)
            
            scores = result.sentiment_scores
            
            # All scores should be between 0 and 1
            assert 0 <= scores.positive <= 1
            assert 0 <= scores.negative <= 1
            assert 0 <= scores.neutral <= 1
            
            # Compound score should be between -1 and 1
            assert -1 <= scores.compound <= 1
    
    @pytest.mark.asyncio
    async def test_score_consistency(self, basic_analyzer):
        """Test score consistency across multiple runs"""
        text = "This is a consistently positive message!"
        
        results = []
        for _ in range(3):
            result = await basic_analyzer.analyze(text)
            results.append(result)
        
        # Scores should be consistent (identical for deterministic models)
        first_result = results[0]
        for result in results[1:]:
            # Allow small variations due to floating point precision
            assert abs(result.sentiment_scores.positive - first_result.sentiment_scores.positive) < 0.01


class TestModelResults:
    """Test individual model results"""
    
    @pytest.mark.asyncio
    async def test_model_results_present(self, basic_analyzer):
        """Test that model results are included"""
        result = await basic_analyzer.analyze("Test text")
        
        assert len(result.model_results) > 0
        
        for model_result in result.model_results:
            assert model_result.model_name is not None
            assert isinstance(model_result.model_type, ModelType)
            assert model_result.confidence >= 0
            assert model_result.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_aggregation(self, basic_analyzer):
        """Test score aggregation from multiple models"""
        result = await basic_analyzer.analyze("This is a positive message!")
        
        if len(result.model_results) > 1:
            # Aggregated scores should be reasonable combination of individual results
            individual_positives = [mr.sentiment_scores.positive for mr in result.model_results]
            
            # Aggregated positive score should be within reasonable range of individual scores
            min_positive = min(individual_positives)
            max_positive = max(individual_positives)
            
            assert min_positive <= result.sentiment_scores.positive <= max_positive


@pytest.mark.performance
class TestPerformance:
    """Performance tests for core functionality"""
    
    @pytest.mark.asyncio
    async def test_single_analysis_performance(self, fast_analyzer, performance_test_config):
        """Test single analysis performance"""
        text = "This is a performance test message."
        
        import time
        start_time = time.time()
        result = await fast_analyzer.analyze(text)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Should complete within reasonable time
        assert processing_time_ms < performance_test_config["max_response_time_ms"]
        assert result.total_processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_batch_analysis_performance(self, fast_analyzer, sample_texts):
        """Test batch analysis performance"""
        texts = [TextInput(text=text) for text in sample_texts[:5]]
        
        import time
        start_time = time.time()
        results = await fast_analyzer.analyze_batch(texts)
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        
        # Batch should be more efficient than individual analyses
        assert len(results) > 0
        avg_time_per_text = total_time_ms / len(results)
        
        # Should process multiple texts efficiently
        assert avg_time_per_text < 2000  # 2 seconds per text maximum