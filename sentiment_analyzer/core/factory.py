"""
Factory for creating configured sentiment analyzer instances
"""

import os
import logging
from typing import Optional, Dict, Any, List
from .analyzer import SentimentAnalyzer
from .models import AnalysisConfig, ModelType

logger = logging.getLogger(__name__)


class SentimentAnalyzerFactory:
    """Factory for creating pre-configured sentiment analyzer instances"""
    
    @staticmethod
    def create_default() -> SentimentAnalyzer:
        """Create analyzer with default configuration"""
        config = AnalysisConfig()
        return SentimentAnalyzer(config)
    
    @staticmethod
    def create_fast() -> SentimentAnalyzer:
        """Create analyzer optimized for speed"""
        config = AnalysisConfig(
            models=[ModelType.VADER, ModelType.TEXTBLOB],
            include_emotions=False,
            include_entities=False,
            include_key_phrases=False,
            timeout_seconds=10
        )
        return SentimentAnalyzer(config)
    
    @staticmethod
    def create_accurate() -> SentimentAnalyzer:
        """Create analyzer optimized for accuracy"""
        config = AnalysisConfig(
            models=[ModelType.TRANSFORMERS, ModelType.VADER, ModelType.TEXTBLOB],
            include_emotions=True,
            include_entities=True,
            include_key_phrases=True,
            timeout_seconds=60
        )
        return SentimentAnalyzer(config)
    
    @staticmethod
    def create_enterprise() -> SentimentAnalyzer:
        """Create analyzer with enterprise features"""
        config = AnalysisConfig(
            models=[ModelType.TRANSFORMERS, ModelType.OPENAI, ModelType.ANTHROPIC],
            include_emotions=True,
            include_entities=True,
            include_key_phrases=True,
            include_topics=True,
            timeout_seconds=120,
            max_retries=3
        )
        return SentimentAnalyzer(config)
    
    @staticmethod
    def create_from_env() -> SentimentAnalyzer:
        """Create analyzer from environment variables"""
        models = []
        
        # Determine available models from environment
        if os.getenv('TRANSFORMERS_ENABLED', 'true').lower() == 'true':
            models.append(ModelType.TRANSFORMERS)
        if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_ENABLED', 'false').lower() == 'true':
            models.append(ModelType.OPENAI)
        if os.getenv('ANTHROPIC_API_KEY') and os.getenv('ANTHROPIC_ENABLED', 'false').lower() == 'true':
            models.append(ModelType.ANTHROPIC)
        if os.getenv('VADER_ENABLED', 'true').lower() == 'true':
            models.append(ModelType.VADER)
        if os.getenv('TEXTBLOB_ENABLED', 'true').lower() == 'true':
            models.append(ModelType.TEXTBLOB)
        
        # Fallback to basic models if none configured
        if not models:
            models = [ModelType.VADER, ModelType.TEXTBLOB]
        
        config = AnalysisConfig(
            models=models,
            include_emotions=os.getenv('INCLUDE_EMOTIONS', 'false').lower() == 'true',
            include_entities=os.getenv('INCLUDE_ENTITIES', 'false').lower() == 'true',
            include_key_phrases=os.getenv('INCLUDE_KEY_PHRASES', 'false').lower() == 'true',
            include_topics=os.getenv('INCLUDE_TOPICS', 'false').lower() == 'true',
            transformers_model=os.getenv('TRANSFORMERS_MODEL', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
            openai_model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            anthropic_model=os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307'),
            timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', '30')),
            max_retries=int(os.getenv('MAX_RETRIES', '3'))
        )
        
        return SentimentAnalyzer(config)
    
    @staticmethod
    def create_custom(
        models: List[ModelType],
        include_emotions: bool = False,
        include_entities: bool = False,
        include_key_phrases: bool = False,
        include_topics: bool = False,
        **kwargs
    ) -> SentimentAnalyzer:
        """Create analyzer with custom configuration"""
        config = AnalysisConfig(
            models=models,
            include_emotions=include_emotions,
            include_entities=include_entities,
            include_key_phrases=include_key_phrases,
            include_topics=include_topics,
            **kwargs
        )
        return SentimentAnalyzer(config)
    
    @staticmethod
    def create_from_config(config_dict: Dict[str, Any]) -> SentimentAnalyzer:
        """Create analyzer from configuration dictionary"""
        config = AnalysisConfig(**config_dict)
        return SentimentAnalyzer(config)
    
    @staticmethod
    def list_presets() -> Dict[str, Dict[str, Any]]:
        """List available analyzer presets"""
        return {
            "default": {
                "description": "Balanced configuration with transformers model",
                "models": ["transformers"],
                "features": ["basic_sentiment"],
                "speed": "medium",
                "accuracy": "high"
            },
            "fast": {
                "description": "Optimized for speed with rule-based models",
                "models": ["vader", "textblob"],
                "features": ["basic_sentiment"],
                "speed": "very_high",
                "accuracy": "medium"
            },
            "accurate": {
                "description": "Optimized for accuracy with multiple models",
                "models": ["transformers", "vader", "textblob"],
                "features": ["sentiment", "emotions", "entities", "key_phrases"],
                "speed": "medium",
                "accuracy": "very_high"
            },
            "enterprise": {
                "description": "Full-featured enterprise configuration",
                "models": ["transformers", "openai", "anthropic"],
                "features": ["sentiment", "emotions", "entities", "key_phrases", "topics"],
                "speed": "low",
                "accuracy": "very_high"
            }
        }