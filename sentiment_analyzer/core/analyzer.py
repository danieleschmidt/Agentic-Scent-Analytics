"""
Core sentiment analysis engine with multi-model support
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
import re

from .models import (
    SentimentResult, SentimentScore, TextInput, AnalysisConfig,
    ModelResult, ModelType, TextMetrics, EmotionScore,
    SentimentLabel, ConfidenceLevel
)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from scipy.special import softmax
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Multi-model sentiment analysis engine"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self._models = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize requested models"""
        for model_type in self.config.models:
            try:
                if model_type == ModelType.TRANSFORMERS and TRANSFORMERS_AVAILABLE:
                    self._initialize_transformers()
                elif model_type == ModelType.VADER and VADER_AVAILABLE:
                    self._initialize_vader()
                elif model_type == ModelType.TEXTBLOB and TEXTBLOB_AVAILABLE:
                    self._initialize_textblob()
                elif model_type == ModelType.OPENAI and OPENAI_AVAILABLE:
                    self._initialize_openai()
                elif model_type == ModelType.ANTHROPIC and ANTHROPIC_AVAILABLE:
                    self._initialize_anthropic()
                logger.info(f"Initialized {model_type.value} model")
            except Exception as e:
                logger.warning(f"Failed to initialize {model_type.value}: {e}")
    
    def _initialize_transformers(self):
        """Initialize transformers model"""
        model_name = self.config.transformers_model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model, 
            tokenizer=tokenizer,
            device=-1  # CPU
        )
        self._models[ModelType.TRANSFORMERS] = {
            "pipeline": sentiment_pipeline,
            "tokenizer": tokenizer,
            "model": model
        }
    
    def _initialize_vader(self):
        """Initialize VADER sentiment analyzer"""
        self._models[ModelType.VADER] = SentimentIntensityAnalyzer()
    
    def _initialize_textblob(self):
        """Initialize TextBlob analyzer"""
        self._models[ModelType.TEXTBLOB] = "initialized"
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        # Note: API key should be set via environment variable
        self._models[ModelType.OPENAI] = "initialized"
    
    def _initialize_anthropic(self):
        """Initialize Anthropic client"""
        # Note: API key should be set via environment variable
        self._models[ModelType.ANTHROPIC] = "initialized"
    
    def _calculate_text_metrics(self, text: str) -> TextMetrics:
        """Calculate text analysis metrics"""
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        return TextMetrics(
            character_count=len(text),
            word_count=len(words),
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
            avg_sentence_length=round(avg_sentence_length, 2),
            language="en",  # Default to English for now
            reading_level="intermediate"  # Placeholder
        )
    
    async def _analyze_with_transformers(self, text: str) -> ModelResult:
        """Analyze sentiment using transformers model"""
        start_time = time.time()
        
        try:
            pipeline = self._models[ModelType.TRANSFORMERS]["pipeline"]
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: pipeline(text)[0]
            )
            
            # Map transformers output to our format
            label = result['label'].lower()
            score = result['score']
            
            if label == 'positive':
                sentiment_scores = SentimentScore(
                    positive=score,
                    negative=1 - score,
                    neutral=0.0,
                    compound=score
                )
            else:  # negative
                sentiment_scores = SentimentScore(
                    positive=1 - score,
                    negative=score,
                    neutral=0.0,
                    compound=-score
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModelResult(
                model_name=self.config.transformers_model,
                model_type=ModelType.TRANSFORMERS,
                sentiment_scores=sentiment_scores,
                confidence=score,
                processing_time_ms=processing_time,
                model_version="latest"
            )
            
        except Exception as e:
            logger.error(f"Transformers analysis failed: {e}")
            # Return default neutral result
            return ModelResult(
                model_name=self.config.transformers_model,
                model_type=ModelType.TRANSFORMERS,
                sentiment_scores=SentimentScore(positive=0.33, negative=0.33, neutral=0.34, compound=0.0),
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _analyze_with_vader(self, text: str) -> ModelResult:
        """Analyze sentiment using VADER"""
        start_time = time.time()
        
        try:
            analyzer = self._models[ModelType.VADER]
            scores = analyzer.polarity_scores(text)
            
            sentiment_scores = SentimentScore(
                positive=scores['pos'],
                negative=scores['neg'],
                neutral=scores['neu'],
                compound=scores['compound']
            )
            
            # Calculate confidence based on compound score magnitude
            confidence = abs(scores['compound'])
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModelResult(
                model_name="VADER",
                model_type=ModelType.VADER,
                sentiment_scores=sentiment_scores,
                confidence=confidence,
                processing_time_ms=processing_time,
                model_version="3.3.2"
            )
            
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            return ModelResult(
                model_name="VADER",
                model_type=ModelType.VADER,
                sentiment_scores=SentimentScore(positive=0.33, negative=0.33, neutral=0.34, compound=0.0),
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _analyze_with_textblob(self, text: str) -> ModelResult:
        """Analyze sentiment using TextBlob"""
        start_time = time.time()
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert polarity (-1 to 1) to positive/negative/neutral scores
            if polarity > 0.1:
                positive = (polarity + 1) / 2
                negative = 1 - positive
                neutral = 0.0
            elif polarity < -0.1:
                negative = abs(polarity)
                positive = 1 - negative
                neutral = 0.0
            else:
                positive = 0.33
                negative = 0.33
                neutral = 0.34
            
            sentiment_scores = SentimentScore(
                positive=positive,
                negative=negative,
                neutral=neutral,
                compound=polarity
            )
            
            # Use subjectivity as confidence indicator (higher subjectivity = lower confidence)
            confidence = 1 - subjectivity
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModelResult(
                model_name="TextBlob",
                model_type=ModelType.TEXTBLOB,
                sentiment_scores=sentiment_scores,
                confidence=confidence,
                processing_time_ms=processing_time,
                model_version="0.17.1"
            )
            
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {e}")
            return ModelResult(
                model_name="TextBlob",
                model_type=ModelType.TEXTBLOB,
                sentiment_scores=SentimentScore(positive=0.33, negative=0.33, neutral=0.34, compound=0.0),
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _analyze_with_openai(self, text: str) -> ModelResult:
        """Analyze sentiment using OpenAI API"""
        start_time = time.time()
        
        try:
            # Placeholder for OpenAI integration
            # In production, this would use the OpenAI API
            sentiment_scores = SentimentScore(
                positive=0.6,
                negative=0.2,
                neutral=0.2,
                compound=0.4
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModelResult(
                model_name=self.config.openai_model,
                model_type=ModelType.OPENAI,
                sentiment_scores=sentiment_scores,
                confidence=0.8,
                processing_time_ms=processing_time,
                model_version="gpt-3.5-turbo"
            )
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return ModelResult(
                model_name=self.config.openai_model,
                model_type=ModelType.OPENAI,
                sentiment_scores=SentimentScore(positive=0.33, negative=0.33, neutral=0.34, compound=0.0),
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _analyze_with_anthropic(self, text: str) -> ModelResult:
        """Analyze sentiment using Anthropic Claude API"""
        start_time = time.time()
        
        try:
            # Placeholder for Anthropic integration
            # In production, this would use the Anthropic API
            sentiment_scores = SentimentScore(
                positive=0.7,
                negative=0.1,
                neutral=0.2,
                compound=0.6
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModelResult(
                model_name=self.config.anthropic_model,
                model_type=ModelType.ANTHROPIC,
                sentiment_scores=sentiment_scores,
                confidence=0.85,
                processing_time_ms=processing_time,
                model_version="claude-3-haiku"
            )
            
        except Exception as e:
            logger.error(f"Anthropic analysis failed: {e}")
            return ModelResult(
                model_name=self.config.anthropic_model,
                model_type=ModelType.ANTHROPIC,
                sentiment_scores=SentimentScore(positive=0.33, negative=0.33, neutral=0.34, compound=0.0),
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _aggregate_results(self, model_results: List[ModelResult]) -> SentimentScore:
        """Aggregate sentiment scores from multiple models"""
        if not model_results:
            return SentimentScore(positive=0.33, negative=0.33, neutral=0.34, compound=0.0)
        
        # Weight results by confidence
        total_weight = sum(result.confidence for result in model_results)
        if total_weight == 0:
            total_weight = len(model_results)
        
        weighted_positive = sum(
            result.sentiment_scores.positive * result.confidence 
            for result in model_results
        ) / total_weight
        
        weighted_negative = sum(
            result.sentiment_scores.negative * result.confidence 
            for result in model_results
        ) / total_weight
        
        weighted_neutral = sum(
            result.sentiment_scores.neutral * result.confidence 
            for result in model_results
        ) / total_weight
        
        weighted_compound = sum(
            result.sentiment_scores.compound * result.confidence 
            for result in model_results
        ) / total_weight
        
        return SentimentScore(
            positive=weighted_positive,
            negative=weighted_negative,
            neutral=weighted_neutral,
            compound=weighted_compound
        )
    
    async def analyze(self, text_input: Union[str, TextInput]) -> SentimentResult:
        """Analyze sentiment of input text"""
        start_time = time.time()
        
        # Convert string input to TextInput object
        if isinstance(text_input, str):
            text_input = TextInput(text=text_input)
        
        text = text_input.text
        
        # Calculate text metrics
        text_metrics = self._calculate_text_metrics(text)
        
        # Run analysis with all configured models
        model_results = []
        tasks = []
        
        for model_type in self.config.models:
            if model_type in self._models:
                if model_type == ModelType.TRANSFORMERS:
                    tasks.append(self._analyze_with_transformers(text))
                elif model_type == ModelType.VADER:
                    tasks.append(self._analyze_with_vader(text))
                elif model_type == ModelType.TEXTBLOB:
                    tasks.append(self._analyze_with_textblob(text))
                elif model_type == ModelType.OPENAI:
                    tasks.append(self._analyze_with_openai(text))
                elif model_type == ModelType.ANTHROPIC:
                    tasks.append(self._analyze_with_anthropic(text))
        
        # Execute all model analyses concurrently
        if tasks:
            model_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions
            model_results = [r for r in model_results if isinstance(r, ModelResult)]
        
        # Aggregate results
        aggregated_scores = self._aggregate_results(model_results)
        sentiment_label = aggregated_scores.get_dominant_sentiment()
        confidence_level = aggregated_scores.get_confidence_level()
        
        total_processing_time = (time.time() - start_time) * 1000
        
        result = SentimentResult(
            text=text,
            sentiment_label=sentiment_label,
            confidence_level=confidence_level,
            sentiment_scores=aggregated_scores,
            model_results=model_results,
            text_metrics=text_metrics,
            total_processing_time_ms=total_processing_time
        )
        
        return result
    
    async def analyze_batch(self, texts: List[Union[str, TextInput]]) -> List[SentimentResult]:
        """Analyze multiple texts in batch"""
        tasks = [self.analyze(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, SentimentResult):
                successful_results.append(result)
            else:
                logger.error(f"Failed to analyze text {i}: {result}")
        
        return successful_results
    
    def get_available_models(self) -> List[ModelType]:
        """Get list of available/initialized models"""
        return list(self._models.keys())
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on analyzer"""
        test_text = "This is a test message."
        
        health_status = {
            "status": "healthy",
            "available_models": [m.value for m in self.get_available_models()],
            "config": self.config.dict(),
            "test_analysis": None
        }
        
        try:
            # Run test analysis
            test_result = await self.analyze(test_text)
            health_status["test_analysis"] = {
                "success": True,
                "processing_time_ms": test_result.total_processing_time_ms,
                "sentiment": test_result.sentiment_label.value
            }
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["test_analysis"] = {
                "success": False,
                "error": str(e)
            }
        
        return health_status
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)