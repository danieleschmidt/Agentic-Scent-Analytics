"""
Core data models for sentiment analysis
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import uuid


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ModelType(str, Enum):
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VADER = "vader"
    TEXTBLOB = "textblob"
    ENSEMBLE = "ensemble"


class EmotionScore(BaseModel):
    joy: float = Field(ge=0, le=1, description="Joy/happiness score")
    anger: float = Field(ge=0, le=1, description="Anger score")
    fear: float = Field(ge=0, le=1, description="Fear/anxiety score")
    sadness: float = Field(ge=0, le=1, description="Sadness score")
    surprise: float = Field(ge=0, le=1, description="Surprise score")
    disgust: float = Field(ge=0, le=1, description="Disgust score")
    
    @validator('*', pre=True)
    def round_scores(cls, v):
        if isinstance(v, float):
            return round(v, 4)
        return v


class SentimentScore(BaseModel):
    positive: float = Field(ge=0, le=1, description="Positive sentiment probability")
    negative: float = Field(ge=0, le=1, description="Negative sentiment probability") 
    neutral: float = Field(ge=0, le=1, description="Neutral sentiment probability")
    compound: float = Field(ge=-1, le=1, description="Compound sentiment score")
    
    @validator('*', pre=True)
    def round_scores(cls, v):
        if isinstance(v, float):
            return round(v, 4)
        return v
    
    def get_dominant_sentiment(self) -> SentimentLabel:
        scores = {"positive": self.positive, "negative": self.negative, "neutral": self.neutral}
        max_sentiment = max(scores, key=scores.get)
        
        # Check for mixed sentiment
        sorted_scores = sorted(scores.values(), reverse=True)
        if sorted_scores[0] - sorted_scores[1] < 0.1:  # Close scores indicate mixed sentiment
            return SentimentLabel.MIXED
            
        return SentimentLabel(max_sentiment)
    
    def get_confidence_level(self) -> ConfidenceLevel:
        max_score = max(self.positive, self.negative, self.neutral)
        if max_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif max_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif max_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class TextMetrics(BaseModel):
    character_count: int = Field(ge=0, description="Total character count")
    word_count: int = Field(ge=0, description="Total word count") 
    sentence_count: int = Field(ge=0, description="Total sentence count")
    paragraph_count: int = Field(ge=0, description="Total paragraph count")
    avg_sentence_length: float = Field(ge=0, description="Average sentence length in words")
    language: Optional[str] = Field(description="Detected language code")
    reading_level: Optional[str] = Field(description="Estimated reading level")


class ModelResult(BaseModel):
    model_name: str = Field(description="Name of the model used")
    model_type: ModelType = Field(description="Type of model")
    sentiment_scores: SentimentScore = Field(description="Sentiment probability scores")
    emotion_scores: Optional[EmotionScore] = Field(description="Emotion detection scores")
    confidence: float = Field(ge=0, le=1, description="Model confidence score")
    processing_time_ms: float = Field(ge=0, description="Processing time in milliseconds")
    model_version: Optional[str] = Field(description="Model version identifier")


class SentimentResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique analysis ID")
    text: str = Field(description="Input text analyzed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Overall results
    sentiment_label: SentimentLabel = Field(description="Overall sentiment classification")
    confidence_level: ConfidenceLevel = Field(description="Overall confidence level")
    sentiment_scores: SentimentScore = Field(description="Aggregated sentiment scores")
    
    # Individual model results
    model_results: List[ModelResult] = Field(description="Results from individual models")
    
    # Metadata
    text_metrics: TextMetrics = Field(description="Text analysis metrics")
    total_processing_time_ms: float = Field(ge=0, description="Total processing time")
    
    # Optional enhanced features
    emotion_scores: Optional[EmotionScore] = Field(description="Emotion analysis results")
    key_phrases: Optional[List[str]] = Field(description="Extracted key phrases")
    entities: Optional[List[Dict[str, Any]]] = Field(description="Named entities")
    topics: Optional[List[str]] = Field(description="Detected topics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=100000, description="Text to analyze")
    language: Optional[str] = Field(description="Language hint (ISO 639-1 code)")
    metadata: Optional[Dict[str, Any]] = Field(description="Additional metadata")
    
    @validator('text')
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class BatchTextInput(BaseModel):
    texts: List[TextInput] = Field(min_items=1, max_items=1000, description="Batch of texts to analyze")
    batch_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    
    @validator('texts')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 texts")
        return v


class AnalysisConfig(BaseModel):
    models: List[ModelType] = Field(default=[ModelType.TRANSFORMERS], description="Models to use")
    include_emotions: bool = Field(default=False, description="Include emotion analysis")
    include_entities: bool = Field(default=False, description="Include named entity recognition")
    include_key_phrases: bool = Field(default=False, description="Include key phrase extraction")
    include_topics: bool = Field(default=False, description="Include topic detection")
    
    # Model-specific configurations
    transformers_model: str = Field(default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    openai_model: str = Field(default="gpt-3.5-turbo")
    anthropic_model: str = Field(default="claude-3-haiku-20240307")
    
    # Processing options
    batch_size: int = Field(default=32, ge=1, le=128, description="Batch processing size")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=5, description="Maximum retry attempts")
    
    # Output options
    round_scores: bool = Field(default=True, description="Round scores to 4 decimal places")
    include_raw_outputs: bool = Field(default=False, description="Include raw model outputs")


class HealthStatus(BaseModel):
    status: str = Field(description="Overall health status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checks: Dict[str, bool] = Field(description="Individual health checks")
    metrics: Dict[str, float] = Field(description="Performance metrics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalysisStats(BaseModel):
    total_analyses: int = Field(ge=0, description="Total number of analyses performed")
    successful_analyses: int = Field(ge=0, description="Number of successful analyses") 
    failed_analyses: int = Field(ge=0, description="Number of failed analyses")
    average_processing_time_ms: float = Field(ge=0, description="Average processing time")
    model_usage_stats: Dict[str, int] = Field(description="Usage statistics by model")
    
    @property
    def success_rate(self) -> float:
        if self.total_analyses == 0:
            return 0.0
        return round(self.successful_analyses / self.total_analyses, 4)