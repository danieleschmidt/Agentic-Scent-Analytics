"""
Sentiment Analyzer Pro - Advanced Multi-Model Sentiment Analysis Platform

A comprehensive sentiment analysis system with multi-model support, real-time processing,
and enterprise-grade features for production deployment.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__email__ = "info@terragonlabs.com"

from .core.analyzer import SentimentAnalyzer
from .core.models import SentimentResult, TextInput, AnalysisConfig
from .core.factory import SentimentAnalyzerFactory

__all__ = [
    "SentimentAnalyzer",
    "SentimentResult", 
    "TextInput",
    "AnalysisConfig",
    "SentimentAnalyzerFactory"
]