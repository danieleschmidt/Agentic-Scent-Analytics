"""
Agentic Scent Analytics - LLM-powered analytics platform for smart factory e-nose deployments.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core.factory import ScentAnalyticsFactory
from .agents.quality_control import QualityControlAgent
from .agents.orchestrator import AgentOrchestrator
from .sensors.base import SensorInterface
from .analytics.fingerprinting import ScentFingerprinter
from .predictive.quality import QualityPredictor

__all__ = [
    "ScentAnalyticsFactory",
    "QualityControlAgent", 
    "AgentOrchestrator",
    "SensorInterface",
    "ScentFingerprinter",
    "QualityPredictor",
]