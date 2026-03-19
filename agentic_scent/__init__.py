"""
Agentic Scent Analytics
=======================
Agentic framework for chemical sensor (e-nose) data analysis,
odorant classification, and anomaly detection in industrial environments.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

from .sensor import OdorSensor, SensorReading, ODORANT_PROFILES
from .fusion import SensorFusionAgent
from .classifier import OdorantClassifier
from .anomaly import AnomalyDetectionAgent, AnomalyResult
from .simulator import ScenarioSimulator, ScenarioReport, SampleResult

__all__ = [
    "OdorSensor",
    "SensorReading",
    "ODORANT_PROFILES",
    "SensorFusionAgent",
    "OdorantClassifier",
    "AnomalyDetectionAgent",
    "AnomalyResult",
    "ScenarioSimulator",
    "ScenarioReport",
    "SampleResult",
]
