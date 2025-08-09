"""
Base agent framework for intelligent scent analysis.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..sensors.base import SensorReading
from ..llm.client import create_llm_client, LLMClient


class AnalysisResult:
    """Result of agent analysis."""
    
    def __init__(self, agent_id: str, confidence: float = 0.0, 
                 anomaly_detected: bool = False, **kwargs):
        self.agent_id = agent_id
        self.confidence = confidence
        self.anomaly_detected = anomaly_detected
        self.timestamp = datetime.now()
        
        # Dynamic attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class AgentCapability(Enum):
    """Capabilities that agents can have."""
    ANOMALY_DETECTION = "anomaly_detection"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    PROCESS_OPTIMIZATION = "process_optimization"
    COMPLIANCE_MONITORING = "compliance_monitoring"


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    capabilities: List[AgentCapability]
    confidence_threshold: float = 0.8
    llm_model: str = "gpt-4"
    knowledge_base: Optional[str] = None
    alert_threshold: float = 0.95


class BaseAgent(ABC):
    """
    Base class for all intelligent agents in the system.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self.is_active = False
        self._analysis_history: List[AnalysisResult] = []
        
        # Initialize LLM client
        self.llm_client = create_llm_client(
            model=config.llm_model,
            provider=None  # Auto-detect from model name
        )
    
    @abstractmethod
    async def analyze(self, sensor_reading: SensorReading) -> Optional[AnalysisResult]:
        """
        Analyze sensor reading and return results.
        
        Args:
            sensor_reading: Raw sensor data to analyze
            
        Returns:
            AnalysisResult or None if no significant findings
        """
        pass
    
    async def start(self):
        """Start the agent."""
        self.is_active = True
        self.logger.info(f"Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the agent."""
        self.is_active = False
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        return self.config.capabilities
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability."""
        return capability in self.config.capabilities
    
    def get_analysis_history(self, limit: int = 10) -> List[AnalysisResult]:
        """Get recent analysis history."""
        return self._analysis_history[-limit:]
    
    def _record_analysis(self, result: AnalysisResult):
        """Record analysis result in history."""
        self._analysis_history.append(result)
        # Keep only last 1000 results to prevent memory issues
        if len(self._analysis_history) > 1000:
            self._analysis_history = self._analysis_history[-1000:]
    
    async def _call_llm(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Call LLM for analysis using real LLM integration.
        """
        try:
            if context:
                # Use specialized sensor data analysis
                response = await self.llm_client.analyze_sensor_data(context, prompt)
            else:
                # Use general generation
                response = await self.llm_client.generate(prompt)
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            # Fallback to simple heuristic response
            if "anomaly" in prompt.lower():
                return "Analysis indicates potential quality deviation based on sensor pattern analysis."
            return "Analysis completed with normal parameters."
    
    async def explain_decision(self, analysis_result: AnalysisResult) -> str:
        """
        Generate explanation for an analysis decision.
        """
        prompt = f"""
        Explain the reasoning behind this analysis result:
        - Agent: {analysis_result.agent_id}
        - Confidence: {analysis_result.confidence}
        - Anomaly Detected: {analysis_result.anomaly_detected}
        - Timestamp: {analysis_result.timestamp}
        
        Provide a clear explanation for regulatory compliance.
        """
        
        return await self._call_llm(prompt)


class MockLLMAgent(BaseAgent):
    """
    Mock implementation of an LLM-powered agent for testing.
    """
    
    async def analyze(self, sensor_reading: SensorReading) -> Optional[AnalysisResult]:
        """Mock analysis implementation."""
        if not self.is_active:
            return None
        
        # Simple mock logic
        anomaly_score = sum(sensor_reading.values) / len(sensor_reading.values)
        confidence = min(0.95, max(0.1, anomaly_score / 1000.0))
        anomaly_detected = confidence > self.config.confidence_threshold
        
        result = AnalysisResult(
            agent_id=self.agent_id,
            confidence=confidence,
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            sensor_id=sensor_reading.sensor_id
        )
        
        self._record_analysis(result)
        
        if anomaly_detected:
            self.logger.warning(f"Anomaly detected with confidence {confidence:.2f}")
        
        return result