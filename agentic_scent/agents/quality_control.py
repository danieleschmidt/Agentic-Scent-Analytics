"""
Quality control agents for manufacturing monitoring.
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .base import BaseAgent, AnalysisResult, AgentConfig, AgentCapability
from ..sensors.base import SensorReading


@dataclass
class QualityAssessment:
    """Quality assessment result."""
    batch_id: str
    overall_quality: float  # 0-1 score
    passed_checks: List[str]
    failed_checks: List[str]
    confidence: float
    recommendation: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL


class QualityControlAgent(BaseAgent):
    """
    Specialized agent for quality control monitoring and assessment.
    """
    
    def __init__(self, llm_model: str = "gpt-4", knowledge_base: str = None, 
                 alert_threshold: float = 0.95, agent_id: str = "qc_agent"):
        
        config = AgentConfig(
            agent_id=agent_id,
            capabilities=[
                AgentCapability.ANOMALY_DETECTION,
                AgentCapability.ROOT_CAUSE_ANALYSIS,
                AgentCapability.COMPLIANCE_MONITORING
            ],
            llm_model=llm_model,
            knowledge_base=knowledge_base,
            alert_threshold=alert_threshold
        )
        
        super().__init__(config)
        
        # Quality control specific parameters
        self.quality_standards = self._load_quality_standards()
        self.contamination_patterns = self._load_contamination_patterns()
        
    def _load_quality_standards(self) -> Dict[str, Any]:
        """Load quality standards from knowledge base."""
        # Mock implementation - would load from actual knowledge base
        return {
            "pharmaceutical": {
                "volatile_compounds_max": 100.0,
                "moisture_content_max": 0.05,
                "particle_size_tolerance": 0.1,
                "uniformity_threshold": 0.95
            },
            "food": {
                "freshness_indicators": ["aldehydes", "ketones", "esters"],
                "spoilage_markers": ["amines", "sulfur_compounds"],
                "acceptable_deviation": 0.15
            }
        }
    
    def _load_contamination_patterns(self) -> Dict[str, List[float]]:
        """Load known contamination patterns."""
        # Mock contamination signatures
        return {
            "bacterial": [0.2, 0.8, 0.3, 0.1, 0.9, 0.4],
            "chemical": [0.9, 0.1, 0.8, 0.2, 0.1, 0.7],
            "cross_contamination": [0.4, 0.6, 0.5, 0.4, 0.6, 0.5]
        }
    
    async def analyze(self, sensor_reading: SensorReading) -> Optional[AnalysisResult]:
        """
        Analyze sensor reading for quality control issues.
        """
        if not self.is_active:
            return None
        
        # Multi-stage quality analysis
        contamination_analysis = await self._analyze_contamination(sensor_reading)
        process_analysis = await self._analyze_process_parameters(sensor_reading)
        trend_analysis = await self._analyze_trends(sensor_reading)
        
        # Combine analyses
        overall_confidence = np.mean([
            contamination_analysis["confidence"],
            process_analysis["confidence"], 
            trend_analysis["confidence"]
        ])
        
        anomaly_detected = (
            contamination_analysis["anomaly"] or 
            process_analysis["anomaly"] or
            trend_analysis["anomaly"]
        )
        
        # Generate root cause analysis if anomaly detected
        root_cause = None
        recommended_action = "Continue monitoring"
        
        if anomaly_detected:
            root_cause = await self._perform_root_cause_analysis(
                sensor_reading, contamination_analysis, process_analysis, trend_analysis
            )
            recommended_action = await self._generate_corrective_action(root_cause)
        
        result = AnalysisResult(
            agent_id=self.agent_id,
            confidence=overall_confidence,
            anomaly_detected=anomaly_detected,
            contamination_analysis=contamination_analysis,
            process_analysis=process_analysis,
            trend_analysis=trend_analysis,
            root_cause=root_cause,
            recommended_action=recommended_action,
            sensor_id=sensor_reading.sensor_id
        )
        
        self._record_analysis(result)
        return result
    
    async def _analyze_contamination(self, reading: SensorReading) -> Dict[str, Any]:
        """Analyze for contamination patterns."""
        # Compare against known contamination signatures
        max_similarity = 0.0
        best_match = None
        
        reading_pattern = np.array(reading.values[:6])  # Use first 6 channels
        reading_pattern = reading_pattern / np.linalg.norm(reading_pattern)
        
        for contaminant, pattern in self.contamination_patterns.items():
            pattern_norm = np.array(pattern) / np.linalg.norm(pattern)
            similarity = np.dot(reading_pattern, pattern_norm)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = contaminant
        
        contamination_detected = max_similarity > 0.7
        confidence = max_similarity if contamination_detected else 1.0 - max_similarity
        
        return {
            "anomaly": contamination_detected,
            "confidence": confidence,
            "suspected_contaminant": best_match if contamination_detected else None,
            "similarity_score": max_similarity
        }
    
    async def _analyze_process_parameters(self, reading: SensorReading) -> Dict[str, Any]:
        """Analyze process parameter deviations."""
        # Check against acceptable ranges
        mean_value = np.mean(reading.values)
        std_value = np.std(reading.values)
        
        # Mock process limits
        expected_mean = 500.0
        expected_std_max = 50.0
        
        mean_deviation = abs(mean_value - expected_mean) / expected_mean
        std_excessive = std_value > expected_std_max
        
        process_anomaly = mean_deviation > 0.2 or std_excessive
        confidence = 1.0 - min(1.0, mean_deviation)
        
        return {
            "anomaly": process_anomaly,
            "confidence": confidence,
            "mean_deviation": mean_deviation,
            "std_excessive": std_excessive,
            "mean_value": mean_value,
            "std_value": std_value
        }
    
    async def _analyze_trends(self, reading: SensorReading) -> Dict[str, Any]:
        """Analyze temporal trends."""
        # Simple trend analysis - would be more sophisticated in practice
        recent_analyses = self.get_analysis_history(limit=10)
        
        if len(recent_analyses) < 3:
            return {"anomaly": False, "confidence": 0.5, "trend": "insufficient_data"}
        
        # Check for increasing anomaly scores
        recent_scores = [a.confidence for a in recent_analyses[-3:]]
        trend_increasing = all(recent_scores[i] <= recent_scores[i+1] for i in range(len(recent_scores)-1))
        
        trend_anomaly = trend_increasing and recent_scores[-1] > 0.8
        confidence = 0.8 if len(recent_analyses) >= 5 else 0.5
        
        return {
            "anomaly": trend_anomaly,
            "confidence": confidence,
            "trend": "increasing" if trend_increasing else "stable",
            "recent_scores": recent_scores
        }
    
    async def _perform_root_cause_analysis(self, reading: SensorReading, 
                                         contamination: Dict, process: Dict, 
                                         trend: Dict) -> str:
        """Perform root cause analysis using LLM."""
        prompt = f"""
        Analyze the following quality control data and identify the most likely root cause:
        
        Sensor Reading Summary:
        - Mean value: {np.mean(reading.values):.2f}
        - Standard deviation: {np.std(reading.values):.2f}
        - Number of channels: {len(reading.values)}
        
        Contamination Analysis:
        - Contamination detected: {contamination['anomaly']}
        - Suspected contaminant: {contamination.get('suspected_contaminant', 'None')}
        - Similarity score: {contamination['similarity_score']:.3f}
        
        Process Analysis:
        - Process anomaly: {process['anomaly']}
        - Mean deviation: {process['mean_deviation']:.3f}
        - Excessive variation: {process['std_excessive']}
        
        Trend Analysis:
        - Trend anomaly: {trend['anomaly']}
        - Trend direction: {trend['trend']}
        
        Based on this data, what is the most likely root cause of the quality issue?
        Consider: equipment malfunction, raw material issues, process drift, contamination, or operator error.
        """
        
        return await self._call_llm(prompt)
    
    async def _generate_corrective_action(self, root_cause: str) -> str:
        """Generate corrective action recommendations."""
        prompt = f"""
        Given the root cause analysis: "{root_cause}"
        
        Recommend specific corrective actions that should be taken immediately:
        1. Immediate actions to stop further quality issues
        2. Investigation steps to confirm the root cause
        3. Long-term preventive measures
        
        Prioritize actions based on risk level and implementation feasibility.
        """
        
        return await self._call_llm(prompt)
    
    async def evaluate_batch(self, batch_id: str, sensor_data: List[SensorReading] = None) -> QualityAssessment:
        """
        Comprehensive batch quality evaluation.
        """
        if not sensor_data:
            # Mock sensor data for batch
            sensor_data = []
        
        # Analyze all readings for the batch
        analyses = []
        for reading in sensor_data:
            analysis = await self.analyze(reading)
            if analysis:
                analyses.append(analysis)
        
        # Aggregate results
        if not analyses:
            return QualityAssessment(
                batch_id=batch_id,
                overall_quality=0.5,
                passed_checks=[],
                failed_checks=["insufficient_data"],
                confidence=0.0,
                recommendation="Collect more data",
                risk_level="MEDIUM"
            )
        
        anomaly_rate = sum(1 for a in analyses if a.anomaly_detected) / len(analyses)
        avg_confidence = np.mean([a.confidence for a in analyses])
        
        overall_quality = 1.0 - anomaly_rate
        
        passed_checks = []
        failed_checks = []
        
        if anomaly_rate < 0.1:
            passed_checks.append("contamination_check")
        else:
            failed_checks.append("contamination_check")
            
        if avg_confidence > 0.8:
            passed_checks.append("confidence_check")
        else:
            failed_checks.append("confidence_check")
        
        # Determine risk level
        if anomaly_rate > 0.5:
            risk_level = "CRITICAL"
        elif anomaly_rate > 0.3:
            risk_level = "HIGH"
        elif anomaly_rate > 0.1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Generate recommendation
        if overall_quality > 0.9:
            recommendation = "Batch approved for release"
        elif overall_quality > 0.7:
            recommendation = "Batch requires additional testing"
        else:
            recommendation = "Batch should be rejected"
        
        return QualityAssessment(
            batch_id=batch_id,
            overall_quality=overall_quality,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            confidence=avg_confidence,
            recommendation=recommendation,
            risk_level=risk_level
        )