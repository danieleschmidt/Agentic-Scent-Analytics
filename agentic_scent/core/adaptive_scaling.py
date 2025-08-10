"""
Adaptive scaling system with machine learning-based resource optimization.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json


class ScalingDirection(Enum):
    """Scaling direction indicators."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadPattern(Enum):
    """Load pattern types for prediction."""
    STEADY = "steady"
    PERIODIC = "periodic"
    SPIKE = "spike"
    DECLINE = "decline"
    VOLATILE = "volatile"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_usage: float
    memory_usage: float
    active_connections: int
    queue_depth: int
    response_time_p95: float
    throughput: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'active_connections': self.active_connections,
            'queue_depth': self.queue_depth,
            'response_time_p95': self.response_time_p95,
            'throughput': self.throughput,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ScalingDecision:
    """Scaling decision with confidence and reasoning."""
    direction: ScalingDirection
    magnitude: float  # Scale factor (e.g., 1.5 = 50% increase)
    confidence: float  # 0-1 confidence score
    reasoning: List[str]
    predicted_load: Optional[float] = None
    estimated_cost: Optional[float] = None


class MLPredictor:
    """Machine Learning-based load prediction."""
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.metrics_history = deque(maxlen=history_window)
        self.logger = logging.getLogger(__name__)
        
        # Simple moving averages for trend detection
        self.short_window = 10
        self.long_window = 30
        
    def add_metrics(self, metrics: ResourceMetrics):
        """Add new metrics to history."""
        self.metrics_history.append(metrics)
    
    def predict_load(self, horizon_minutes: int = 15) -> Tuple[float, LoadPattern, float]:
        """
        Predict load for the next horizon_minutes.
        
        Returns:
            (predicted_load, pattern_type, confidence)
        """
        if len(self.metrics_history) < self.long_window:
            # Not enough data, return current load
            if self.metrics_history:
                current = self.metrics_history[-1]
                return current.cpu_usage, LoadPattern.STEADY, 0.5
            return 0.5, LoadPattern.STEADY, 0.3
        
        # Extract time series data
        cpu_usage = np.array([m.cpu_usage for m in self.metrics_history])
        timestamps = np.array([m.timestamp for m in self.metrics_history])
        
        # Calculate moving averages
        short_ma = np.mean(cpu_usage[-self.short_window:])
        long_ma = np.mean(cpu_usage[-self.long_window:])
        
        # Trend analysis
        trend = short_ma - long_ma
        volatility = np.std(cpu_usage[-self.short_window:])
        
        # Pattern recognition
        pattern = self._identify_pattern(cpu_usage, trend, volatility)
        
        # Simple linear extrapolation with pattern adjustment
        recent_slope = self._calculate_slope(cpu_usage[-10:])
        base_prediction = cpu_usage[-1] + (recent_slope * horizon_minutes)
        
        # Pattern-based adjustment
        pattern_multiplier = self._get_pattern_multiplier(pattern, horizon_minutes)
        predicted_load = base_prediction * pattern_multiplier
        
        # Clamp prediction to reasonable bounds
        predicted_load = max(0.0, min(1.0, predicted_load))
        
        # Calculate confidence based on trend consistency and data quality
        confidence = self._calculate_prediction_confidence(cpu_usage, trend, volatility)
        
        return predicted_load, pattern, confidence
    
    def _identify_pattern(self, cpu_data: np.ndarray, trend: float, volatility: float) -> LoadPattern:
        """Identify the load pattern type."""
        if volatility > 0.3:
            return LoadPattern.VOLATILE
        elif abs(trend) < 0.05:
            return LoadPattern.STEADY
        elif trend > 0.1:
            return LoadPattern.SPIKE
        elif trend < -0.1:
            return LoadPattern.DECLINE
        else:
            # Check for periodicity
            if self._detect_periodicity(cpu_data):
                return LoadPattern.PERIODIC
            return LoadPattern.STEADY
    
    def _detect_periodicity(self, data: np.ndarray) -> bool:
        """Simple periodicity detection using autocorrelation."""
        if len(data) < 20:
            return False
        
        # Calculate autocorrelation at different lags
        autocorrs = []
        for lag in range(1, min(len(data) // 2, 20)):
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrs.append(abs(corr))
        
        # Check if any lag shows strong correlation
        return any(corr > 0.7 for corr in autocorrs)
    
    def _calculate_slope(self, data: np.ndarray) -> float:
        """Calculate slope of recent data."""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return slope
    
    def _get_pattern_multiplier(self, pattern: LoadPattern, horizon: int) -> float:
        """Get multiplier based on pattern type."""
        multipliers = {
            LoadPattern.STEADY: 1.0,
            LoadPattern.PERIODIC: 1.0 + 0.1 * np.sin(2 * np.pi * horizon / 60),
            LoadPattern.SPIKE: 1.2 if horizon < 30 else 0.9,
            LoadPattern.DECLINE: 0.8,
            LoadPattern.VOLATILE: 1.0 + 0.2 * (np.random.random() - 0.5)
        }
        return multipliers.get(pattern, 1.0)
    
    def _calculate_prediction_confidence(self, data: np.ndarray, 
                                       trend: float, volatility: float) -> float:
        """Calculate confidence in prediction."""
        base_confidence = 0.7
        
        # Reduce confidence for high volatility
        volatility_penalty = min(0.4, volatility)
        confidence = base_confidence - volatility_penalty
        
        # Increase confidence for consistent trends
        if len(data) >= self.long_window:
            trend_consistency = 1.0 - np.std(np.diff(data[-10:])) / (np.mean(data[-10:]) + 0.001)
            confidence += 0.2 * min(1.0, trend_consistency)
        
        return max(0.1, min(1.0, confidence))


class AdaptiveScaler:
    """
    Intelligent auto-scaling system with ML-based predictions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Scaling parameters
        self.min_instances = self.config.get('min_instances', 1)
        self.max_instances = self.config.get('max_instances', 10)
        self.target_cpu_utilization = self.config.get('target_cpu_utilization', 0.7)
        self.scale_up_threshold = self.config.get('scale_up_threshold', 0.8)
        self.scale_down_threshold = self.config.get('scale_down_threshold', 0.3)
        
        # Scaling behavior
        self.scale_up_cooldown = self.config.get('scale_up_cooldown_seconds', 300)
        self.scale_down_cooldown = self.config.get('scale_down_cooldown_seconds', 600)
        self.max_scale_step = self.config.get('max_scale_step', 2.0)
        
        # ML predictor
        self.predictor = MLPredictor(self.config.get('history_window', 100))
        
        # State
        self.current_instances = self.min_instances
        self.last_scale_time = {}
        self.scaling_history = deque(maxlen=100)
        
    async def evaluate_scaling(self, metrics: ResourceMetrics) -> Optional[ScalingDecision]:
        """
        Evaluate whether scaling is needed based on current metrics.
        
        Args:
            metrics: Current resource utilization metrics
            
        Returns:
            ScalingDecision if scaling is recommended, None otherwise
        """
        # Add metrics to predictor
        self.predictor.add_metrics(metrics)
        
        # Get load prediction
        predicted_load, pattern, confidence = self.predictor.predict_load()
        
        # Current state analysis
        current_cpu = metrics.cpu_usage
        current_memory = metrics.memory_usage
        queue_depth = metrics.queue_depth
        response_time = metrics.response_time_p95
        
        reasoning = []
        
        # Determine scaling direction
        direction = ScalingDirection.STABLE
        magnitude = 1.0
        
        # Scale up conditions
        should_scale_up = (
            current_cpu > self.scale_up_threshold or
            predicted_load > self.scale_up_threshold or
            queue_depth > 100 or
            response_time > 5000  # 5 seconds
        )
        
        # Scale down conditions  
        should_scale_down = (
            current_cpu < self.scale_down_threshold and
            predicted_load < self.scale_down_threshold and
            queue_depth < 10 and
            response_time < 1000  # 1 second
        )
        
        # Check cooldown periods
        current_time = time.time()
        last_scale_up = self.last_scale_time.get('up', 0)
        last_scale_down = self.last_scale_time.get('down', 0)
        
        if should_scale_up and (current_time - last_scale_up) > self.scale_up_cooldown:
            if self.current_instances < self.max_instances:
                direction = ScalingDirection.UP
                
                # Calculate magnitude based on urgency
                urgency = max(current_cpu - self.target_cpu_utilization,
                            predicted_load - self.target_cpu_utilization,
                            queue_depth / 100.0)
                magnitude = min(self.max_scale_step, 1.0 + urgency)
                
                reasoning.extend([
                    f"CPU usage: {current_cpu:.1%} > {self.scale_up_threshold:.1%}",
                    f"Predicted load: {predicted_load:.1%}",
                    f"Queue depth: {queue_depth}",
                    f"Response time: {response_time:.0f}ms"
                ])
                
        elif should_scale_down and (current_time - last_scale_down) > self.scale_down_cooldown:
            if self.current_instances > self.min_instances:
                direction = ScalingDirection.DOWN
                
                # Conservative scale down
                underutilization = self.target_cpu_utilization - max(current_cpu, predicted_load)
                magnitude = max(0.5, 1.0 - underutilization)
                
                reasoning.extend([
                    f"CPU usage: {current_cpu:.1%} < {self.scale_down_threshold:.1%}",
                    f"Predicted load: {predicted_load:.1%}",
                    f"System underutilized"
                ])
        
        if direction == ScalingDirection.STABLE:
            return None
        
        # Calculate decision confidence
        decision_confidence = self._calculate_decision_confidence(
            metrics, predicted_load, confidence, direction
        )
        
        # Estimate cost impact
        cost_multiplier = magnitude if direction == ScalingDirection.UP else 1.0 / magnitude
        estimated_cost = self.current_instances * cost_multiplier * 0.10  # $0.10 per instance-hour
        
        decision = ScalingDecision(
            direction=direction,
            magnitude=magnitude,
            confidence=decision_confidence,
            reasoning=reasoning,
            predicted_load=predicted_load,
            estimated_cost=estimated_cost
        )
        
        return decision
    
    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """
        Execute scaling decision.
        
        Args:
            decision: Scaling decision to execute
            
        Returns:
            True if scaling was executed, False otherwise
        """
        if decision.confidence < 0.6:
            self.logger.info(f"Skipping scaling due to low confidence: {decision.confidence:.2f}")
            return False
        
        old_instances = self.current_instances
        
        if decision.direction == ScalingDirection.UP:
            new_instances = min(self.max_instances, 
                              int(self.current_instances * decision.magnitude))
            self.last_scale_time['up'] = time.time()
            
        else:  # Scale down
            new_instances = max(self.min_instances,
                              int(self.current_instances / decision.magnitude))
            self.last_scale_time['down'] = time.time()
        
        if new_instances != old_instances:
            self.current_instances = new_instances
            
            # Record scaling event
            scaling_event = {
                'timestamp': datetime.now().isoformat(),
                'direction': decision.direction.value,
                'old_instances': old_instances,
                'new_instances': new_instances,
                'magnitude': decision.magnitude,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'predicted_load': decision.predicted_load
            }
            
            self.scaling_history.append(scaling_event)
            
            self.logger.info(
                f"Scaled {decision.direction.value}: {old_instances} -> {new_instances} instances "
                f"(confidence: {decision.confidence:.2f}, magnitude: {decision.magnitude:.2f})"
            )
            
            return True
        
        return False
    
    def _calculate_decision_confidence(self, metrics: ResourceMetrics,
                                     predicted_load: float, prediction_confidence: float,
                                     direction: ScalingDirection) -> float:
        """Calculate confidence in scaling decision."""
        base_confidence = 0.8
        
        # Factor in prediction confidence
        confidence = base_confidence * prediction_confidence
        
        # Increase confidence for extreme conditions
        if direction == ScalingDirection.UP:
            if metrics.cpu_usage > 0.9 or metrics.queue_depth > 200:
                confidence += 0.15
        else:
            if metrics.cpu_usage < 0.2 and metrics.queue_depth < 5:
                confidence += 0.1
        
        # Reduce confidence for rapid changes
        if len(self.scaling_history) > 0:
            last_event = self.scaling_history[-1]
            last_time = datetime.fromisoformat(last_event['timestamp'])
            time_since_last = (datetime.now() - last_time).total_seconds()
            
            if time_since_last < 600:  # Less than 10 minutes
                confidence *= 0.8
        
        return max(0.1, min(1.0, confidence))
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling metrics."""
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'target_cpu_utilization': self.target_cpu_utilization,
            'scaling_history': list(self.scaling_history),
            'prediction_metrics': {
                'history_length': len(self.predictor.metrics_history),
                'prediction_window': self.predictor.history_window
            }
        }
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """
        Optimize scaling configuration based on historical performance.
        
        Returns:
            Optimized configuration parameters
        """
        if len(self.scaling_history) < 10:
            return self.config
        
        # Analyze scaling effectiveness
        scale_ups = [e for e in self.scaling_history if e['direction'] == 'up']
        scale_downs = [e for e in self.scaling_history if e['direction'] == 'down']
        
        # Calculate average time between scaling events
        timestamps = [datetime.fromisoformat(e['timestamp']) for e in self.scaling_history]
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                      for i in range(1, len(timestamps))]
        avg_time_between = np.mean(time_diffs) if time_diffs else 0
        
        optimized_config = self.config.copy()
        
        # Adjust cooldown periods based on frequency
        if avg_time_between < 300:  # Too frequent scaling
            optimized_config['scale_up_cooldown_seconds'] *= 1.5
            optimized_config['scale_down_cooldown_seconds'] *= 1.2
        elif avg_time_between > 1800:  # Too infrequent scaling
            optimized_config['scale_up_cooldown_seconds'] *= 0.8
            optimized_config['scale_down_cooldown_seconds'] *= 0.9
        
        # Adjust thresholds based on prediction accuracy
        high_confidence_events = [e for e in self.scaling_history if e['confidence'] > 0.8]
        if len(high_confidence_events) / len(self.scaling_history) > 0.8:
            # High accuracy, can be more aggressive
            optimized_config['scale_up_threshold'] *= 0.95
            optimized_config['scale_down_threshold'] *= 1.05
        else:
            # Lower accuracy, be more conservative
            optimized_config['scale_up_threshold'] *= 1.05
            optimized_config['scale_down_threshold'] *= 0.95
        
        return optimized_config


class HorizontalPodAutoscaler:
    """
    Kubernetes-compatible HPA with advanced ML predictions.
    """
    
    def __init__(self, namespace: str = "default", deployment_name: str = "agentic-scent"):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.scaler = AdaptiveScaler()
        self.logger = logging.getLogger(__name__)
        
    async def get_pod_metrics(self) -> ResourceMetrics:
        """Get current pod metrics (mock implementation)."""
        # In production, this would query Kubernetes metrics
        current_time = datetime.now()
        
        # Simulate realistic metrics with some randomness
        base_cpu = 0.5 + 0.3 * np.sin(2 * np.pi * current_time.hour / 24)
        cpu_usage = max(0.1, min(0.9, base_cpu + 0.1 * (np.random.random() - 0.5)))
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=0.6 + 0.2 * (np.random.random() - 0.5),
            active_connections=int(50 + 30 * np.random.random()),
            queue_depth=int(20 * cpu_usage + 10 * (np.random.random() - 0.5)),
            response_time_p95=1000 + 2000 * cpu_usage,
            throughput=100 * (1.0 - cpu_usage) + 20
        )
    
    async def auto_scale_loop(self, interval_seconds: int = 30):
        """
        Main auto-scaling loop.
        
        Args:
            interval_seconds: Evaluation interval
        """
        self.logger.info(f"Starting auto-scaling loop for {self.deployment_name}")
        
        while True:
            try:
                # Get current metrics
                metrics = await self.get_pod_metrics()
                
                # Evaluate scaling need
                decision = await self.scaler.evaluate_scaling(metrics)
                
                if decision:
                    self.logger.info(
                        f"Scaling decision: {decision.direction.value} "
                        f"(magnitude: {decision.magnitude:.2f}, confidence: {decision.confidence:.2f})"
                    )
                    
                    # Execute scaling
                    executed = self.scaler.execute_scaling(decision)
                    
                    if executed:
                        await self._apply_scaling_to_k8s(decision)
                
                else:
                    self.logger.debug("No scaling needed")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _apply_scaling_to_k8s(self, decision: ScalingDecision):
        """Apply scaling decision to Kubernetes (mock implementation)."""
        # In production, this would use kubectl or Kubernetes Python client
        self.logger.info(
            f"Applying scaling to K8s: {self.deployment_name} "
            f"-> {self.scaler.current_instances} replicas"
        )
        
        # Mock kubectl command
        kubectl_cmd = (
            f"kubectl scale deployment {self.deployment_name} "
            f"--replicas={self.scaler.current_instances} "
            f"--namespace={self.namespace}"
        )
        
        self.logger.info(f"Would execute: {kubectl_cmd}")


# Convenience function for easy integration
def create_adaptive_scaler(config: Optional[Dict[str, Any]] = None) -> AdaptiveScaler:
    """Create and configure an adaptive scaler."""
    return AdaptiveScaler(config)


async def run_hpa_demo():
    """Demo function showing HPA in action."""
    hpa = HorizontalPodAutoscaler()
    
    # Run for 5 minutes as demo
    demo_duration = 300  # seconds
    start_time = time.time()
    
    while time.time() - start_time < demo_duration:
        await hpa.auto_scale_loop(30)  # 30-second intervals
        
        # Print scaling metrics every 2 minutes
        if int(time.time() - start_time) % 120 == 0:
            metrics = hpa.scaler.get_scaling_metrics()
            print(f"Current instances: {metrics['current_instances']}")
            print(f"Recent scaling events: {len(metrics['scaling_history'])}")


if __name__ == "__main__":
    asyncio.run(run_hpa_demo())