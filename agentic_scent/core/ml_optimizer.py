"""
Machine Learning-based system optimizer for adaptive performance tuning.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json


class OptimizationTarget(Enum):
    """Optimization targets for the ML optimizer."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    COST = "cost"
    ENERGY = "energy"
    MEMORY = "memory"
    BALANCED = "balanced"


class OptimizationStrategy(Enum):
    """Optimization strategies."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"
    EXPERIMENTAL = "experimental"


@dataclass
class SystemMetrics:
    """Comprehensive system performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io_ops: float
    network_io_mbps: float
    cache_hit_rate: float
    request_latency_ms: float
    throughput_rps: float
    error_rate: float
    queue_depth: int
    active_connections: int
    temperature_celsius: float = 25.0
    
    def to_vector(self) -> np.ndarray:
        """Convert metrics to feature vector for ML."""
        return np.array([
            self.cpu_usage,
            self.memory_usage,
            self.disk_io_ops,
            self.network_io_mbps,
            self.cache_hit_rate,
            self.request_latency_ms / 1000.0,  # Normalize to seconds
            self.throughput_rps / 100.0,  # Normalize
            self.error_rate,
            self.queue_depth / 100.0,  # Normalize
            self.active_connections / 1000.0,  # Normalize
            self.temperature_celsius / 100.0  # Normalize
        ])


@dataclass
class OptimizationAction:
    """Action that can be taken to optimize the system."""
    name: str
    parameter: str
    current_value: Any
    suggested_value: Any
    confidence: float
    expected_improvement: float
    risk_level: float
    rollback_plan: str
    cost_estimate: float = 0.0


@dataclass
class OptimizationResult:
    """Result of applying an optimization."""
    action: OptimizationAction
    applied_at: datetime
    success: bool
    actual_improvement: Optional[float] = None
    side_effects: List[str] = field(default_factory=list)
    metrics_before: Optional[SystemMetrics] = None
    metrics_after: Optional[SystemMetrics] = None


class MLPerformanceModel:
    """Machine Learning model for system performance prediction."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.action_history = deque(maxlen=history_size)
        self.logger = logging.getLogger(__name__)
        
        # Simple linear model weights (in production, use proper ML library)
        self.feature_weights = np.random.normal(0, 0.1, 11)  # 11 features
        self.performance_baseline = 0.5
        
        # Model performance tracking
        self.prediction_errors = deque(maxlen=100)
        self.model_accuracy = 0.5
    
    def add_observation(self, metrics: SystemMetrics, performance_score: float):
        """Add observation for model training."""
        self.metrics_history.append((metrics, performance_score))
        
        # Simple online learning update
        if len(self.metrics_history) > 10:
            self._update_model()
    
    def predict_performance(self, metrics: SystemMetrics) -> Tuple[float, float]:
        """
        Predict system performance score.
        
        Returns:
            (predicted_score, confidence)
        """
        feature_vector = metrics.to_vector()
        
        # Simple linear prediction
        predicted_score = np.dot(self.feature_weights, feature_vector) + self.performance_baseline
        predicted_score = max(0.0, min(1.0, predicted_score))  # Clamp to [0,1]
        
        # Confidence based on model accuracy and data quality
        confidence = self.model_accuracy * self._calculate_data_quality(metrics)
        
        return predicted_score, confidence
    
    def predict_impact(self, current_metrics: SystemMetrics, 
                      action: OptimizationAction) -> Tuple[float, float]:
        """
        Predict impact of applying an optimization action.
        
        Returns:
            (predicted_improvement, confidence)
        """
        # Simulate the impact of the action on metrics
        modified_metrics = self._simulate_action_impact(current_metrics, action)
        
        # Predict performance before and after
        current_score, _ = self.predict_performance(current_metrics)
        modified_score, confidence = self.predict_performance(modified_metrics)
        
        improvement = modified_score - current_score
        
        return improvement, confidence * 0.8  # Lower confidence for predictions
    
    def _update_model(self):
        """Update model weights using simple gradient descent."""
        if len(self.metrics_history) < 20:
            return
        
        # Get recent observations
        recent_obs = list(self.metrics_history)[-20:]
        
        # Calculate gradients
        learning_rate = 0.01
        total_error = 0.0
        
        for metrics, actual_score in recent_obs:
            feature_vector = metrics.to_vector()
            predicted_score = np.dot(self.feature_weights, feature_vector) + self.performance_baseline
            
            error = actual_score - predicted_score
            total_error += abs(error)
            
            # Update weights
            self.feature_weights += learning_rate * error * feature_vector
            self.performance_baseline += learning_rate * error * 0.1
        
        # Update accuracy estimate
        avg_error = total_error / len(recent_obs)
        self.model_accuracy = max(0.1, 1.0 - avg_error)
        self.prediction_errors.append(avg_error)
    
    def _calculate_data_quality(self, metrics: SystemMetrics) -> float:
        """Calculate data quality score for confidence estimation."""
        # Check for reasonable metric ranges
        quality_score = 1.0
        
        if metrics.cpu_usage > 1.0 or metrics.cpu_usage < 0.0:
            quality_score *= 0.5
        if metrics.memory_usage > 1.0 or metrics.memory_usage < 0.0:
            quality_score *= 0.5
        if metrics.cache_hit_rate > 1.0 or metrics.cache_hit_rate < 0.0:
            quality_score *= 0.5
        
        return quality_score
    
    def _simulate_action_impact(self, metrics: SystemMetrics, 
                               action: OptimizationAction) -> SystemMetrics:
        """Simulate the impact of an action on system metrics."""
        # Create a copy of metrics
        import copy
        modified_metrics = copy.deepcopy(metrics)
        
        # Simple impact simulation based on action type
        if "cache" in action.parameter.lower():
            modified_metrics.cache_hit_rate *= 1.1  # Assume cache optimization improves hit rate
            modified_metrics.request_latency_ms *= 0.9  # Lower latency
        
        elif "memory" in action.parameter.lower():
            modified_metrics.memory_usage *= 0.95  # Assume better memory management
            modified_metrics.throughput_rps *= 1.05  # Slight throughput improvement
        
        elif "cpu" in action.parameter.lower():
            modified_metrics.cpu_usage *= 0.9  # Assume CPU optimization
            modified_metrics.throughput_rps *= 1.15  # Better throughput
        
        elif "connection" in action.parameter.lower():
            modified_metrics.active_connections = min(modified_metrics.active_connections, 
                                                    int(action.suggested_value))
            modified_metrics.queue_depth *= 0.8
        
        return modified_metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model performance information."""
        return {
            'observations': len(self.metrics_history),
            'accuracy': self.model_accuracy,
            'recent_error': np.mean(list(self.prediction_errors)) if self.prediction_errors else 0.0,
            'feature_weights': self.feature_weights.tolist(),
            'baseline': self.performance_baseline
        }


class IntelligentOptimizer:
    """
    ML-powered system optimizer that learns and adapts.
    """
    
    def __init__(self, target: OptimizationTarget = OptimizationTarget.BALANCED,
                 strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
        self.target = target
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # ML model for performance prediction
        self.performance_model = MLPerformanceModel()
        
        # Optimization state
        self.optimization_history = deque(maxlen=1000)
        self.current_configuration = {}
        self.baseline_metrics = None
        
        # Safety and rollback
        self.rollback_stack = []
        self.safety_threshold = 0.8  # Don't apply changes with <80% confidence
        
        # Learning parameters
        self.exploration_rate = 0.1  # 10% of actions are exploratory
        self.success_threshold = 0.05  # 5% improvement threshold
        
    def analyze_system(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """
        Analyze current system performance and identify optimization opportunities.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Analysis results with optimization recommendations
        """
        # Calculate performance score
        performance_score = self._calculate_performance_score(metrics)
        
        # Add to model training data
        self.performance_model.add_observation(metrics, performance_score)
        
        # Identify optimization opportunities
        opportunities = self._identify_opportunities(metrics)
        
        # Rank opportunities by impact and confidence
        ranked_opportunities = self._rank_opportunities(metrics, opportunities)
        
        # Generate report
        analysis = {
            'timestamp': metrics.timestamp.isoformat(),
            'current_performance': performance_score,
            'performance_prediction': self.performance_model.predict_performance(metrics),
            'optimization_opportunities': ranked_opportunities[:5],  # Top 5
            'system_health': self._assess_system_health(metrics),
            'recommendations': self._generate_recommendations(metrics, ranked_opportunities)
        }
        
        return analysis
    
    def optimize(self, metrics: SystemMetrics, max_actions: int = 3) -> List[OptimizationResult]:
        """
        Apply optimizations to the system.
        
        Args:
            metrics: Current system metrics
            max_actions: Maximum number of optimization actions to apply
            
        Returns:
            List of optimization results
        """
        results = []
        current_metrics = metrics
        
        # Get optimization opportunities
        analysis = self.analyze_system(current_metrics)
        opportunities = analysis['optimization_opportunities']
        
        actions_applied = 0
        for opportunity in opportunities:
            if actions_applied >= max_actions:
                break
            
            action = opportunity['action']
            
            # Safety check
            if action.confidence < self.safety_threshold:
                self.logger.info(f"Skipping action {action.name} due to low confidence: {action.confidence:.2f}")
                continue
            
            # Apply the optimization
            result = self._apply_optimization(action, current_metrics)
            results.append(result)
            
            if result.success:
                actions_applied += 1
                # Update current metrics for next iteration
                if result.metrics_after:
                    current_metrics = result.metrics_after
        
        return results
    
    def _calculate_performance_score(self, metrics: SystemMetrics) -> float:
        """Calculate overall system performance score (0-1)."""
        scores = {}
        
        # Individual metric scores
        scores['cpu'] = 1.0 - metrics.cpu_usage  # Lower CPU usage = better
        scores['memory'] = 1.0 - metrics.memory_usage
        scores['latency'] = max(0.0, 1.0 - metrics.request_latency_ms / 5000.0)  # <5s is good
        scores['throughput'] = min(1.0, metrics.throughput_rps / 1000.0)  # 1000 RPS = perfect
        scores['cache'] = metrics.cache_hit_rate
        scores['errors'] = 1.0 - metrics.error_rate
        
        # Weighted combination based on optimization target
        if self.target == OptimizationTarget.THROUGHPUT:
            weights = {'throughput': 0.4, 'cpu': 0.2, 'memory': 0.2, 'cache': 0.1, 'errors': 0.1}
        elif self.target == OptimizationTarget.LATENCY:
            weights = {'latency': 0.4, 'cache': 0.2, 'cpu': 0.2, 'memory': 0.1, 'errors': 0.1}
        elif self.target == OptimizationTarget.COST:
            weights = {'cpu': 0.3, 'memory': 0.3, 'throughput': 0.2, 'cache': 0.1, 'errors': 0.1}
        else:  # BALANCED
            weights = {'throughput': 0.2, 'latency': 0.2, 'cpu': 0.15, 'memory': 0.15, 
                      'cache': 0.15, 'errors': 0.13}
        
        # Calculate weighted score
        total_score = sum(scores.get(metric, 0.0) * weight 
                         for metric, weight in weights.items())
        
        return max(0.0, min(1.0, total_score))
    
    def _identify_opportunities(self, metrics: SystemMetrics) -> List[OptimizationAction]:
        """Identify potential optimization actions."""
        opportunities = []
        
        # CPU optimization
        if metrics.cpu_usage > 0.8:
            opportunities.append(OptimizationAction(
                name="Reduce CPU Usage",
                parameter="cpu_threads",
                current_value=4,
                suggested_value=6,
                confidence=0.8,
                expected_improvement=0.15,
                risk_level=0.3,
                rollback_plan="Reduce threads back to original value"
            ))
        
        # Memory optimization
        if metrics.memory_usage > 0.9:
            opportunities.append(OptimizationAction(
                name="Optimize Memory Usage",
                parameter="memory_pool_size",
                current_value="512MB",
                suggested_value="256MB",
                confidence=0.7,
                expected_improvement=0.2,
                risk_level=0.4,
                rollback_plan="Increase memory pool back to 512MB"
            ))
        
        # Cache optimization
        if metrics.cache_hit_rate < 0.8:
            opportunities.append(OptimizationAction(
                name="Improve Cache Hit Rate",
                parameter="cache_size",
                current_value=1000,
                suggested_value=2000,
                confidence=0.85,
                expected_improvement=0.1,
                risk_level=0.2,
                rollback_plan="Reduce cache size to original value"
            ))
        
        # Connection pool optimization
        if metrics.queue_depth > 50:
            opportunities.append(OptimizationAction(
                name="Optimize Connection Pool",
                parameter="max_connections",
                current_value=100,
                suggested_value=150,
                confidence=0.75,
                expected_improvement=0.08,
                risk_level=0.3,
                rollback_plan="Reduce max connections to 100"
            ))
        
        return opportunities
    
    def _rank_opportunities(self, metrics: SystemMetrics, 
                          opportunities: List[OptimizationAction]) -> List[Dict[str, Any]]:
        """Rank optimization opportunities by predicted impact."""
        ranked = []
        
        for action in opportunities:
            # Get ML model prediction
            predicted_improvement, ml_confidence = self.performance_model.predict_impact(metrics, action)
            
            # Combine with action confidence
            combined_confidence = (action.confidence + ml_confidence) / 2.0
            
            # Calculate value score (improvement / risk)
            value_score = (predicted_improvement * combined_confidence) / (action.risk_level + 0.1)
            
            ranked.append({
                'action': action,
                'predicted_improvement': predicted_improvement,
                'ml_confidence': ml_confidence,
                'combined_confidence': combined_confidence,
                'value_score': value_score
            })
        
        # Sort by value score
        ranked.sort(key=lambda x: x['value_score'], reverse=True)
        
        return ranked
    
    def _apply_optimization(self, action: OptimizationAction, 
                          metrics: SystemMetrics) -> OptimizationResult:
        """Apply an optimization action."""
        self.logger.info(f"Applying optimization: {action.name}")
        
        # Record current state for rollback
        rollback_info = {
            'parameter': action.parameter,
            'value': action.current_value,
            'timestamp': datetime.now()
        }
        self.rollback_stack.append(rollback_info)
        
        try:
            # Simulate applying the optimization
            success = self._execute_optimization_action(action)
            
            if success:
                # Create result with simulated improvement
                result = OptimizationResult(
                    action=action,
                    applied_at=datetime.now(),
                    success=True,
                    actual_improvement=action.expected_improvement * 0.8,  # Simulated
                    metrics_before=metrics,
                    metrics_after=self._simulate_improved_metrics(metrics, action)
                )
                
                self.logger.info(f"Successfully applied {action.name}")
                return result
            
            else:
                result = OptimizationResult(
                    action=action,
                    applied_at=datetime.now(),
                    success=False,
                    side_effects=["Optimization failed to apply"]
                )
                
                self.logger.error(f"Failed to apply {action.name}")
                return result
        
        except Exception as e:
            # Rollback on error
            self._rollback_last_action()
            
            result = OptimizationResult(
                action=action,
                applied_at=datetime.now(),
                success=False,
                side_effects=[f"Exception occurred: {str(e)}"]
            )
            
            self.logger.error(f"Exception applying {action.name}: {e}")
            return result
    
    def _execute_optimization_action(self, action: OptimizationAction) -> bool:
        """Execute the actual optimization (mocked for demo)."""
        # In production, this would make actual system changes
        self.logger.info(f"Would set {action.parameter} = {action.suggested_value}")
        
        # Simulate success/failure based on confidence
        return np.random.random() < action.confidence
    
    def _simulate_improved_metrics(self, metrics: SystemMetrics, 
                                 action: OptimizationAction) -> SystemMetrics:
        """Simulate metrics after optimization."""
        # Use the ML model's simulation
        return self.performance_model._simulate_action_impact(metrics, action)
    
    def _assess_system_health(self, metrics: SystemMetrics) -> Dict[str, str]:
        """Assess overall system health."""
        health = {}
        
        # CPU health
        if metrics.cpu_usage < 0.6:
            health['cpu'] = 'healthy'
        elif metrics.cpu_usage < 0.8:
            health['cpu'] = 'warning'
        else:
            health['cpu'] = 'critical'
        
        # Memory health
        if metrics.memory_usage < 0.7:
            health['memory'] = 'healthy'
        elif metrics.memory_usage < 0.9:
            health['memory'] = 'warning'
        else:
            health['memory'] = 'critical'
        
        # Latency health
        if metrics.request_latency_ms < 1000:
            health['latency'] = 'healthy'
        elif metrics.request_latency_ms < 3000:
            health['latency'] = 'warning'
        else:
            health['latency'] = 'critical'
        
        return health
    
    def _generate_recommendations(self, metrics: SystemMetrics, 
                                opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate human-readable recommendations."""
        recommendations = []
        
        if metrics.cpu_usage > 0.8:
            recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
        
        if metrics.cache_hit_rate < 0.7:
            recommendations.append("Cache hit rate is low - consider increasing cache size or improving cache strategy")
        
        if metrics.request_latency_ms > 2000:
            recommendations.append("High latency detected - investigate database queries and external API calls")
        
        if len(opportunities) > 0 and opportunities[0]['value_score'] > 0.5:
            top_action = opportunities[0]['action']
            recommendations.append(f"High-value optimization available: {top_action.name}")
        
        return recommendations
    
    def _rollback_last_action(self):
        """Rollback the last applied optimization."""
        if not self.rollback_stack:
            return
        
        rollback_info = self.rollback_stack.pop()
        self.logger.warning(f"Rolling back {rollback_info['parameter']} to {rollback_info['value']}")
        
        # In production, would actually revert the change
        # self._execute_rollback(rollback_info)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        model_info = self.performance_model.get_model_info()
        
        # Calculate success rate
        successful_optimizations = sum(1 for result in self.optimization_history if result.success)
        success_rate = successful_optimizations / len(self.optimization_history) if self.optimization_history else 0.0
        
        # Calculate average improvement
        improvements = [result.actual_improvement for result in self.optimization_history 
                       if result.success and result.actual_improvement]
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        return {
            'optimization_target': self.target.value,
            'strategy': self.strategy.value,
            'total_optimizations': len(self.optimization_history),
            'success_rate': success_rate,
            'average_improvement': avg_improvement,
            'model_accuracy': model_info['accuracy'],
            'recent_actions': [
                {
                    'name': result.action.name,
                    'success': result.success,
                    'improvement': result.actual_improvement,
                    'timestamp': result.applied_at.isoformat()
                }
                for result in list(self.optimization_history)[-10:]
            ],
            'model_info': model_info
        }


# Convenience functions
def create_optimizer(target: OptimizationTarget = OptimizationTarget.BALANCED) -> IntelligentOptimizer:
    """Create an ML optimizer with specified target."""
    return IntelligentOptimizer(target=target)


async def run_optimization_loop(optimizer: IntelligentOptimizer, 
                               metrics_source: Callable[[], SystemMetrics],
                               interval_seconds: int = 300):
    """Run continuous optimization loop."""
    while True:
        try:
            # Get current metrics
            current_metrics = metrics_source()
            
            # Analyze system
            analysis = optimizer.analyze_system(current_metrics)
            
            # Apply optimizations if needed
            if analysis['optimization_opportunities']:
                results = optimizer.optimize(current_metrics, max_actions=2)
                
                for result in results:
                    optimizer.optimization_history.append(result)
                    if result.success:
                        optimizer.logger.info(f"Applied optimization: {result.action.name}")
        
        except Exception as e:
            optimizer.logger.error(f"Error in optimization loop: {e}")
        
        await asyncio.sleep(interval_seconds)


# Demo function
def create_demo_metrics() -> SystemMetrics:
    """Create demo metrics for testing."""
    return SystemMetrics(
        timestamp=datetime.now(),
        cpu_usage=0.6 + 0.3 * np.random.random(),
        memory_usage=0.5 + 0.4 * np.random.random(),
        disk_io_ops=100 + 50 * np.random.random(),
        network_io_mbps=50 + 25 * np.random.random(),
        cache_hit_rate=0.7 + 0.2 * np.random.random(),
        request_latency_ms=1000 + 500 * np.random.random(),
        throughput_rps=500 + 200 * np.random.random(),
        error_rate=0.01 + 0.02 * np.random.random(),
        queue_depth=int(20 + 30 * np.random.random()),
        active_connections=int(100 + 50 * np.random.random())
    )