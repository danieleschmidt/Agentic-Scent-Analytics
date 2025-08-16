#!/usr/bin/env python3
"""
ML-Powered Performance Optimizer - Adaptive resource management and optimization
Part of Agentic Scent Analytics Platform

This module implements an advanced machine learning-powered performance optimization
system that continuously monitors, learns, and adapts system performance through
intelligent resource allocation, predictive scaling, and automated tuning.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import pickle
import hashlib
import logging
import threading
import psutil
import gc

import numpy as np
from collections import deque, defaultdict

from .config import ConfigManager
from .validation import AdvancedDataValidator
from .security import SecurityManager
from .metrics import PrometheusMetrics


class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    DATABASE_CONNECTIONS = "database_connections"
    CACHE_SIZE = "cache_size"
    THREAD_POOL = "thread_pool"
    ASYNC_WORKERS = "async_workers"


class PerformanceMetric(Enum):
    """Performance metrics to optimize"""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    MEMORY_EFFICIENCY = "memory_efficiency"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_DEPTH = "queue_depth"


@dataclass
class ResourceConfiguration:
    """Configuration for a specific resource"""
    resource_type: ResourceType
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    target_utilization: float
    cost_per_unit: float = 1.0
    adjustment_cooldown_seconds: int = 60


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_mbps: float
    network_io_mbps: float
    active_connections: int
    throughput_rps: float
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    cache_hit_rate: float
    queue_depth: int
    gc_collections: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationAction:
    """An optimization action to be executed"""
    resource_type: ResourceType
    action_type: str  # increase, decrease, tune
    current_value: float
    target_value: float
    expected_impact: float
    confidence: float
    execution_time: datetime
    reasoning: str
    rollback_plan: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result of an optimization action"""
    action: OptimizationAction
    executed: bool
    success: bool
    performance_before: PerformanceSnapshot
    performance_after: Optional[PerformanceSnapshot] = None
    actual_impact: Optional[float] = None
    side_effects: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0


class PerformancePredictor:
    """ML model for predicting performance impact of changes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.training_data: List[Dict] = []
        self.model_weights: Dict[str, float] = self._initialize_weights()
        self.feature_importance: Dict[str, float] = {}
        self.prediction_accuracy_history: deque = deque(maxlen=1000)
        
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize model weights based on domain knowledge"""
        return {
            # Resource impact weights
            'cpu_utilization': 0.25,
            'memory_utilization': 0.20,
            'disk_io': 0.15,
            'network_io': 0.15,
            'cache_efficiency': 0.15,
            'concurrent_load': 0.10,
            
            # Interaction effects
            'cpu_memory_interaction': 0.05,
            'io_contention': 0.03,
            'temporal_patterns': 0.02,
        }
    
    def predict_performance_impact(self, 
                                 action: OptimizationAction,
                                 current_state: PerformanceSnapshot,
                                 historical_context: List[PerformanceSnapshot]) -> Tuple[float, float]:
        """Predict performance impact and confidence"""
        
        # Extract features
        features = self._extract_features(action, current_state, historical_context)
        
        # Calculate predicted impact using weighted feature combination
        predicted_impact = 0.0
        total_weight = 0.0
        
        for feature_name, feature_value in features.items():
            if feature_name in self.model_weights:
                weight = self.model_weights[feature_name]
                predicted_impact += feature_value * weight
                total_weight += weight
        
        if total_weight > 0:
            predicted_impact /= total_weight
        
        # Calculate confidence based on historical accuracy and data quality
        confidence = self._calculate_prediction_confidence(features, historical_context)
        
        # Apply domain-specific adjustments
        adjusted_impact = self._apply_domain_adjustments(
            predicted_impact, action, current_state
        )
        
        return adjusted_impact, confidence
    
    def _extract_features(self, 
                         action: OptimizationAction,
                         current_state: PerformanceSnapshot,
                         historical_context: List[PerformanceSnapshot]) -> Dict[str, float]:
        """Extract features for prediction model"""
        features = {}
        
        # Current state features
        features['cpu_utilization'] = current_state.cpu_percent / 100.0
        features['memory_utilization'] = current_state.memory_percent / 100.0
        features['disk_io'] = min(1.0, current_state.disk_io_mbps / 1000.0)
        features['network_io'] = min(1.0, current_state.network_io_mbps / 1000.0)
        features['cache_efficiency'] = current_state.cache_hit_rate
        features['concurrent_load'] = min(1.0, current_state.active_connections / 1000.0)
        
        # Action features
        change_ratio = action.target_value / action.current_value if action.current_value > 0 else 1.0
        features['change_magnitude'] = abs(change_ratio - 1.0)
        features['resource_type_cpu'] = 1.0 if action.resource_type == ResourceType.CPU else 0.0
        features['resource_type_memory'] = 1.0 if action.resource_type == ResourceType.MEMORY else 0.0
        
        # Historical trend features
        if len(historical_context) >= 5:
            recent_snapshots = historical_context[-5:]
            
            # Trend in CPU usage
            cpu_trend = self._calculate_trend([s.cpu_percent for s in recent_snapshots])
            features['cpu_trend'] = cpu_trend
            
            # Trend in throughput
            throughput_trend = self._calculate_trend([s.throughput_rps for s in recent_snapshots])
            features['throughput_trend'] = throughput_trend
            
            # Volatility (standard deviation of recent metrics)
            latencies = [s.avg_latency_ms for s in recent_snapshots]
            features['latency_volatility'] = np.std(latencies) / np.mean(latencies) if np.mean(latencies) > 0 else 0.0
        
        # Interaction features
        features['cpu_memory_interaction'] = features['cpu_utilization'] * features['memory_utilization']
        features['io_contention'] = features['disk_io'] * features['network_io']
        
        # Time-based features
        hour_of_day = current_state.timestamp.hour
        features['peak_hours'] = 1.0 if 9 <= hour_of_day <= 17 else 0.0
        features['weekend'] = 1.0 if current_state.timestamp.weekday() >= 5 else 0.0
        
        return features
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from a series of values (-1 to 1)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        # Normalize slope to -1 to 1 range
        max_change = max(values) - min(values)
        if max_change > 0:
            normalized_slope = np.clip(slope / max_change, -1.0, 1.0)
        else:
            normalized_slope = 0.0
        
        return normalized_slope
    
    def _calculate_prediction_confidence(self, 
                                       features: Dict[str, float],
                                       historical_context: List[PerformanceSnapshot]) -> float:
        """Calculate confidence in prediction"""
        
        # Base confidence
        confidence = 0.5
        
        # Increase confidence with more historical data
        data_quality_score = min(1.0, len(historical_context) / 100.0)
        confidence += data_quality_score * 0.2
        
        # Increase confidence if we have recent similar scenarios
        similar_scenarios = self._count_similar_scenarios(features)
        similarity_score = min(1.0, similar_scenarios / 10.0)
        confidence += similarity_score * 0.2
        
        # Decrease confidence for volatile conditions
        if 'latency_volatility' in features and features['latency_volatility'] > 0.5:
            confidence -= 0.1
        
        # Model accuracy history
        if self.prediction_accuracy_history:
            recent_accuracy = np.mean(list(self.prediction_accuracy_history)[-20:])
            confidence += (recent_accuracy - 0.5) * 0.2
        
        return np.clip(confidence, 0.1, 0.95)
    
    def _count_similar_scenarios(self, features: Dict[str, float]) -> int:
        """Count similar scenarios in training data"""
        similar_count = 0
        
        for training_example in self.training_data[-100:]:  # Last 100 examples
            training_features = training_example.get('features', {})
            
            # Calculate feature similarity
            similarity = self._calculate_feature_similarity(features, training_features)
            
            if similarity > 0.8:
                similar_count += 1
        
        return similar_count
    
    def _calculate_feature_similarity(self, 
                                    features1: Dict[str, float],
                                    features2: Dict[str, float]) -> float:
        """Calculate similarity between two feature sets"""
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            return 0.0
        
        differences = []
        for feature in common_features:
            diff = abs(features1[feature] - features2[feature])
            differences.append(diff)
        
        avg_difference = np.mean(differences)
        similarity = max(0.0, 1.0 - avg_difference)
        
        return similarity
    
    def _apply_domain_adjustments(self, 
                                predicted_impact: float,
                                action: OptimizationAction,
                                current_state: PerformanceSnapshot) -> float:
        """Apply domain-specific adjustments to predictions"""
        
        adjusted_impact = predicted_impact
        
        # CPU scaling adjustments
        if action.resource_type == ResourceType.CPU:
            # CPU increases have diminishing returns at high utilization
            if current_state.cpu_percent > 80:
                adjusted_impact *= 0.7
            # CPU decreases are riskier under load
            elif action.target_value < action.current_value and current_state.throughput_rps > 100:
                adjusted_impact *= 0.8
        
        # Memory scaling adjustments
        elif action.resource_type == ResourceType.MEMORY:
            # Memory pressure can cause sudden performance drops
            if current_state.memory_percent > 85:
                adjusted_impact *= 1.2
            # GC pressure indicator
            if current_state.gc_collections > 10:
                adjusted_impact *= 1.1
        
        # Cache adjustments
        elif action.resource_type == ResourceType.CACHE_SIZE:
            # Cache hit rate strongly correlates with performance
            cache_effectiveness = current_state.cache_hit_rate
            adjusted_impact *= (0.5 + cache_effectiveness)
        
        return adjusted_impact
    
    def learn_from_result(self, result: OptimizationResult):
        """Learn from optimization result to improve future predictions"""
        
        if not result.executed or not result.performance_after:
            return
        
        # Calculate actual impact
        if result.actual_impact is None:
            before_score = self._calculate_performance_score(result.performance_before)
            after_score = self._calculate_performance_score(result.performance_after)
            actual_impact = after_score - before_score
        else:
            actual_impact = result.actual_impact
        
        # Extract features that were used for prediction
        features = self._extract_features(
            result.action,
            result.performance_before,
            []  # No historical context in this simplified version
        )
        
        # Store training example
        training_example = {
            'features': features,
            'action': {
                'resource_type': result.action.resource_type.value,
                'change_ratio': result.action.target_value / result.action.current_value,
                'action_type': result.action.action_type
            },
            'actual_impact': actual_impact,
            'predicted_impact': result.action.expected_impact,
            'timestamp': result.performance_before.timestamp.isoformat()
        }
        
        self.training_data.append(training_example)
        
        # Calculate prediction accuracy
        prediction_error = abs(actual_impact - result.action.expected_impact)
        accuracy = max(0.0, 1.0 - prediction_error)
        self.prediction_accuracy_history.append(accuracy)
        
        # Update model weights based on prediction error
        self._update_model_weights(features, prediction_error, actual_impact)
        
        # Keep only last 10000 training examples
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-10000:]
        
        self.logger.debug(f"Learned from optimization: accuracy={accuracy:.3f}, "
                         f"predicted={result.action.expected_impact:.3f}, "
                         f"actual={actual_impact:.3f}")
    
    def _calculate_performance_score(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate overall performance score from snapshot"""
        
        # Normalize metrics to 0-1 scale (higher is better)
        throughput_score = min(1.0, snapshot.throughput_rps / 1000.0)
        latency_score = max(0.0, 1.0 - snapshot.avg_latency_ms / 500.0)
        error_score = max(0.0, 1.0 - snapshot.error_rate * 10.0)
        cache_score = snapshot.cache_hit_rate
        
        # Resource efficiency (lower utilization with same performance is better)
        resource_efficiency = 1.0 - (snapshot.cpu_percent + snapshot.memory_percent) / 200.0
        resource_efficiency = max(0.0, resource_efficiency)
        
        # Weighted combination
        score = (
            throughput_score * 0.3 +
            latency_score * 0.3 +
            error_score * 0.2 +
            cache_score * 0.1 +
            resource_efficiency * 0.1
        )
        
        return score
    
    def _update_model_weights(self, 
                            features: Dict[str, float],
                            prediction_error: float,
                            actual_impact: float):
        """Update model weights based on prediction error"""
        
        learning_rate = 0.01
        
        # Update weights based on feature contribution to error
        for feature_name, feature_value in features.items():
            if feature_name in self.model_weights:
                # If prediction was wrong, adjust weight
                if prediction_error > 0.1:
                    # Gradient-like update
                    weight_adjustment = learning_rate * prediction_error * feature_value * actual_impact
                    
                    # Apply adjustment
                    self.model_weights[feature_name] -= weight_adjustment
                    
                    # Keep weights in reasonable bounds
                    self.model_weights[feature_name] = np.clip(
                        self.model_weights[feature_name], 0.01, 1.0
                    )
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for key in self.model_weights:
                self.model_weights[key] /= total_weight
    
    def export_model(self, filepath: str):
        """Export trained model to file"""
        model_data = {
            'weights': self.model_weights,
            'training_data_count': len(self.training_data),
            'accuracy_history': list(self.prediction_accuracy_history),
            'feature_importance': self.feature_importance,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        self.logger.info(f"Exported performance prediction model to {filepath}")
    
    def import_model(self, filepath: str):
        """Import trained model from file"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.model_weights = model_data.get('weights', self.model_weights)
            self.feature_importance = model_data.get('feature_importance', {})
            
            accuracy_history = model_data.get('accuracy_history', [])
            self.prediction_accuracy_history.extend(accuracy_history[-100:])
            
            self.logger.info(f"Imported performance prediction model from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to import model from {filepath}: {e}")


class AdaptiveResourceManager:
    """Manages and adapts system resources dynamically"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.resource_configs: Dict[ResourceType, ResourceConfiguration] = {}
        self.last_adjustments: Dict[ResourceType, datetime] = {}
        self.adjustment_history: List[OptimizationResult] = []
        self._initialize_resource_configs()
        
    def _initialize_resource_configs(self):
        """Initialize default resource configurations"""
        
        # CPU configuration
        self.resource_configs[ResourceType.CPU] = ResourceConfiguration(
            resource_type=ResourceType.CPU,
            current_value=psutil.cpu_count(),
            min_value=1,
            max_value=psutil.cpu_count() * 2,  # Allow hyperthreading
            step_size=1,
            target_utilization=70.0,
            cost_per_unit=1.0,
            adjustment_cooldown_seconds=300
        )
        
        # Memory configuration
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.resource_configs[ResourceType.MEMORY] = ResourceConfiguration(
            resource_type=ResourceType.MEMORY,
            current_value=total_memory_gb * 0.8,  # 80% of total
            min_value=total_memory_gb * 0.3,
            max_value=total_memory_gb * 0.95,
            step_size=total_memory_gb * 0.1,
            target_utilization=75.0,
            cost_per_unit=2.0,
            adjustment_cooldown_seconds=180
        )
        
        # Thread pool configuration
        self.resource_configs[ResourceType.THREAD_POOL] = ResourceConfiguration(
            resource_type=ResourceType.THREAD_POOL,
            current_value=20,
            min_value=5,
            max_value=100,
            step_size=5,
            target_utilization=80.0,
            cost_per_unit=0.1,
            adjustment_cooldown_seconds=60
        )
        
        # Cache size configuration
        self.resource_configs[ResourceType.CACHE_SIZE] = ResourceConfiguration(
            resource_type=ResourceType.CACHE_SIZE,
            current_value=1024,  # MB
            min_value=256,
            max_value=8192,
            step_size=256,
            target_utilization=85.0,
            cost_per_unit=0.5,
            adjustment_cooldown_seconds=120
        )
        
        # Database connections
        self.resource_configs[ResourceType.DATABASE_CONNECTIONS] = ResourceConfiguration(
            resource_type=ResourceType.DATABASE_CONNECTIONS,
            current_value=10,
            min_value=2,
            max_value=50,
            step_size=2,
            target_utilization=75.0,
            cost_per_unit=1.5,
            adjustment_cooldown_seconds=90
        )
    
    def can_adjust_resource(self, resource_type: ResourceType) -> bool:
        """Check if resource can be adjusted (cooldown period)"""
        
        if resource_type not in self.last_adjustments:
            return True
        
        config = self.resource_configs[resource_type]
        last_adjustment = self.last_adjustments[resource_type]
        cooldown = timedelta(seconds=config.adjustment_cooldown_seconds)
        
        return datetime.now() - last_adjustment > cooldown
    
    def calculate_resource_adjustment(self, 
                                    resource_type: ResourceType,
                                    current_metrics: PerformanceSnapshot,
                                    target_improvement: float) -> Optional[OptimizationAction]:
        """Calculate optimal resource adjustment"""
        
        if not self.can_adjust_resource(resource_type):
            return None
        
        config = self.resource_configs[resource_type]
        current_utilization = self._get_current_utilization(resource_type, current_metrics)
        
        # Determine adjustment direction and magnitude
        if current_utilization > config.target_utilization:
            # Scale up
            adjustment_factor = min(2.0, current_utilization / config.target_utilization)
            target_value = min(
                config.max_value,
                config.current_value + (config.step_size * adjustment_factor)
            )
            action_type = "increase"
            
        elif current_utilization < config.target_utilization * 0.5:
            # Scale down (conservative)
            adjustment_factor = config.target_utilization / max(current_utilization, 1.0)
            target_value = max(
                config.min_value,
                config.current_value - config.step_size
            )
            action_type = "decrease"
            
        else:
            # No adjustment needed
            return None
        
        # Calculate expected impact (simplified)
        utilization_gap = abs(current_utilization - config.target_utilization)
        expected_impact = min(target_improvement, utilization_gap / 100.0)
        
        # Confidence based on how far we are from target
        confidence = max(0.5, 1.0 - utilization_gap / 100.0)
        
        action = OptimizationAction(
            resource_type=resource_type,
            action_type=action_type,
            current_value=config.current_value,
            target_value=target_value,
            expected_impact=expected_impact,
            confidence=confidence,
            execution_time=datetime.now(),
            reasoning=f"Current utilization {current_utilization:.1f}% vs target {config.target_utilization:.1f}%",
            rollback_plan=f"Revert to {config.current_value}"
        )
        
        return action
    
    def _get_current_utilization(self, 
                               resource_type: ResourceType,
                               metrics: PerformanceSnapshot) -> float:
        """Get current utilization for a resource type"""
        
        if resource_type == ResourceType.CPU:
            return metrics.cpu_percent
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_percent
        elif resource_type == ResourceType.THREAD_POOL:
            # Simulate thread pool utilization
            return min(100.0, metrics.active_connections / 20.0 * 100.0)
        elif resource_type == ResourceType.CACHE_SIZE:
            # Cache utilization based on hit rate (inverse relationship)
            return max(0.0, (1.0 - metrics.cache_hit_rate) * 100.0)
        elif resource_type == ResourceType.DATABASE_CONNECTIONS:
            # Simulate DB connection utilization
            return min(100.0, metrics.active_connections / 10.0 * 100.0)
        else:
            return 50.0  # Default
    
    async def execute_resource_adjustment(self, action: OptimizationAction) -> bool:
        """Execute a resource adjustment"""
        
        try:
            config = self.resource_configs[action.resource_type]
            
            # Simulate resource adjustment execution
            if action.resource_type == ResourceType.CPU:
                success = await self._adjust_cpu_allocation(action.target_value)
            elif action.resource_type == ResourceType.MEMORY:
                success = await self._adjust_memory_allocation(action.target_value)
            elif action.resource_type == ResourceType.THREAD_POOL:
                success = await self._adjust_thread_pool(action.target_value)
            elif action.resource_type == ResourceType.CACHE_SIZE:
                success = await self._adjust_cache_size(action.target_value)
            elif action.resource_type == ResourceType.DATABASE_CONNECTIONS:
                success = await self._adjust_db_connections(action.target_value)
            else:
                success = False
            
            if success:
                # Update configuration
                config.current_value = action.target_value
                self.last_adjustments[action.resource_type] = datetime.now()
                
                self.logger.info(f"Successfully adjusted {action.resource_type.value} "
                               f"to {action.target_value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to execute resource adjustment: {e}")
            return False
    
    # Placeholder methods for actual resource adjustments
    async def _adjust_cpu_allocation(self, target_value: float) -> bool:
        """Adjust CPU allocation (container/cgroup limits)"""
        # In real implementation, this would adjust container CPU limits
        await asyncio.sleep(0.1)  # Simulate adjustment time
        return True
    
    async def _adjust_memory_allocation(self, target_value: float) -> bool:
        """Adjust memory allocation"""
        # In real implementation, this would adjust container memory limits
        await asyncio.sleep(0.1)
        
        # Trigger garbage collection to free up memory if reducing
        if target_value < self.resource_configs[ResourceType.MEMORY].current_value:
            gc.collect()
        
        return True
    
    async def _adjust_thread_pool(self, target_value: float) -> bool:
        """Adjust thread pool size"""
        # In real implementation, this would adjust executor thread pool sizes
        await asyncio.sleep(0.05)
        return True
    
    async def _adjust_cache_size(self, target_value: float) -> bool:
        """Adjust cache size"""
        # In real implementation, this would adjust Redis/Memcached cache sizes
        await asyncio.sleep(0.05)
        return True
    
    async def _adjust_db_connections(self, target_value: float) -> bool:
        """Adjust database connection pool size"""
        # In real implementation, this would adjust SQLAlchemy pool sizes
        await asyncio.sleep(0.05)
        return True


class MLPerformanceOptimizer:
    """Main ML-powered performance optimizer"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.validation = AdvancedDataValidator()
        self.security = SecurityManager()
        self.metrics = PrometheusMetrics()
        
        self.predictor = PerformancePredictor()
        self.resource_manager = AdaptiveResourceManager()
        
        # Optimization settings
        self.optimization_strategy = OptimizationStrategy.ADAPTIVE
        self.target_performance_score = 0.85
        self.max_optimizations_per_hour = 10
        self.optimization_interval_seconds = 300  # 5 minutes
        
        # State tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.optimization_results: List[OptimizationResult] = []
        self.last_optimization_time = datetime.now()
        self.optimizations_this_hour = 0
        self.hour_counter = datetime.now().hour
        
        # Performance monitoring
        self._monitoring_active = False
        self._optimization_lock = asyncio.Lock()
        
    async def start_continuous_optimization(self):
        """Start continuous performance optimization loop"""
        
        self.logger.info("Starting continuous ML-powered performance optimization")
        self._monitoring_active = True
        
        # Start monitoring and optimization tasks
        tasks = [
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._optimization_loop()),
            asyncio.create_task(self._model_learning_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in continuous optimization: {e}")
        finally:
            self._monitoring_active = False
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring"""
        
        while self._monitoring_active:
            try:
                # Collect current performance snapshot
                snapshot = await self._collect_performance_snapshot()
                self.performance_history.append(snapshot)
                
                # Log performance metrics
                self.logger.debug(f"Performance snapshot: "
                                f"CPU={snapshot.cpu_percent:.1f}%, "
                                f"Memory={snapshot.memory_percent:.1f}%, "
                                f"Latency={snapshot.avg_latency_ms:.1f}ms, "
                                f"Throughput={snapshot.throughput_rps:.1f}rps")
                
                # Reset hourly counter
                current_hour = datetime.now().hour
                if current_hour != self.hour_counter:
                    self.optimizations_this_hour = 0
                    self.hour_counter = current_hour
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_loop(self):
        """Main optimization decision loop"""
        
        while self._monitoring_active:
            try:
                async with self._optimization_lock:
                    await self._run_optimization_cycle()
                
                await asyncio.sleep(self.optimization_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _model_learning_loop(self):
        """Model learning and adaptation loop"""
        
        while self._monitoring_active:
            try:
                # Update model with recent results
                recent_results = [r for r in self.optimization_results 
                                if r.executed and r.performance_after]
                
                for result in recent_results[-10:]:  # Last 10 results
                    self.predictor.learn_from_result(result)
                
                # Adapt optimization strategy based on performance
                await self._adapt_optimization_strategy()
                
                # Export model periodically
                if len(self.optimization_results) % 50 == 0:
                    model_path = f"performance_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.predictor.export_model(model_path)
                
                await asyncio.sleep(1800)  # Learn every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in model learning: {e}")
                await asyncio.sleep(300)
    
    async def _run_optimization_cycle(self):
        """Run a single optimization cycle"""
        
        if not self.performance_history:
            return
        
        # Check optimization limits
        if self.optimizations_this_hour >= self.max_optimizations_per_hour:
            self.logger.debug("Optimization limit reached for this hour")
            return
        
        current_snapshot = self.performance_history[-1]
        current_score = self.predictor._calculate_performance_score(current_snapshot)
        
        # Only optimize if performance is below target
        if current_score >= self.target_performance_score:
            self.logger.debug(f"Performance score {current_score:.3f} meets target {self.target_performance_score:.3f}")
            return
        
        self.logger.info(f"Performance score {current_score:.3f} below target, initiating optimization")
        
        # Identify optimization opportunities
        optimization_actions = await self._identify_optimization_opportunities(current_snapshot)
        
        if not optimization_actions:
            self.logger.debug("No optimization opportunities identified")
            return
        
        # Select best action
        best_action = self._select_best_optimization_action(optimization_actions)
        
        if best_action and best_action.confidence > 0.6:
            # Execute optimization
            result = await self._execute_optimization(best_action, current_snapshot)
            self.optimization_results.append(result)
            
            if result.success:
                self.optimizations_this_hour += 1
                self.last_optimization_time = datetime.now()
    
    async def _collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance metrics"""
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Simulate application-specific metrics
        # In real implementation, these would come from application monitoring
        throughput_rps = np.random.normal(120, 20)
        avg_latency_ms = np.random.normal(95, 15)
        p95_latency_ms = avg_latency_ms * 1.8
        error_rate = np.random.uniform(0.001, 0.01)
        cache_hit_rate = np.random.uniform(0.8, 0.95)
        queue_depth = np.random.randint(0, 20)
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_mbps=0.0,  # Simplified for now
            network_io_mbps=0.0,  # Simplified for now
            active_connections=np.random.randint(50, 200),
            throughput_rps=max(0, throughput_rps),
            avg_latency_ms=max(10, avg_latency_ms),
            p95_latency_ms=max(avg_latency_ms, p95_latency_ms),
            error_rate=max(0, error_rate),
            cache_hit_rate=np.clip(cache_hit_rate, 0, 1),
            queue_depth=max(0, queue_depth),
            gc_collections=len(gc.get_stats())
        )
    
    async def _identify_optimization_opportunities(self, 
                                                 current_snapshot: PerformanceSnapshot) -> List[OptimizationAction]:
        """Identify potential optimization actions"""
        
        opportunities = []
        target_improvement = self.target_performance_score - self.predictor._calculate_performance_score(current_snapshot)
        
        # Check each resource type for optimization opportunities
        for resource_type in ResourceType:
            action = self.resource_manager.calculate_resource_adjustment(
                resource_type, current_snapshot, target_improvement
            )
            
            if action:
                # Use ML predictor to estimate impact and confidence
                predicted_impact, confidence = self.predictor.predict_performance_impact(
                    action, current_snapshot, list(self.performance_history)[-20:]
                )
                
                action.expected_impact = predicted_impact
                action.confidence = confidence
                
                opportunities.append(action)
        
        return opportunities
    
    def _select_best_optimization_action(self, 
                                       actions: List[OptimizationAction]) -> Optional[OptimizationAction]:
        """Select the best optimization action based on impact and confidence"""
        
        if not actions:
            return None
        
        # Calculate score for each action
        scored_actions = []
        for action in actions:
            # Score based on expected impact, confidence, and cost
            config = self.resource_manager.resource_configs[action.resource_type]
            
            cost_factor = 1.0 / (1.0 + config.cost_per_unit)
            score = action.expected_impact * action.confidence * cost_factor
            
            scored_actions.append((score, action))
        
        # Sort by score and return best
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_action = scored_actions[0]
        
        self.logger.info(f"Selected optimization: {best_action.resource_type.value} "
                        f"{best_action.action_type} to {best_action.target_value} "
                        f"(score={best_score:.3f}, confidence={best_action.confidence:.3f})")
        
        return best_action
    
    async def _execute_optimization(self, 
                                  action: OptimizationAction,
                                  performance_before: PerformanceSnapshot) -> OptimizationResult:
        """Execute an optimization action and measure results"""
        
        start_time = time.time()
        
        result = OptimizationResult(
            action=action,
            executed=False,
            success=False,
            performance_before=performance_before
        )
        
        try:
            # Execute the resource adjustment
            success = await self.resource_manager.execute_resource_adjustment(action)
            result.executed = True
            result.success = success
            
            if success:
                # Wait for changes to take effect
                await asyncio.sleep(30)
                
                # Collect post-optimization metrics
                performance_after = await self._collect_performance_snapshot()
                result.performance_after = performance_after
                
                # Calculate actual impact
                before_score = self.predictor._calculate_performance_score(performance_before)
                after_score = self.predictor._calculate_performance_score(performance_after)
                result.actual_impact = after_score - before_score
                
                # Check for side effects
                result.side_effects = self._detect_side_effects(performance_before, performance_after)
                
                # Rollback if negative impact
                if result.actual_impact < -0.05:  # 5% performance degradation
                    self.logger.warning(f"Optimization caused performance degradation, rolling back")
                    await self._rollback_optimization(action)
                    result.side_effects.append("performance_degradation_rollback")
                
            result.execution_time_seconds = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Optimization execution failed: {e}")
            result.executed = True
            result.success = False
            result.execution_time_seconds = time.time() - start_time
            result.side_effects.append(f"execution_error: {str(e)}")
        
        return result
    
    def _detect_side_effects(self, 
                           before: PerformanceSnapshot,
                           after: PerformanceSnapshot) -> List[str]:
        """Detect potential side effects of optimization"""
        
        side_effects = []
        
        # Check for significant increases in error rate
        if after.error_rate > before.error_rate * 1.5:
            side_effects.append("increased_error_rate")
        
        # Check for memory pressure
        if after.memory_percent > 90 and after.memory_percent > before.memory_percent + 10:
            side_effects.append("memory_pressure")
        
        # Check for latency spikes
        if after.p95_latency_ms > before.p95_latency_ms * 1.3:
            side_effects.append("latency_spike")
        
        # Check for resource contention
        if after.cpu_percent > 95 and after.cpu_percent > before.cpu_percent + 15:
            side_effects.append("cpu_contention")
        
        return side_effects
    
    async def _rollback_optimization(self, action: OptimizationAction):
        """Rollback an optimization action"""
        
        try:
            # Create rollback action
            rollback_action = OptimizationAction(
                resource_type=action.resource_type,
                action_type="rollback",
                current_value=action.target_value,
                target_value=action.current_value,
                expected_impact=0.0,
                confidence=1.0,
                execution_time=datetime.now(),
                reasoning="Rollback due to negative impact"
            )
            
            # Execute rollback
            await self.resource_manager.execute_resource_adjustment(rollback_action)
            
            self.logger.info(f"Successfully rolled back {action.resource_type.value} optimization")
            
        except Exception as e:
            self.logger.error(f"Failed to rollback optimization: {e}")
    
    async def _adapt_optimization_strategy(self):
        """Adapt optimization strategy based on historical performance"""
        
        if len(self.optimization_results) < 10:
            return
        
        recent_results = self.optimization_results[-20:]
        successful_optimizations = [r for r in recent_results if r.success and r.actual_impact > 0]
        
        success_rate = len(successful_optimizations) / len(recent_results)
        
        # Adapt strategy based on success rate
        if success_rate < 0.5:
            # Too many failures, be more conservative
            self.optimization_strategy = OptimizationStrategy.CONSERVATIVE
            self.max_optimizations_per_hour = max(2, self.max_optimizations_per_hour - 1)
            self.target_performance_score = min(0.95, self.target_performance_score + 0.05)
            
        elif success_rate > 0.8:
            # High success rate, can be more aggressive
            self.optimization_strategy = OptimizationStrategy.AGGRESSIVE
            self.max_optimizations_per_hour = min(20, self.max_optimizations_per_hour + 1)
            self.target_performance_score = max(0.7, self.target_performance_score - 0.02)
        
        else:
            # Balanced approach
            self.optimization_strategy = OptimizationStrategy.BALANCED
        
        self.logger.info(f"Adapted optimization strategy to {self.optimization_strategy.value} "
                        f"(success_rate={success_rate:.2f}, max_opts={self.max_optimizations_per_hour})")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization activity"""
        
        if not self.optimization_results:
            return {'message': 'No optimization data available'}
        
        recent_results = self.optimization_results[-50:]
        successful_results = [r for r in recent_results if r.success]
        
        total_impact = sum(r.actual_impact or 0 for r in successful_results)
        avg_impact = total_impact / len(successful_results) if successful_results else 0
        
        current_score = 0.5
        if self.performance_history:
            current_score = self.predictor._calculate_performance_score(self.performance_history[-1])
        
        resource_adjustments = defaultdict(int)
        for result in recent_results:
            resource_adjustments[result.action.resource_type.value] += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_performance_score': current_score,
            'target_performance_score': self.target_performance_score,
            'optimization_strategy': self.optimization_strategy.value,
            'total_optimizations': len(self.optimization_results),
            'recent_success_rate': len(successful_results) / len(recent_results) if recent_results else 0,
            'average_impact': avg_impact,
            'optimizations_this_hour': self.optimizations_this_hour,
            'resource_adjustments': dict(resource_adjustments),
            'model_accuracy': (
                np.mean(list(self.predictor.prediction_accuracy_history))
                if self.predictor.prediction_accuracy_history else 0.5
            )
        }
    
    def export_optimization_report(self) -> str:
        """Export detailed optimization report"""
        
        summary = self.get_optimization_summary()
        
        # Add detailed results
        detailed_results = []
        for result in self.optimization_results[-20:]:
            detailed_results.append({
                'timestamp': result.performance_before.timestamp.isoformat(),
                'resource_type': result.action.resource_type.value,
                'action_type': result.action.action_type,
                'success': result.success,
                'expected_impact': result.action.expected_impact,
                'actual_impact': result.actual_impact,
                'confidence': result.action.confidence,
                'side_effects': result.side_effects,
                'execution_time_seconds': result.execution_time_seconds
            })
        
        report = {
            'summary': summary,
            'detailed_results': detailed_results,
            'performance_history': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'performance_score': self.predictor._calculate_performance_score(snapshot),
                    'cpu_percent': snapshot.cpu_percent,
                    'memory_percent': snapshot.memory_percent,
                    'avg_latency_ms': snapshot.avg_latency_ms,
                    'throughput_rps': snapshot.throughput_rps,
                    'error_rate': snapshot.error_rate
                }
                for snapshot in list(self.performance_history)[-100:]
            ]
        }
        
        return json.dumps(report, indent=2, default=str)


# Factory function
def create_ml_performance_optimizer(config_path: Optional[str] = None) -> MLPerformanceOptimizer:
    """Create and configure ML performance optimizer"""
    config = ConfigManager(config_path)
    return MLPerformanceOptimizer(config)


# CLI interface
if __name__ == "__main__":
    import sys
    
    async def main():
        optimizer = create_ml_performance_optimizer()
        
        if len(sys.argv) > 1 and sys.argv[1] == "start":
            # Start continuous optimization
            print("Starting ML-powered performance optimization...")
            await optimizer.start_continuous_optimization()
            
        elif len(sys.argv) > 1 and sys.argv[1] == "summary":
            # Print summary
            summary = optimizer.get_optimization_summary()
            print(json.dumps(summary, indent=2))
            
        elif len(sys.argv) > 1 and sys.argv[1] == "report":
            # Export detailed report
            report = optimizer.export_optimization_report()
            with open("ml_performance_report.json", "w") as f:
                f.write(report)
            print("Detailed report saved to ml_performance_report.json")
            
        else:
            print("Usage:")
            print("  python ml_performance_optimizer.py start    # Start continuous optimization")
            print("  python ml_performance_optimizer.py summary  # Show optimization summary")
            print("  python ml_performance_optimizer.py report   # Export detailed report")
    
    asyncio.run(main())