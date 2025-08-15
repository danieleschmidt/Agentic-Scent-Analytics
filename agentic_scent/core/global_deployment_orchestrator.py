#!/usr/bin/env python3
"""
Global Deployment Orchestrator - Multi-region auto-scaling and deployment management
Part of Agentic Scent Analytics Platform

This module implements a sophisticated global deployment orchestrator that manages
multi-region deployments, auto-scaling, blue-green deployments, and intelligent
traffic routing across geographical regions with compliance and latency optimization.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import hashlib
import logging
import subprocess
import yaml

import numpy as np
from .config import ConfigManager
from .validation import ValidationManager
from .security import SecurityManager
from .performance import PerformanceMonitor
from .metrics import MetricsCollector


class DeploymentRegion(Enum):
    """Supported deployment regions"""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"


class DeploymentStrategy(Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TEST = "ab_test"


class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class ComplianceRegime(Enum):
    """Compliance regimes for different regions"""
    GDPR = "gdpr"  # EU
    CCPA = "ccpa"  # California
    PDPA = "pdpa"  # Singapore
    PIPEDA = "pipeda"  # Canada
    LGPD = "lgpd"  # Brazil


@dataclass
class RegionConfig:
    """Configuration for a specific region"""
    region: DeploymentRegion
    compliance_regime: ComplianceRegime
    kubernetes_cluster: str
    container_registry: str
    data_residency_required: bool
    min_instances: int
    max_instances: int
    target_cpu_utilization: float
    target_latency_ms: int
    allowed_deployment_hours: List[int]  # UTC hours when deployment is allowed
    maintenance_window: Tuple[int, int]  # Start and end hour for maintenance
    disaster_recovery_region: Optional[DeploymentRegion] = None


@dataclass
class DeploymentTarget:
    """Target for deployment"""
    region: DeploymentRegion
    environment: str  # dev, staging, prod
    version: str
    instances: int
    strategy: DeploymentStrategy
    rollback_threshold: float  # Error rate threshold for auto-rollback
    health_check_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentMetrics:
    """Metrics for a deployment"""
    success_rate: float
    avg_response_time_ms: float
    error_rate: float
    cpu_utilization: float
    memory_utilization: float
    throughput_rps: float
    active_connections: int
    health_score: float


@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    deployment_id: str
    target: DeploymentTarget
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: Optional[DeploymentMetrics] = None
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None
    traffic_split: Dict[str, float] = field(default_factory=dict)


class TrafficRouter:
    """Intelligent traffic routing for multi-region deployments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.routing_rules: Dict[str, Any] = {}
        self.latency_matrix: Dict[Tuple[str, str], float] = {}
        
    def calculate_optimal_routing(self, user_location: str,
                                regions: List[DeploymentRegion],
                                region_health: Dict[DeploymentRegion, float]) -> Dict[DeploymentRegion, float]:
        """Calculate optimal traffic distribution across regions"""
        
        # Calculate latency scores (lower is better)
        latency_scores = {}
        for region in regions:
            latency = self._estimate_latency(user_location, region.value)
            latency_scores[region] = 1.0 / (1.0 + latency / 100.0)  # Normalize
        
        # Calculate composite scores
        composite_scores = {}
        for region in regions:
            health = region_health.get(region, 0.5)
            latency_score = latency_scores[region]
            
            # Weighted combination: 60% health, 40% latency
            composite_scores[region] = (health * 0.6) + (latency_score * 0.4)
        
        # Convert to traffic distribution
        total_score = sum(composite_scores.values())
        if total_score == 0:
            # Fallback: equal distribution
            return {region: 1.0 / len(regions) for region in regions}
        
        traffic_distribution = {}
        for region, score in composite_scores.items():
            traffic_distribution[region] = score / total_score
        
        return traffic_distribution
    
    def _estimate_latency(self, user_location: str, region: str) -> float:
        """Estimate latency between user location and region"""
        # Simplified latency estimation based on geographical distance
        latency_map = {
            ('us', 'us-east-1'): 20,
            ('us', 'us-west-2'): 30,
            ('us', 'eu-west-1'): 120,
            ('eu', 'eu-west-1'): 20,
            ('eu', 'eu-central-1'): 30,
            ('eu', 'us-east-1'): 120,
            ('asia', 'ap-southeast-1'): 20,
            ('asia', 'ap-northeast-1'): 40,
            ('asia', 'us-west-2'): 150,
        }
        
        # Extract continent from user location
        continent = user_location.split('-')[0] if '-' in user_location else user_location
        
        return latency_map.get((continent, region), 100)  # Default 100ms
    
    def update_routing_rules(self, rules: Dict[str, Any]):
        """Update traffic routing rules"""
        self.routing_rules.update(rules)
        self.logger.info("Updated traffic routing rules")
    
    def apply_geo_restrictions(self, traffic_distribution: Dict[DeploymentRegion, float],
                             user_country: str,
                             compliance_requirements: Dict[str, List[str]]) -> Dict[DeploymentRegion, float]:
        """Apply geo-restrictions based on compliance requirements"""
        restricted_distribution = {}
        
        for region, weight in traffic_distribution.items():
            # Check if user's country has restrictions for this region
            allowed_countries = compliance_requirements.get(region.value, [])
            
            if not allowed_countries or user_country in allowed_countries:
                restricted_distribution[region] = weight
        
        # Renormalize if some regions were excluded
        total_weight = sum(restricted_distribution.values())
        if total_weight > 0:
            return {region: weight / total_weight 
                   for region, weight in restricted_distribution.items()}
        else:
            # Fallback: route to first available region
            first_region = next(iter(traffic_distribution.keys()))
            return {first_region: 1.0}


class AutoScaler:
    """Intelligent auto-scaling based on ML predictions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaling_history: List[Dict] = []
        self.prediction_model = self._initialize_prediction_model()
        
    def _initialize_prediction_model(self) -> Dict[str, Any]:
        """Initialize simple prediction model"""
        return {
            'weights': {
                'cpu_utilization': 0.3,
                'memory_utilization': 0.2,
                'request_rate': 0.25,
                'response_time': 0.15,
                'error_rate': 0.1
            },
            'thresholds': {
                'scale_up': 0.7,
                'scale_down': 0.3
            }
        }
    
    async def calculate_optimal_instances(self, current_metrics: DeploymentMetrics,
                                        current_instances: int,
                                        region_config: RegionConfig) -> int:
        """Calculate optimal number of instances based on current metrics"""
        
        # Calculate scaling score
        scaling_score = self._calculate_scaling_score(current_metrics)
        
        # Predict future load (simplified)
        predicted_load = self._predict_future_load(current_metrics)
        
        # Determine scaling action
        if scaling_score > self.prediction_model['thresholds']['scale_up']:
            # Scale up
            scale_factor = min(2.0, scaling_score * 1.5)
            new_instances = min(
                int(current_instances * scale_factor),
                region_config.max_instances
            )
        elif scaling_score < self.prediction_model['thresholds']['scale_down']:
            # Scale down
            scale_factor = max(0.5, scaling_score * 2.0)
            new_instances = max(
                int(current_instances * scale_factor),
                region_config.min_instances
            )
        else:
            # No scaling needed
            new_instances = current_instances
        
        # Record scaling decision
        self.scaling_history.append({
            'timestamp': datetime.now().isoformat(),
            'current_instances': current_instances,
            'new_instances': new_instances,
            'scaling_score': scaling_score,
            'metrics': {
                'cpu_utilization': current_metrics.cpu_utilization,
                'memory_utilization': current_metrics.memory_utilization,
                'response_time': current_metrics.avg_response_time_ms,
                'error_rate': current_metrics.error_rate
            }
        })
        
        # Keep only last 1000 entries
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-1000:]
        
        return new_instances
    
    def _calculate_scaling_score(self, metrics: DeploymentMetrics) -> float:
        """Calculate scaling score based on current metrics"""
        weights = self.prediction_model['weights']
        
        # Normalize metrics to 0-1 scale
        cpu_score = min(1.0, metrics.cpu_utilization / 100.0)
        memory_score = min(1.0, metrics.memory_utilization / 100.0)
        
        # Response time score (higher response time = higher score)
        response_score = min(1.0, metrics.avg_response_time_ms / 500.0)
        
        # Error rate score
        error_score = min(1.0, metrics.error_rate * 10.0)
        
        # Request rate score (simplified)
        request_score = min(1.0, metrics.throughput_rps / 1000.0)
        
        # Weighted combination
        scaling_score = (
            cpu_score * weights['cpu_utilization'] +
            memory_score * weights['memory_utilization'] +
            request_score * weights['request_rate'] +
            response_score * weights['response_time'] +
            error_score * weights['error_rate']
        )
        
        return scaling_score
    
    def _predict_future_load(self, current_metrics: DeploymentMetrics) -> float:
        """Predict future load based on historical patterns"""
        if len(self.scaling_history) < 10:
            return 1.0  # No enough data, assume current load
        
        # Simple trend analysis
        recent_metrics = self.scaling_history[-10:]
        cpu_values = [m['metrics']['cpu_utilization'] for m in recent_metrics]
        
        # Calculate trend
        if len(cpu_values) >= 3:
            trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
            # Predict next value
            predicted_cpu = cpu_values[-1] + trend
            return max(0.1, min(2.0, predicted_cpu / 50.0))  # Normalize
        
        return 1.0
    
    def adapt_thresholds(self):
        """Adapt scaling thresholds based on historical performance"""
        if len(self.scaling_history) < 50:
            return
        
        # Analyze scaling decisions and their outcomes
        recent_history = self.scaling_history[-50:]
        
        # Calculate success rate of scaling decisions
        # (This is simplified - in real implementation, you'd track actual outcomes)
        scale_up_decisions = [h for h in recent_history 
                            if h['new_instances'] > h['current_instances']]
        
        if len(scale_up_decisions) > 10:
            # Adjust thresholds based on frequency of scaling
            current_threshold = self.prediction_model['thresholds']['scale_up']
            
            # If scaling too frequently, increase threshold
            scaling_frequency = len(scale_up_decisions) / len(recent_history)
            if scaling_frequency > 0.3:
                new_threshold = min(0.9, current_threshold * 1.1)
                self.prediction_model['thresholds']['scale_up'] = new_threshold
                self.logger.info(f"Adapted scale-up threshold to {new_threshold:.3f}")


class DeploymentHealth:
    """Monitor and assess deployment health"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_checks: Dict[str, Callable] = {}
        
    async def assess_deployment_health(self, deployment_target: DeploymentTarget) -> float:
        """Assess overall health of a deployment (0-1 scale)"""
        health_scores = []
        
        # Basic health check
        basic_health = await self._basic_health_check(deployment_target)
        health_scores.append(('basic', basic_health, 0.4))
        
        # Performance health
        perf_health = await self._performance_health_check(deployment_target)
        health_scores.append(('performance', perf_health, 0.3))
        
        # Security health
        security_health = await self._security_health_check(deployment_target)
        health_scores.append(('security', security_health, 0.2))
        
        # Business metrics health
        business_health = await self._business_metrics_health_check(deployment_target)
        health_scores.append(('business', business_health, 0.1))
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for name, score, weight in health_scores:
            if score is not None:
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    async def _basic_health_check(self, target: DeploymentTarget) -> Optional[float]:
        """Basic health check via HTTP endpoint"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    target.health_check_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return 1.0
                    elif response.status < 500:
                        return 0.7  # Client error but service is responding
                    else:
                        return 0.3  # Server error
                        
        except Exception as e:
            self.logger.warning(f"Health check failed for {target.region.value}: {e}")
            return 0.0
    
    async def _performance_health_check(self, target: DeploymentTarget) -> Optional[float]:
        """Check performance metrics"""
        try:
            # Simulate performance check
            response_time = np.random.normal(target.metadata.get('expected_response_time', 100), 20)
            
            if response_time <= target.metadata.get('target_response_time', 200):
                return 1.0
            elif response_time <= target.metadata.get('max_response_time', 500):
                return 0.7
            else:
                return 0.3
                
        except Exception:
            return None
    
    async def _security_health_check(self, target: DeploymentTarget) -> Optional[float]:
        """Check security status"""
        try:
            # Check SSL certificate validity
            # Check for known vulnerabilities
            # Verify authentication is working
            
            # Simplified security score
            return 0.9  # Assume good security
            
        except Exception:
            return None
    
    async def _business_metrics_health_check(self, target: DeploymentTarget) -> Optional[float]:
        """Check business-critical metrics"""
        try:
            # Check conversion rates, user satisfaction, etc.
            # This would integrate with business analytics
            
            return 0.85  # Assume acceptable business metrics
            
        except Exception:
            return None


class GlobalDeploymentOrchestrator:
    """Main orchestrator for global multi-region deployments"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.validation = ValidationManager()
        self.security = SecurityManager()
        self.performance = PerformanceMonitor()
        self.metrics = MetricsCollector()
        
        self.traffic_router = TrafficRouter()
        self.auto_scaler = AutoScaler()
        self.health_monitor = DeploymentHealth()
        
        # Region configurations
        self.regions = self._initialize_regions()
        
        # Active deployments
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        
    def _initialize_regions(self) -> Dict[DeploymentRegion, RegionConfig]:
        """Initialize region configurations"""
        return {
            DeploymentRegion.US_EAST_1: RegionConfig(
                region=DeploymentRegion.US_EAST_1,
                compliance_regime=ComplianceRegime.CCPA,
                kubernetes_cluster="us-east-1-cluster",
                container_registry="us-east-1-registry",
                data_residency_required=False,
                min_instances=2,
                max_instances=50,
                target_cpu_utilization=70.0,
                target_latency_ms=100,
                allowed_deployment_hours=list(range(2, 8)),  # 2 AM - 8 AM UTC
                maintenance_window=(3, 5),
                disaster_recovery_region=DeploymentRegion.US_WEST_2
            ),
            
            DeploymentRegion.EU_WEST_1: RegionConfig(
                region=DeploymentRegion.EU_WEST_1,
                compliance_regime=ComplianceRegime.GDPR,
                kubernetes_cluster="eu-west-1-cluster",
                container_registry="eu-west-1-registry",
                data_residency_required=True,
                min_instances=2,
                max_instances=30,
                target_cpu_utilization=65.0,
                target_latency_ms=80,
                allowed_deployment_hours=list(range(1, 7)),
                maintenance_window=(2, 4),
                disaster_recovery_region=DeploymentRegion.EU_CENTRAL_1
            ),
            
            DeploymentRegion.AP_SOUTHEAST_1: RegionConfig(
                region=DeploymentRegion.AP_SOUTHEAST_1,
                compliance_regime=ComplianceRegime.PDPA,
                kubernetes_cluster="ap-southeast-1-cluster",
                container_registry="ap-southeast-1-registry",
                data_residency_required=True,
                min_instances=1,
                max_instances=20,
                target_cpu_utilization=75.0,
                target_latency_ms=120,
                allowed_deployment_hours=list(range(14, 20)),  # 2 PM - 8 PM UTC (Asia friendly)
                maintenance_window=(15, 17)
            )
        }
    
    async def orchestrate_global_deployment(self, 
                                          version: str,
                                          target_regions: List[DeploymentRegion],
                                          strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
                                          rollout_percentage: float = 100.0) -> Dict[str, DeploymentResult]:
        """Orchestrate deployment across multiple regions"""
        
        self.logger.info(f"Starting global deployment of version {version} "
                        f"to regions: {[r.value for r in target_regions]}")
        
        deployment_id = self._generate_deployment_id(version, target_regions)
        results = {}
        
        # Validate deployment readiness
        readiness_check = await self._validate_deployment_readiness(
            version, target_regions, strategy
        )
        
        if not readiness_check['ready']:
            self.logger.error(f"Deployment readiness check failed: {readiness_check['reason']}")
            raise ValueError(f"Deployment not ready: {readiness_check['reason']}")
        
        # Plan deployment order based on strategy
        deployment_order = self._plan_deployment_order(target_regions, strategy)
        
        try:
            for region in deployment_order:
                region_config = self.regions[region]
                
                # Check deployment window
                if not self._is_deployment_allowed(region_config):
                    self.logger.warning(f"Deployment to {region.value} outside allowed window, scheduling for later")
                    continue
                
                # Create deployment target
                target = DeploymentTarget(
                    region=region,
                    environment="production",
                    version=version,
                    instances=region_config.min_instances,
                    strategy=strategy,
                    rollback_threshold=0.05,  # 5% error rate threshold
                    health_check_url=f"https://{region.value}.api.company.com/health",
                    metadata={
                        'deployment_id': deployment_id,
                        'rollout_percentage': rollout_percentage,
                        'expected_response_time': region_config.target_latency_ms,
                        'target_response_time': region_config.target_latency_ms,
                        'max_response_time': region_config.target_latency_ms * 2
                    }
                )
                
                # Execute deployment
                result = await self._execute_region_deployment(target)
                results[region.value] = result
                
                # Check if deployment failed and should stop rollout
                if result.status == DeploymentStatus.FAILED:
                    self.logger.error(f"Deployment failed in {region.value}, stopping rollout")
                    
                    # Rollback previous deployments if strategy requires it
                    if strategy == DeploymentStrategy.BLUE_GREEN:
                        await self._rollback_deployments(list(results.values())[:-1])
                    
                    break
                
                # Wait between regions for rolling strategies
                if strategy == DeploymentStrategy.ROLLING and len(deployment_order) > 1:
                    await asyncio.sleep(30)  # 30 second delay between regions
        
        except Exception as e:
            self.logger.error(f"Global deployment failed: {e}")
            
            # Attempt to rollback all deployments
            await self._rollback_deployments(list(results.values()))
            raise
        
        # Validate overall deployment success
        overall_success = await self._validate_global_deployment(results)
        
        if overall_success:
            self.logger.info(f"Global deployment {deployment_id} completed successfully")
            
            # Update traffic routing
            await self._update_global_traffic_routing(results)
        else:
            self.logger.warning(f"Global deployment {deployment_id} completed with issues")
        
        return results
    
    async def _validate_deployment_readiness(self, 
                                           version: str,
                                           regions: List[DeploymentRegion],
                                           strategy: DeploymentStrategy) -> Dict[str, Any]:
        """Validate that deployment is ready to proceed"""
        
        # Check version exists in registries
        for region in regions:
            region_config = self.regions[region]
            
            # Simulate registry check
            registry_available = await self._check_container_registry(
                region_config.container_registry, version
            )
            
            if not registry_available:
                return {
                    'ready': False,
                    'reason': f"Version {version} not available in {region.value} registry"
                }
        
        # Check cluster health
        for region in regions:
            cluster_health = await self._check_cluster_health(region)
            
            if cluster_health < 0.8:
                return {
                    'ready': False,
                    'reason': f"Cluster {region.value} health too low: {cluster_health:.2f}"
                }
        
        # Check compliance requirements
        compliance_check = await self._validate_compliance_requirements(regions)
        if not compliance_check['compliant']:
            return {
                'ready': False,
                'reason': f"Compliance validation failed: {compliance_check['reason']}"
            }
        
        return {'ready': True, 'reason': 'All checks passed'}
    
    def _plan_deployment_order(self, 
                             regions: List[DeploymentRegion],
                             strategy: DeploymentStrategy) -> List[DeploymentRegion]:
        """Plan the order of regional deployments"""
        
        if strategy == DeploymentStrategy.BLUE_GREEN:
            # Deploy to all regions simultaneously
            return regions
        
        elif strategy == DeploymentStrategy.CANARY:
            # Start with lowest risk region
            risk_scores = {}
            for region in regions:
                config = self.regions[region]
                # Lower max instances = lower risk
                risk_scores[region] = config.max_instances
            
            return sorted(regions, key=lambda r: risk_scores[r])
        
        elif strategy == DeploymentStrategy.ROLLING:
            # Deploy in time zone order to minimize disruption
            timezone_order = {
                DeploymentRegion.AP_SOUTHEAST_1: 0,
                DeploymentRegion.AP_NORTHEAST_1: 1,
                DeploymentRegion.EU_CENTRAL_1: 2,
                DeploymentRegion.EU_WEST_1: 3,
                DeploymentRegion.US_EAST_1: 4,
                DeploymentRegion.US_WEST_2: 5
            }
            
            return sorted(regions, key=lambda r: timezone_order.get(r, 99))
        
        else:
            # Default: alphabetical order
            return sorted(regions, key=lambda r: r.value)
    
    def _is_deployment_allowed(self, region_config: RegionConfig) -> bool:
        """Check if deployment is allowed in the current time window"""
        current_hour = datetime.utcnow().hour
        return current_hour in region_config.allowed_deployment_hours
    
    async def _execute_region_deployment(self, target: DeploymentTarget) -> DeploymentResult:
        """Execute deployment to a specific region"""
        
        deployment_result = DeploymentResult(
            deployment_id=target.metadata['deployment_id'],
            target=target,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=datetime.now()
        )
        
        try:
            self.logger.info(f"Executing deployment to {target.region.value}")
            
            # Update active deployments
            result_key = f"{target.region.value}-{target.version}"
            self.active_deployments[result_key] = deployment_result
            
            if target.strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._execute_blue_green_deployment(target)
            elif target.strategy == DeploymentStrategy.CANARY:
                success = await self._execute_canary_deployment(target)
            elif target.strategy == DeploymentStrategy.ROLLING:
                success = await self._execute_rolling_deployment(target)
            else:
                success = await self._execute_recreate_deployment(target)
            
            if success:
                deployment_result.status = DeploymentStatus.COMPLETED
                
                # Collect metrics
                metrics = await self._collect_deployment_metrics(target)
                deployment_result.metrics = metrics
                
                # Auto-scale based on metrics
                await self._auto_scale_deployment(target, metrics)
                
            else:
                deployment_result.status = DeploymentStatus.FAILED
                deployment_result.error_message = "Deployment validation failed"
            
        except Exception as e:
            self.logger.error(f"Deployment to {target.region.value} failed: {e}")
            deployment_result.status = DeploymentStatus.FAILED
            deployment_result.error_message = str(e)
        
        finally:
            deployment_result.end_time = datetime.now()
            
            # Move to history
            self.deployment_history.append(deployment_result)
            if result_key in self.active_deployments:
                del self.active_deployments[result_key]
        
        return deployment_result
    
    async def _execute_blue_green_deployment(self, target: DeploymentTarget) -> bool:
        """Execute blue-green deployment strategy"""
        
        # Deploy to green environment
        green_success = await self._deploy_to_environment(target, "green")
        if not green_success:
            return False
        
        # Validate green environment
        health_score = await self.health_monitor.assess_deployment_health(target)
        if health_score < 0.8:
            self.logger.warning(f"Green environment health too low: {health_score:.2f}")
            return False
        
        # Switch traffic to green
        await self._switch_traffic(target, from_env="blue", to_env="green")
        
        # Monitor for rollback threshold
        await asyncio.sleep(60)  # Wait 1 minute
        
        post_switch_health = await self.health_monitor.assess_deployment_health(target)
        if post_switch_health < 0.7:
            # Rollback
            await self._switch_traffic(target, from_env="green", to_env="blue")
            return False
        
        # Cleanup old blue environment
        await self._cleanup_environment(target, "blue")
        
        return True
    
    async def _execute_canary_deployment(self, target: DeploymentTarget) -> bool:
        """Execute canary deployment strategy"""
        
        # Deploy canary with 10% traffic
        canary_success = await self._deploy_canary(target, traffic_percentage=10)
        if not canary_success:
            return False
        
        # Monitor canary for 5 minutes
        for i in range(5):
            await asyncio.sleep(60)
            
            health_score = await self.health_monitor.assess_deployment_health(target)
            if health_score < 0.8:
                await self._rollback_canary(target)
                return False
        
        # Gradually increase traffic
        for percentage in [25, 50, 75, 100]:
            await self._update_canary_traffic(target, percentage)
            await asyncio.sleep(30)
            
            health_score = await self.health_monitor.assess_deployment_health(target)
            if health_score < 0.8:
                await self._rollback_canary(target)
                return False
        
        # Promote canary to production
        await self._promote_canary(target)
        
        return True
    
    async def _execute_rolling_deployment(self, target: DeploymentTarget) -> bool:
        """Execute rolling deployment strategy"""
        
        total_instances = target.instances
        batch_size = max(1, total_instances // 4)  # 25% at a time
        
        for batch_start in range(0, total_instances, batch_size):
            batch_end = min(batch_start + batch_size, total_instances)
            
            # Update instances in this batch
            success = await self._update_instance_batch(
                target, batch_start, batch_end
            )
            
            if not success:
                return False
            
            # Wait for instances to be ready
            await asyncio.sleep(30)
            
            # Check health
            health_score = await self.health_monitor.assess_deployment_health(target)
            if health_score < 0.8:
                return False
        
        return True
    
    async def _execute_recreate_deployment(self, target: DeploymentTarget) -> bool:
        """Execute recreate deployment strategy (downtime)"""
        
        # Stop all instances
        await self._stop_all_instances(target)
        
        # Deploy new version
        success = await self._deploy_to_environment(target, "production")
        
        return success
    
    async def _collect_deployment_metrics(self, target: DeploymentTarget) -> DeploymentMetrics:
        """Collect metrics for a deployed service"""
        
        # Simulate metrics collection
        return DeploymentMetrics(
            success_rate=0.995,
            avg_response_time_ms=85.0,
            error_rate=0.005,
            cpu_utilization=45.0,
            memory_utilization=60.0,
            throughput_rps=150.0,
            active_connections=300,
            health_score=0.92
        )
    
    async def _auto_scale_deployment(self, target: DeploymentTarget, 
                                   metrics: DeploymentMetrics):
        """Auto-scale deployment based on metrics"""
        
        region_config = self.regions[target.region]
        
        optimal_instances = await self.auto_scaler.calculate_optimal_instances(
            metrics, target.instances, region_config
        )
        
        if optimal_instances != target.instances:
            self.logger.info(f"Auto-scaling {target.region.value} from "
                           f"{target.instances} to {optimal_instances} instances")
            
            await self._scale_instances(target, optimal_instances)
            target.instances = optimal_instances
    
    # Placeholder methods for actual deployment operations
    async def _check_container_registry(self, registry: str, version: str) -> bool:
        """Check if version exists in container registry"""
        # In real implementation, this would check Docker registry
        return True
    
    async def _check_cluster_health(self, region: DeploymentRegion) -> float:
        """Check Kubernetes cluster health"""
        # In real implementation, this would check cluster metrics
        return 0.95
    
    async def _validate_compliance_requirements(self, regions: List[DeploymentRegion]) -> Dict[str, Any]:
        """Validate compliance requirements for regions"""
        # In real implementation, this would check data residency, etc.
        return {'compliant': True, 'reason': 'All requirements met'}
    
    async def _deploy_to_environment(self, target: DeploymentTarget, environment: str) -> bool:
        """Deploy to specific environment"""
        # Simulate deployment
        await asyncio.sleep(2)
        return True
    
    async def _switch_traffic(self, target: DeploymentTarget, from_env: str, to_env: str):
        """Switch traffic between environments"""
        await asyncio.sleep(1)
    
    async def _cleanup_environment(self, target: DeploymentTarget, environment: str):
        """Cleanup old environment"""
        await asyncio.sleep(1)
    
    async def _deploy_canary(self, target: DeploymentTarget, traffic_percentage: int) -> bool:
        """Deploy canary version"""
        await asyncio.sleep(2)
        return True
    
    async def _rollback_canary(self, target: DeploymentTarget):
        """Rollback canary deployment"""
        await asyncio.sleep(1)
    
    async def _update_canary_traffic(self, target: DeploymentTarget, percentage: int):
        """Update canary traffic percentage"""
        await asyncio.sleep(1)
    
    async def _promote_canary(self, target: DeploymentTarget):
        """Promote canary to production"""
        await asyncio.sleep(1)
    
    async def _update_instance_batch(self, target: DeploymentTarget, 
                                   start: int, end: int) -> bool:
        """Update a batch of instances"""
        await asyncio.sleep(1)
        return True
    
    async def _stop_all_instances(self, target: DeploymentTarget):
        """Stop all instances"""
        await asyncio.sleep(1)
    
    async def _scale_instances(self, target: DeploymentTarget, new_count: int):
        """Scale instances to new count"""
        await asyncio.sleep(1)
    
    def _generate_deployment_id(self, version: str, regions: List[DeploymentRegion]) -> str:
        """Generate unique deployment ID"""
        region_str = "-".join(sorted(r.value for r in regions))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        hash_input = f"{version}-{region_str}-{timestamp}"
        hash_short = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"deploy-{version}-{hash_short}"
    
    async def _rollback_deployments(self, deployments: List[DeploymentResult]):
        """Rollback multiple deployments"""
        for deployment in deployments:
            if deployment.status == DeploymentStatus.COMPLETED:
                self.logger.info(f"Rolling back deployment in {deployment.target.region.value}")
                # Implementation would rollback to previous version
                deployment.status = DeploymentStatus.ROLLED_BACK
    
    async def _validate_global_deployment(self, results: Dict[str, DeploymentResult]) -> bool:
        """Validate overall global deployment success"""
        successful_deployments = [
            r for r in results.values() 
            if r.status == DeploymentStatus.COMPLETED
        ]
        
        # Require at least 75% of regions to succeed
        success_rate = len(successful_deployments) / len(results)
        return success_rate >= 0.75
    
    async def _update_global_traffic_routing(self, results: Dict[str, DeploymentResult]):
        """Update global traffic routing after successful deployment"""
        
        # Calculate new traffic distribution
        healthy_regions = [
            DeploymentRegion(r) for r, result in results.items()
            if result.status == DeploymentStatus.COMPLETED
        ]
        
        if healthy_regions:
            # Update routing rules
            self.traffic_router.update_routing_rules({
                'active_regions': [r.value for r in healthy_regions],
                'last_updated': datetime.now().isoformat()
            })
    
    async def continuous_monitoring_loop(self, interval_seconds: int = 300):
        """Continuous monitoring and optimization loop"""
        
        self.logger.info(f"Starting continuous monitoring loop with {interval_seconds}s interval")
        
        while True:
            try:
                # Monitor all active deployments
                for deployment in self.active_deployments.values():
                    health = await self.health_monitor.assess_deployment_health(
                        deployment.target
                    )
                    
                    if health < 0.5:
                        self.logger.warning(
                            f"Low health detected in {deployment.target.region.value}: {health:.2f}"
                        )
                        
                        # Trigger auto-healing or alerts
                        await self._trigger_auto_healing(deployment)
                
                # Adapt auto-scaling thresholds
                self.auto_scaler.adapt_thresholds()
                
                # Optimize traffic routing
                await self._optimize_traffic_routing()
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _trigger_auto_healing(self, deployment: DeploymentResult):
        """Trigger auto-healing for unhealthy deployment"""
        
        self.logger.info(f"Triggering auto-healing for {deployment.target.region.value}")
        
        # Restart unhealthy instances
        # Scale up if needed
        # Route traffic away temporarily
        
        # For now, just log the action
        deployment.status = DeploymentStatus.IN_PROGRESS
    
    async def _optimize_traffic_routing(self):
        """Optimize traffic routing based on current conditions"""
        
        # Collect current metrics from all regions
        region_health = {}
        
        for region in self.regions.keys():
            # Get current health score
            # This would integrate with actual monitoring
            region_health[region] = 0.9  # Placeholder
        
        # Update routing based on health and latency
        for user_location in ['us', 'eu', 'asia']:
            optimal_routing = self.traffic_router.calculate_optimal_routing(
                user_location, list(self.regions.keys()), region_health
            )
            
            # Apply routing updates
            # This would update load balancer configuration
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a specific deployment"""
        
        # Check active deployments
        for deployment in self.active_deployments.values():
            if deployment.deployment_id == deployment_id:
                return deployment
        
        # Check history
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        
        return None
    
    def export_deployment_report(self) -> str:
        """Export comprehensive deployment report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'active_deployments': len(self.active_deployments),
            'total_deployments': len(self.deployment_history),
            'regions': {
                region.value: {
                    'min_instances': config.min_instances,
                    'max_instances': config.max_instances,
                    'compliance_regime': config.compliance_regime.value,
                    'data_residency_required': config.data_residency_required
                }
                for region, config in self.regions.items()
            },
            'recent_deployments': [
                {
                    'deployment_id': d.deployment_id,
                    'region': d.target.region.value,
                    'version': d.target.version,
                    'status': d.status.value,
                    'start_time': d.start_time.isoformat(),
                    'duration_seconds': (
                        (d.end_time - d.start_time).total_seconds()
                        if d.end_time else None
                    )
                }
                for d in self.deployment_history[-10:]
            ]
        }
        
        return json.dumps(report, indent=2)


# Factory function
def create_global_deployment_orchestrator(config_path: Optional[str] = None) -> GlobalDeploymentOrchestrator:
    """Create and configure global deployment orchestrator"""
    config = ConfigManager(config_path)
    return GlobalDeploymentOrchestrator(config)


# CLI interface
if __name__ == "__main__":
    import sys
    
    async def main():
        orchestrator = create_global_deployment_orchestrator()
        
        if len(sys.argv) < 3:
            print("Usage: python global_deployment_orchestrator.py <version> <regions>")
            print("Example: python global_deployment_orchestrator.py v1.2.3 us-east-1,eu-west-1")
            return
        
        version = sys.argv[1]
        region_names = sys.argv[2].split(',')
        
        try:
            regions = [DeploymentRegion(name.strip()) for name in region_names]
        except ValueError as e:
            print(f"Invalid region: {e}")
            return
        
        # Execute deployment
        results = await orchestrator.orchestrate_global_deployment(
            version=version,
            target_regions=regions,
            strategy=DeploymentStrategy.BLUE_GREEN
        )
        
        # Print results
        print(f"\n=== GLOBAL DEPLOYMENT RESULTS ===")
        for region, result in results.items():
            status_emoji = "✅" if result.status == DeploymentStatus.COMPLETED else "❌"
            print(f"{status_emoji} {region}: {result.status.value}")
            
            if result.metrics:
                print(f"   Health Score: {result.metrics.health_score:.2f}")
                print(f"   Response Time: {result.metrics.avg_response_time_ms:.1f}ms")
        
        # Export report
        report = orchestrator.export_deployment_report()
        with open("deployment_report.json", "w") as f:
            f.write(report)
        
        print(f"\nDetailed report saved to deployment_report.json")
    
    asyncio.run(main())