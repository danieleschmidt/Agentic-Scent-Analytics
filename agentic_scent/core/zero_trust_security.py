#!/usr/bin/env python3
"""
Zero-Trust Security Framework - Advanced threat detection and security orchestration
Part of Agentic Scent Analytics Platform

This module implements a comprehensive zero-trust security framework with AI-powered
threat detection, behavioral analysis, adaptive authentication, and automated incident
response capabilities for industrial IoT and manufacturing environments.
"""

import asyncio
import json
import time
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from pathlib import Path
import logging
import ipaddress
import re
from urllib.parse import urlparse
import base64

import numpy as np
from collections import deque, defaultdict

from .config import ConfigManager
from .validation import AdvancedDataValidator
from .security import SecurityManager
from .metrics import PrometheusMetrics


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    MALWARE_DETECTION = "malware_detection"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    COMMAND_INJECTION = "command_injection"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    BRUTE_FORCE = "brute_force"
    DDoS_ATTACK = "ddos_attack"
    INSIDER_THREAT = "insider_threat"


class ResponseAction(Enum):
    """Automated response actions"""
    LOG_ONLY = "log_only"
    ALERT = "alert"
    RATE_LIMIT = "rate_limit"
    BLOCK_IP = "block_ip"
    QUARANTINE_USER = "quarantine_user"
    TERMINATE_SESSION = "terminate_session"
    ISOLATE_DEVICE = "isolate_device"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class SecurityContext:
    """Security context for a user/device/session"""
    user_id: str
    device_id: str
    session_id: str
    ip_address: str
    user_agent: str
    location: Optional[str] = None
    trust_score: float = 0.5
    risk_factors: List[str] = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)
    last_activity: datetime = field(default_factory=datetime.now)
    authentication_methods: List[str] = field(default_factory=list)
    device_fingerprint: str = ""
    network_segment: str = "unknown"


@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: str
    target_resource: str
    user_context: Optional[SecurityContext]
    description: str
    indicators: Dict[str, Any] = field(default_factory=dict)
    response_actions: List[ResponseAction] = field(default_factory=list)
    mitigated: bool = False
    false_positive: bool = False
    impact_score: float = 0.0


@dataclass
class BehaviorPattern:
    """User/device behavior pattern"""
    entity_id: str
    entity_type: str  # user, device, service
    pattern_type: str  # login_times, access_patterns, data_usage
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    anomaly_score: float
    confidence: float
    last_updated: datetime


class ThreatIntelligence:
    """Threat intelligence and IOC management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ioc_database: Dict[str, Dict] = {}
        self.threat_feeds: List[str] = []
        self.reputation_scores: Dict[str, float] = {}
        self.attack_patterns: Dict[str, List[str]] = {}
        self._initialize_threat_intelligence()
        
    def _initialize_threat_intelligence(self):
        """Initialize threat intelligence database"""
        
        # Known malicious IP ranges (simplified)
        self.malicious_ip_ranges = [
            "10.0.0.0/8",    # Example: internal network scanning
            "192.168.0.0/16", # Example: local network threats
        ]
        
        # Known attack patterns
        self.attack_patterns = {
            'sql_injection': [
                r"union\s+select",
                r"drop\s+table",
                r"1'\s*or\s*'1'\s*=\s*'1",
                r"admin'\s*--",
                r"exec\s*\(",
            ],
            'xss': [
                r"<script[^>]*>",
                r"javascript:",
                r"on\w+\s*=",
                r"eval\s*\(",
                r"document\.cookie",
            ],
            'command_injection': [
                r";\s*cat\s+",
                r";\s*ls\s+",
                r";\s*wget\s+",
                r";\s*curl\s+",
                r"&&\s*echo",
            ],
            'directory_traversal': [
                r"\.\.\/",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e\\",
            ]
        }
        
        # Suspicious user agents
        self.suspicious_user_agents = [
            r"sqlmap",
            r"nikto",
            r"nmap",
            r"masscan",
            r"burp",
            r"w3af",
        ]
        
        self.logger.info("Initialized threat intelligence database")
    
    def check_ip_reputation(self, ip_address: str) -> Tuple[float, List[str]]:
        """Check IP address reputation"""
        
        reputation_score = 0.5  # Neutral
        risk_factors = []
        
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check against known malicious ranges
            for range_str in self.malicious_ip_ranges:
                network = ipaddress.ip_network(range_str, strict=False)
                if ip in network:
                    reputation_score = 0.1
                    risk_factors.append(f"ip_in_malicious_range_{range_str}")
                    break
            
            # Check if IP is in private ranges (could be internal threat)
            if ip.is_private:
                risk_factors.append("private_ip")
            
            # Check reputation cache
            if ip_address in self.reputation_scores:
                cached_score = self.reputation_scores[ip_address]
                reputation_score = min(reputation_score, cached_score)
            
        except ValueError:
            reputation_score = 0.3
            risk_factors.append("invalid_ip_format")
        
        return reputation_score, risk_factors
    
    def analyze_request_for_attacks(self, request_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Analyze HTTP request for attack patterns"""
        
        detected_attacks = []
        
        # Get all request data as strings for analysis
        analysis_strings = []
        
        if 'url' in request_data:
            analysis_strings.append(request_data['url'])
        if 'headers' in request_data:
            analysis_strings.extend(str(v) for v in request_data['headers'].values())
        if 'body' in request_data:
            analysis_strings.append(str(request_data['body']))
        if 'parameters' in request_data:
            analysis_strings.extend(str(v) for v in request_data['parameters'].values())
        
        # Check each string against attack patterns
        for data_string in analysis_strings:
            if not isinstance(data_string, str):
                continue
                
            data_lower = data_string.lower()
            
            for attack_type, patterns in self.attack_patterns.items():
                max_confidence = 0.0
                
                for pattern in patterns:
                    if re.search(pattern, data_lower, re.IGNORECASE):
                        # Calculate confidence based on pattern specificity
                        confidence = min(0.95, 0.6 + len(pattern) / 100.0)
                        max_confidence = max(max_confidence, confidence)
                
                if max_confidence > 0.0:
                    detected_attacks.append((attack_type, max_confidence))
        
        # Check user agent
        if 'user_agent' in request_data:
            user_agent = request_data['user_agent'].lower()
            for suspicious_pattern in self.suspicious_user_agents:
                if re.search(suspicious_pattern, user_agent, re.IGNORECASE):
                    detected_attacks.append(('suspicious_tool', 0.8))
                    break
        
        return detected_attacks
    
    def update_reputation(self, ip_address: str, reputation_delta: float):
        """Update IP reputation based on observed behavior"""
        
        current_score = self.reputation_scores.get(ip_address, 0.5)
        new_score = np.clip(current_score + reputation_delta, 0.0, 1.0)
        self.reputation_scores[ip_address] = new_score
        
        # Decay old scores to prevent permanent blacklisting
        if len(self.reputation_scores) % 1000 == 0:
            self._decay_reputation_scores()
    
    def _decay_reputation_scores(self):
        """Apply decay to reputation scores over time"""
        
        decay_factor = 0.95
        for ip in list(self.reputation_scores.keys()):
            current_score = self.reputation_scores[ip]
            
            # Move score towards neutral (0.5)
            if current_score < 0.5:
                decayed_score = current_score + (0.5 - current_score) * (1 - decay_factor)
            else:
                decayed_score = current_score - (current_score - 0.5) * (1 - decay_factor)
            
            self.reputation_scores[ip] = decayed_score
            
            # Remove scores that are very close to neutral
            if abs(decayed_score - 0.5) < 0.01:
                del self.reputation_scores[ip]


class BehaviorAnalyzer:
    """AI-powered behavioral analysis for anomaly detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.behavior_baselines: Dict[str, BehaviorPattern] = {}
        self.activity_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_weights = {
            'temporal_deviation': 0.25,
            'frequency_anomaly': 0.20,
            'location_deviation': 0.15,
            'access_pattern_change': 0.20,
            'volume_anomaly': 0.10,
            'failure_rate_spike': 0.10
        }
        
    def analyze_user_behavior(self, context: SecurityContext, 
                            activity_data: Dict[str, Any]) -> float:
        """Analyze user behavior for anomalies"""
        
        entity_id = f"user_{context.user_id}"
        
        # Get or create behavior baseline
        if entity_id not in self.behavior_baselines:
            self._initialize_baseline(entity_id, context, activity_data)
            return 0.1  # Low anomaly score for new users
        
        baseline = self.behavior_baselines[entity_id]
        
        # Calculate various anomaly scores
        anomaly_scores = {}
        
        # Temporal patterns (login times)
        anomaly_scores['temporal'] = self._analyze_temporal_patterns(
            context, baseline, activity_data
        )
        
        # Access frequency
        anomaly_scores['frequency'] = self._analyze_frequency_patterns(
            entity_id, baseline, activity_data
        )
        
        # Location deviation
        anomaly_scores['location'] = self._analyze_location_patterns(
            context, baseline
        )
        
        # Access patterns (resources accessed)
        anomaly_scores['access_pattern'] = self._analyze_access_patterns(
            entity_id, baseline, activity_data
        )
        
        # Data volume anomalies
        anomaly_scores['volume'] = self._analyze_volume_patterns(
            baseline, activity_data
        )
        
        # Failure rate spikes
        anomaly_scores['failure_rate'] = self._analyze_failure_patterns(
            entity_id, activity_data
        )
        
        # Calculate weighted anomaly score
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in anomaly_scores.items():
            weight_key = f"{metric}_deviation"
            if weight_key in self.model_weights:
                weight = self.model_weights[weight_key]
                total_score += score * weight
                total_weight += weight
        
        overall_anomaly = total_score / total_weight if total_weight > 0 else 0.0
        
        # Update baseline with new data
        self._update_baseline(entity_id, context, activity_data, overall_anomaly)
        
        self.logger.debug(f"Behavior analysis for {entity_id}: "
                         f"anomaly_score={overall_anomaly:.3f}, "
                         f"scores={anomaly_scores}")
        
        return overall_anomaly
    
    def _initialize_baseline(self, entity_id: str, context: SecurityContext,
                           activity_data: Dict[str, Any]):
        """Initialize behavior baseline for new entity"""
        
        baseline = BehaviorPattern(
            entity_id=entity_id,
            entity_type="user",
            pattern_type="activity_baseline",
            baseline_metrics={
                'avg_login_hour': datetime.now().hour,
                'login_frequency_per_day': 1.0,
                'avg_session_duration': 60.0,  # minutes
                'common_resources': [],
                'avg_data_volume': 1.0,  # MB
                'typical_locations': [context.ip_address],
                'failure_rate': 0.0
            },
            current_metrics={},
            anomaly_score=0.0,
            confidence=0.1,  # Low confidence initially
            last_updated=datetime.now()
        )
        
        self.behavior_baselines[entity_id] = baseline
        self.logger.info(f"Initialized behavior baseline for {entity_id}")
    
    def _analyze_temporal_patterns(self, context: SecurityContext,
                                 baseline: BehaviorPattern,
                                 activity_data: Dict[str, Any]) -> float:
        """Analyze temporal access patterns"""
        
        current_hour = datetime.now().hour
        baseline_hour = baseline.baseline_metrics.get('avg_login_hour', 12)
        
        # Calculate hour difference (accounting for 24-hour wrap)
        hour_diff = min(abs(current_hour - baseline_hour),
                       24 - abs(current_hour - baseline_hour))
        
        # Normalize to 0-1 scale (12 hours difference = max anomaly)
        temporal_anomaly = min(1.0, hour_diff / 12.0)
        
        return temporal_anomaly
    
    def _analyze_frequency_patterns(self, entity_id: str,
                                  baseline: BehaviorPattern,
                                  activity_data: Dict[str, Any]) -> float:
        """Analyze access frequency patterns"""
        
        # Count recent activities
        recent_activities = list(self.activity_history[entity_id])
        
        if len(recent_activities) < 5:
            return 0.1  # Not enough data
        
        # Calculate current frequency (activities per hour)
        recent_timestamps = [a.get('timestamp', datetime.now()) for a in recent_activities[-10:]]
        time_span = (max(recent_timestamps) - min(recent_timestamps)).total_seconds() / 3600.0
        
        if time_span > 0:
            current_frequency = len(recent_timestamps) / time_span
        else:
            current_frequency = 0.0
        
        baseline_frequency = baseline.baseline_metrics.get('login_frequency_per_day', 1.0) / 24.0
        
        # Calculate relative change
        if baseline_frequency > 0:
            frequency_ratio = current_frequency / baseline_frequency
            # Anomaly if frequency is >3x or <0.3x normal
            if frequency_ratio > 3.0:
                frequency_anomaly = min(1.0, (frequency_ratio - 3.0) / 7.0)
            elif frequency_ratio < 0.3:
                frequency_anomaly = min(1.0, (0.3 - frequency_ratio) / 0.3)
            else:
                frequency_anomaly = 0.0
        else:
            frequency_anomaly = 0.5 if current_frequency > 0 else 0.0
        
        return frequency_anomaly
    
    def _analyze_location_patterns(self, context: SecurityContext,
                                 baseline: BehaviorPattern) -> float:
        """Analyze location/IP address patterns"""
        
        current_ip = context.ip_address
        typical_locations = baseline.baseline_metrics.get('typical_locations', [])
        
        if not typical_locations:
            return 0.3  # Medium anomaly for unknown location
        
        # Check if current IP is in typical locations
        if current_ip in typical_locations:
            return 0.0  # No anomaly
        
        # Check if IP is in same subnet as typical locations
        try:
            current_ip_obj = ipaddress.ip_address(current_ip)
            for typical_ip in typical_locations:
                try:
                    typical_ip_obj = ipaddress.ip_address(typical_ip)
                    # Check if in same /24 subnet
                    if (current_ip_obj.packed[:3] == typical_ip_obj.packed[:3]):
                        return 0.2  # Low anomaly for same subnet
                except ValueError:
                    continue
        except ValueError:
            return 0.8  # High anomaly for invalid IP
        
        return 0.7  # High anomaly for completely different location
    
    def _analyze_access_patterns(self, entity_id: str,
                               baseline: BehaviorPattern,
                               activity_data: Dict[str, Any]) -> float:
        """Analyze resource access patterns"""
        
        current_resources = activity_data.get('resources_accessed', [])
        common_resources = baseline.baseline_metrics.get('common_resources', [])
        
        if not common_resources:
            return 0.2  # Low anomaly for new users
        
        if not current_resources:
            return 0.1  # Low anomaly for no access
        
        # Calculate overlap between current and typical resources
        common_set = set(common_resources)
        current_set = set(current_resources)
        
        if len(common_set) == 0:
            return 0.5
        
        overlap = len(common_set & current_set) / len(common_set)
        
        # High anomaly if accessing completely different resources
        access_anomaly = 1.0 - overlap
        
        return access_anomaly
    
    def _analyze_volume_patterns(self, baseline: BehaviorPattern,
                               activity_data: Dict[str, Any]) -> float:
        """Analyze data volume patterns"""
        
        current_volume = activity_data.get('data_volume_mb', 0.0)
        baseline_volume = baseline.baseline_metrics.get('avg_data_volume', 1.0)
        
        if baseline_volume <= 0:
            return 0.2 if current_volume > 100 else 0.0
        
        volume_ratio = current_volume / baseline_volume
        
        # Anomaly if volume is >10x or <0.1x normal
        if volume_ratio > 10.0:
            return min(1.0, (volume_ratio - 10.0) / 40.0)
        elif volume_ratio < 0.1 and current_volume > 0:
            return min(1.0, (0.1 - volume_ratio) / 0.1)
        else:
            return 0.0
    
    def _analyze_failure_patterns(self, entity_id: str,
                                activity_data: Dict[str, Any]) -> float:
        """Analyze authentication/authorization failure patterns"""
        
        recent_activities = list(self.activity_history[entity_id])
        
        if len(recent_activities) < 5:
            return 0.0
        
        # Count failures in recent activities
        recent_failures = [a for a in recent_activities[-20:] 
                          if a.get('status') in ['failed', 'denied', 'error']]
        
        failure_rate = len(recent_failures) / len(recent_activities[-20:])
        
        # High anomaly if failure rate > 20%
        if failure_rate > 0.2:
            return min(1.0, (failure_rate - 0.2) / 0.3)
        else:
            return 0.0
    
    def _update_baseline(self, entity_id: str, context: SecurityContext,
                       activity_data: Dict[str, Any], anomaly_score: float):
        """Update behavior baseline with new data"""
        
        baseline = self.behavior_baselines[entity_id]
        
        # Only update baseline if anomaly score is low (normal behavior)
        if anomaly_score < 0.3:
            alpha = 0.1  # Learning rate
            
            # Update temporal patterns
            current_hour = datetime.now().hour
            baseline.baseline_metrics['avg_login_hour'] = (
                baseline.baseline_metrics['avg_login_hour'] * (1 - alpha) +
                current_hour * alpha
            )
            
            # Update location patterns
            if context.ip_address not in baseline.baseline_metrics['typical_locations']:
                baseline.baseline_metrics['typical_locations'].append(context.ip_address)
                # Keep only last 10 locations
                if len(baseline.baseline_metrics['typical_locations']) > 10:
                    baseline.baseline_metrics['typical_locations'] = \
                        baseline.baseline_metrics['typical_locations'][-10:]
            
            # Update resource patterns
            current_resources = activity_data.get('resources_accessed', [])
            for resource in current_resources:
                if resource not in baseline.baseline_metrics['common_resources']:
                    baseline.baseline_metrics['common_resources'].append(resource)
                    # Keep only last 20 resources
                    if len(baseline.baseline_metrics['common_resources']) > 20:
                        baseline.baseline_metrics['common_resources'] = \
                            baseline.baseline_metrics['common_resources'][-20:]
        
        # Always update confidence (increases with more data)
        baseline.confidence = min(0.95, baseline.confidence + 0.01)
        baseline.last_updated = datetime.now()
        
        # Record activity in history
        activity_record = {
            'timestamp': datetime.now(),
            'context': context,
            'activity_data': activity_data,
            'anomaly_score': anomaly_score,
            'status': activity_data.get('status', 'success')
        }
        
        self.activity_history[entity_id].append(activity_record)


class ZeroTrustSecurityFramework:
    """Main zero-trust security framework coordinator"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.validation = AdvancedDataValidator()
        self.security_manager = SecurityManager()
        self.metrics = PrometheusMetrics()
        
        self.threat_intel = ThreatIntelligence()
        self.behavior_analyzer = BehaviorAnalyzer()
        
        # Security state
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.security_incidents: List[SecurityIncident] = []
        self.blocked_ips: Set[str] = set()
        self.quarantined_users: Set[str] = set()
        
        # Configuration
        self.trust_thresholds = {
            'min_trust_for_access': 0.3,
            'anomaly_threshold': 0.7,
            'critical_anomaly_threshold': 0.9,
            'auto_response_threshold': 0.8,
            'ip_block_threshold': 0.1,
            'user_quarantine_threshold': 0.2
        }
        
        # Response mappings
        self.response_mappings = {
            SecurityEvent.AUTHENTICATION_FAILURE: [ResponseAction.LOG_ONLY, ResponseAction.RATE_LIMIT],
            SecurityEvent.AUTHORIZATION_VIOLATION: [ResponseAction.ALERT, ResponseAction.TERMINATE_SESSION],
            SecurityEvent.SUSPICIOUS_ACTIVITY: [ResponseAction.ALERT, ResponseAction.RATE_LIMIT],
            SecurityEvent.ANOMALOUS_BEHAVIOR: [ResponseAction.ALERT],
            SecurityEvent.MALWARE_DETECTION: [ResponseAction.QUARANTINE_USER, ResponseAction.ISOLATE_DEVICE],
            SecurityEvent.DATA_EXFILTRATION: [ResponseAction.BLOCK_IP, ResponseAction.EMERGENCY_SHUTDOWN],
            SecurityEvent.SQL_INJECTION: [ResponseAction.BLOCK_IP, ResponseAction.ALERT],
            SecurityEvent.BRUTE_FORCE: [ResponseAction.BLOCK_IP, ResponseAction.RATE_LIMIT],
            SecurityEvent.DDoS_ATTACK: [ResponseAction.BLOCK_IP, ResponseAction.RATE_LIMIT],
            SecurityEvent.INSIDER_THREAT: [ResponseAction.QUARANTINE_USER, ResponseAction.ALERT]
        }
        
        # Monitoring state
        self._monitoring_active = False
        self.request_queue = asyncio.Queue()
        
    async def start_zero_trust_monitoring(self):
        """Start zero-trust security monitoring"""
        
        self.logger.info("Starting zero-trust security monitoring")
        self._monitoring_active = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._security_monitoring_loop()),
            asyncio.create_task(self._incident_response_loop()),
            asyncio.create_task(self._threat_intelligence_update_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in zero-trust monitoring: {e}")
        finally:
            self._monitoring_active = False
    
    async def evaluate_access_request(self, context: SecurityContext,
                                    resource: str, action: str,
                                    request_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, List[str]]:
        """Evaluate access request using zero-trust principles"""
        
        # Initial trust calculation
        trust_score = await self._calculate_trust_score(context, request_data or {})
        risk_factors = []
        
        # Check if user/IP is blocked
        if context.user_id in self.quarantined_users:
            return False, 0.0, ["user_quarantined"]
        
        if context.ip_address in self.blocked_ips:
            return False, 0.0, ["ip_blocked"]
        
        # Threat intelligence checks
        ip_reputation, ip_risk_factors = self.threat_intel.check_ip_reputation(context.ip_address)
        trust_score *= ip_reputation
        risk_factors.extend(ip_risk_factors)
        
        # Request analysis for attacks
        if request_data:
            detected_attacks = self.threat_intel.analyze_request_for_attacks(request_data)
            for attack_type, confidence in detected_attacks:
                trust_score *= (1.0 - confidence * 0.5)  # Reduce trust based on attack confidence
                risk_factors.append(f"potential_{attack_type}")
                
                # Generate security incident for high-confidence attacks
                if confidence > 0.8:
                    await self._generate_security_incident(
                        SecurityEvent.SUSPICIOUS_ACTIVITY,
                        context,
                        f"Detected {attack_type} with confidence {confidence:.2f}",
                        {"attack_type": attack_type, "confidence": confidence}
                    )
        
        # Behavioral analysis
        activity_data = {
            'resource': resource,
            'action': action,
            'timestamp': datetime.now(),
            'resources_accessed': [resource],
            'data_volume_mb': request_data.get('content_length', 0) / 1024 / 1024 if request_data else 0,
            'status': 'pending'
        }
        
        anomaly_score = self.behavior_analyzer.analyze_user_behavior(context, activity_data)
        
        if anomaly_score > self.trust_thresholds['anomaly_threshold']:
            trust_score *= (1.0 - anomaly_score * 0.3)
            risk_factors.append(f"behavioral_anomaly_{anomaly_score:.2f}")
            
            # Generate incident for high anomaly
            if anomaly_score > self.trust_thresholds['critical_anomaly_threshold']:
                await self._generate_security_incident(
                    SecurityEvent.ANOMALOUS_BEHAVIOR,
                    context,
                    f"High behavioral anomaly score: {anomaly_score:.2f}",
                    {"anomaly_score": anomaly_score, "activity_data": activity_data}
                )
        
        # Resource-specific access control
        resource_trust_modifier = await self._evaluate_resource_access(resource, action, context)
        trust_score *= resource_trust_modifier
        
        # Make access decision
        access_granted = trust_score >= self.trust_thresholds['min_trust_for_access']
        
        # Update context trust score
        context.trust_score = trust_score
        context.risk_factors = risk_factors
        context.last_activity = datetime.now()
        
        # Log access decision
        self.logger.info(f"Access decision: user={context.user_id}, "
                        f"resource={resource}, action={action}, "
                        f"granted={access_granted}, trust={trust_score:.3f}")
        
        # Update session
        if context.session_id:
            self.active_sessions[context.session_id] = context
        
        return access_granted, trust_score, risk_factors
    
    async def _calculate_trust_score(self, context: SecurityContext,
                                   request_data: Dict[str, Any]) -> float:
        """Calculate base trust score for context"""
        
        trust_score = 0.5  # Base neutral trust
        
        # Factor in authentication methods
        auth_methods = context.authentication_methods
        if 'mfa' in auth_methods:
            trust_score += 0.2
        if 'certificate' in auth_methods:
            trust_score += 0.15
        if 'biometric' in auth_methods:
            trust_score += 0.1
        
        # Device trust
        if context.device_fingerprint:
            # Check if device is known and trusted
            device_trust = await self._get_device_trust(context.device_fingerprint)
            trust_score += device_trust * 0.2
        
        # Network segment trust
        network_trust = await self._get_network_trust(context.network_segment)
        trust_score += network_trust * 0.1
        
        # Time-based factors
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            trust_score += 0.05
        elif 22 <= current_hour or current_hour <= 6:  # Night hours
            trust_score -= 0.1
        
        # Recent activity trust
        if context.last_activity:
            time_since_activity = (datetime.now() - context.last_activity).total_seconds()
            if time_since_activity < 300:  # 5 minutes
                trust_score += 0.05
            elif time_since_activity > 3600:  # 1 hour
                trust_score -= 0.1
        
        return np.clip(trust_score, 0.0, 1.0)
    
    async def _get_device_trust(self, device_fingerprint: str) -> float:
        """Get device trust score"""
        # In real implementation, this would check device reputation database
        if device_fingerprint:
            # Hash-based simple trust calculation
            hash_val = int(hashlib.md5(device_fingerprint.encode()).hexdigest()[:8], 16)
            return 0.3 + (hash_val % 100) / 200.0  # 0.3 to 0.8 range
        return 0.0
    
    async def _get_network_trust(self, network_segment: str) -> float:
        """Get network segment trust score"""
        network_trust_map = {
            'corporate': 0.8,
            'datacenter': 0.9,
            'production': 0.7,
            'guest': 0.2,
            'public': 0.1,
            'unknown': 0.3
        }
        return network_trust_map.get(network_segment, 0.3)
    
    async def _evaluate_resource_access(self, resource: str, action: str,
                                      context: SecurityContext) -> float:
        """Evaluate resource-specific access trust modifier"""
        
        # Check if user has required permissions
        required_permission = f"{resource}:{action}"
        if required_permission not in context.permissions:
            return 0.1  # Very low trust for unauthorized access
        
        # Resource sensitivity levels
        if 'admin' in resource.lower() or 'config' in resource.lower():
            return 0.7  # Higher scrutiny for admin resources
        elif 'user' in resource.lower() or 'profile' in resource.lower():
            return 1.0  # Normal trust for user resources
        elif 'public' in resource.lower():
            return 1.1  # Slightly higher trust for public resources
        else:
            return 0.9  # Default moderate trust
    
    async def _generate_security_incident(self, event_type: SecurityEvent,
                                        context: SecurityContext,
                                        description: str,
                                        indicators: Dict[str, Any]):
        """Generate and process security incident"""
        
        # Calculate threat level
        threat_level = self._calculate_threat_level(event_type, indicators, context)
        
        # Generate incident ID
        incident_id = self._generate_incident_id(event_type, context.ip_address)
        
        incident = SecurityIncident(
            incident_id=incident_id,
            event_type=event_type,
            threat_level=threat_level,
            timestamp=datetime.now(),
            source_ip=context.ip_address,
            target_resource=indicators.get('resource', 'unknown'),
            user_context=context,
            description=description,
            indicators=indicators,
            response_actions=[],
            mitigated=False,
            impact_score=self._calculate_impact_score(event_type, threat_level)
        )
        
        self.security_incidents.append(incident)
        
        # Determine automated response
        response_actions = self._determine_response_actions(incident)
        incident.response_actions = response_actions
        
        # Execute high-priority responses immediately
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._execute_incident_response(incident)
        else:
            # Queue for batch processing
            await self.request_queue.put(incident)
        
        self.logger.warning(f"Security incident generated: {incident_id} - "
                           f"{event_type.value} ({threat_level.value}) - {description}")
        
        return incident
    
    def _calculate_threat_level(self, event_type: SecurityEvent,
                              indicators: Dict[str, Any],
                              context: SecurityContext) -> ThreatLevel:
        """Calculate threat level for incident"""
        
        base_severity = {
            SecurityEvent.AUTHENTICATION_FAILURE: ThreatLevel.LOW,
            SecurityEvent.AUTHORIZATION_VIOLATION: ThreatLevel.MEDIUM,
            SecurityEvent.SUSPICIOUS_ACTIVITY: ThreatLevel.MEDIUM,
            SecurityEvent.ANOMALOUS_BEHAVIOR: ThreatLevel.MEDIUM,
            SecurityEvent.MALWARE_DETECTION: ThreatLevel.HIGH,
            SecurityEvent.DATA_EXFILTRATION: ThreatLevel.CRITICAL,
            SecurityEvent.PRIVILEGE_ESCALATION: ThreatLevel.HIGH,
            SecurityEvent.LATERAL_MOVEMENT: ThreatLevel.HIGH,
            SecurityEvent.COMMAND_INJECTION: ThreatLevel.HIGH,
            SecurityEvent.SQL_INJECTION: ThreatLevel.HIGH,
            SecurityEvent.BRUTE_FORCE: ThreatLevel.MEDIUM,
            SecurityEvent.DDoS_ATTACK: ThreatLevel.HIGH,
            SecurityEvent.INSIDER_THREAT: ThreatLevel.HIGH
        }
        
        threat_level = base_severity.get(event_type, ThreatLevel.MEDIUM)
        
        # Escalate based on indicators
        if indicators.get('confidence', 0) > 0.9:
            threat_level = self._escalate_threat_level(threat_level)
        
        if indicators.get('anomaly_score', 0) > 0.9:
            threat_level = self._escalate_threat_level(threat_level)
        
        # Escalate for privileged users
        if 'admin' in context.permissions or 'root' in context.permissions:
            threat_level = self._escalate_threat_level(threat_level)
        
        return threat_level
    
    def _escalate_threat_level(self, current_level: ThreatLevel) -> ThreatLevel:
        """Escalate threat level by one step"""
        escalation_map = {
            ThreatLevel.LOW: ThreatLevel.MEDIUM,
            ThreatLevel.MEDIUM: ThreatLevel.HIGH,
            ThreatLevel.HIGH: ThreatLevel.CRITICAL,
            ThreatLevel.CRITICAL: ThreatLevel.CRITICAL  # Already max
        }
        return escalation_map[current_level]
    
    def _calculate_impact_score(self, event_type: SecurityEvent, 
                              threat_level: ThreatLevel) -> float:
        """Calculate business impact score"""
        
        base_impact = {
            ThreatLevel.LOW: 0.1,
            ThreatLevel.MEDIUM: 0.3,
            ThreatLevel.HIGH: 0.7,
            ThreatLevel.CRITICAL: 1.0
        }
        
        event_multiplier = {
            SecurityEvent.DATA_EXFILTRATION: 1.5,
            SecurityEvent.MALWARE_DETECTION: 1.3,
            SecurityEvent.EMERGENCY_SHUTDOWN: 1.4,
            SecurityEvent.INSIDER_THREAT: 1.2
        }
        
        impact = base_impact[threat_level] * event_multiplier.get(event_type, 1.0)
        return min(1.0, impact)
    
    def _determine_response_actions(self, incident: SecurityIncident) -> List[ResponseAction]:
        """Determine appropriate response actions for incident"""
        
        base_actions = self.response_mappings.get(incident.event_type, [ResponseAction.LOG_ONLY])
        
        # Escalate actions based on threat level
        if incident.threat_level == ThreatLevel.CRITICAL:
            if ResponseAction.ALERT not in base_actions:
                base_actions.append(ResponseAction.ALERT)
            if incident.event_type in [SecurityEvent.DATA_EXFILTRATION, SecurityEvent.MALWARE_DETECTION]:
                base_actions.append(ResponseAction.EMERGENCY_SHUTDOWN)
        
        elif incident.threat_level == ThreatLevel.HIGH:
            if ResponseAction.ALERT not in base_actions:
                base_actions.append(ResponseAction.ALERT)
        
        # Add IP blocking for external threats
        if (incident.user_context and 
            not ipaddress.ip_address(incident.user_context.ip_address).is_private and
            incident.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]):
            base_actions.append(ResponseAction.BLOCK_IP)
        
        return base_actions
    
    async def _execute_incident_response(self, incident: SecurityIncident):
        """Execute automated incident response"""
        
        for action in incident.response_actions:
            try:
                if action == ResponseAction.BLOCK_IP:
                    await self._block_ip(incident.source_ip)
                elif action == ResponseAction.QUARANTINE_USER:
                    if incident.user_context:
                        await self._quarantine_user(incident.user_context.user_id)
                elif action == ResponseAction.TERMINATE_SESSION:
                    if incident.user_context and incident.user_context.session_id:
                        await self._terminate_session(incident.user_context.session_id)
                elif action == ResponseAction.RATE_LIMIT:
                    await self._apply_rate_limit(incident.source_ip)
                elif action == ResponseAction.ISOLATE_DEVICE:
                    if incident.user_context:
                        await self._isolate_device(incident.user_context.device_id)
                elif action == ResponseAction.EMERGENCY_SHUTDOWN:
                    await self._emergency_shutdown(incident)
                elif action == ResponseAction.ALERT:
                    await self._send_security_alert(incident)
                
                self.logger.info(f"Executed response action {action.value} for incident {incident.incident_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to execute response action {action.value}: {e}")
        
        incident.mitigated = True
    
    # Response action implementations
    async def _block_ip(self, ip_address: str):
        """Block IP address"""
        self.blocked_ips.add(ip_address)
        self.threat_intel.update_reputation(ip_address, -0.5)
        # In real implementation, this would update firewall rules
        
    async def _quarantine_user(self, user_id: str):
        """Quarantine user account"""
        self.quarantined_users.add(user_id)
        # Terminate all active sessions for user
        sessions_to_terminate = [s for s in self.active_sessions.values() 
                               if s.user_id == user_id]
        for session in sessions_to_terminate:
            await self._terminate_session(session.session_id)
    
    async def _terminate_session(self, session_id: str):
        """Terminate user session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        # In real implementation, this would invalidate session tokens
    
    async def _apply_rate_limit(self, ip_address: str):
        """Apply rate limiting to IP"""
        # In real implementation, this would configure rate limiting
        pass
    
    async def _isolate_device(self, device_id: str):
        """Isolate device from network"""
        # In real implementation, this would trigger network isolation
        pass
    
    async def _emergency_shutdown(self, incident: SecurityIncident):
        """Trigger emergency shutdown procedures"""
        self.logger.critical(f"EMERGENCY SHUTDOWN triggered by incident {incident.incident_id}")
        # In real implementation, this would shut down critical systems
    
    async def _send_security_alert(self, incident: SecurityIncident):
        """Send security alert to administrators"""
        # In real implementation, this would send notifications
        self.logger.warning(f"SECURITY ALERT: {incident.description}")
    
    def _generate_incident_id(self, event_type: SecurityEvent, source_ip: str) -> str:
        """Generate unique incident ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_input = f"{event_type.value}-{source_ip}-{timestamp}"
        hash_short = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"INC-{timestamp}-{hash_short}"
    
    # Monitoring loops
    async def _security_monitoring_loop(self):
        """Main security monitoring loop"""
        
        while self._monitoring_active:
            try:
                # Monitor active sessions for suspicious activity
                await self._monitor_active_sessions()
                
                # Check for threat intelligence updates
                await self._check_threat_intelligence()
                
                # Analyze security metrics
                await self._analyze_security_metrics()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in security monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _incident_response_loop(self):
        """Process queued security incidents"""
        
        while self._monitoring_active:
            try:
                # Process queued incidents
                incident = await asyncio.wait_for(self.request_queue.get(), timeout=10)
                await self._execute_incident_response(incident)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in incident response loop: {e}")
                await asyncio.sleep(10)
    
    async def _threat_intelligence_update_loop(self):
        """Update threat intelligence periodically"""
        
        while self._monitoring_active:
            try:
                # Update reputation scores decay
                self.threat_intel._decay_reputation_scores()
                
                # Clean up old incidents (keep last 10000)
                if len(self.security_incidents) > 10000:
                    self.security_incidents = self.security_incidents[-10000:]
                
                # Clean up old sessions
                cutoff_time = datetime.now() - timedelta(hours=24)
                expired_sessions = [
                    session_id for session_id, context in self.active_sessions.items()
                    if context.last_activity < cutoff_time
                ]
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                self.logger.error(f"Error in threat intelligence update: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_active_sessions(self):
        """Monitor active sessions for suspicious patterns"""
        
        for session_id, context in list(self.active_sessions.items()):
            # Check for stale sessions
            time_since_activity = (datetime.now() - context.last_activity).total_seconds()
            
            if time_since_activity > 3600:  # 1 hour idle
                await self._terminate_session(session_id)
                continue
            
            # Check for trust score degradation
            if context.trust_score < self.trust_thresholds['user_quarantine_threshold']:
                await self._generate_security_incident(
                    SecurityEvent.SUSPICIOUS_ACTIVITY,
                    context,
                    f"Session trust score degraded to {context.trust_score:.3f}",
                    {"trust_score": context.trust_score, "risk_factors": context.risk_factors}
                )
    
    async def _check_threat_intelligence(self):
        """Check for new threat intelligence"""
        # In real implementation, this would fetch from external threat feeds
        pass
    
    async def _analyze_security_metrics(self):
        """Analyze overall security metrics"""
        
        # Calculate current security posture
        total_incidents = len(self.security_incidents)
        recent_incidents = [i for i in self.security_incidents 
                          if (datetime.now() - i.timestamp).total_seconds() < 3600]
        
        if total_incidents > 0:
            critical_incidents = len([i for i in recent_incidents 
                                    if i.threat_level == ThreatLevel.CRITICAL])
            
            if critical_incidents > 5:  # More than 5 critical incidents in last hour
                self.logger.warning("High volume of critical security incidents detected")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        recent_incidents = [i for i in self.security_incidents if i.timestamp >= last_24h]
        
        incident_by_type = defaultdict(int)
        incident_by_level = defaultdict(int)
        
        for incident in recent_incidents:
            incident_by_type[incident.event_type.value] += 1
            incident_by_level[incident.threat_level.value] += 1
        
        return {
            'timestamp': now.isoformat(),
            'active_sessions': len(self.active_sessions),
            'blocked_ips': len(self.blocked_ips),
            'quarantined_users': len(self.quarantined_users),
            'total_incidents': len(self.security_incidents),
            'recent_incidents_24h': len(recent_incidents),
            'incidents_by_type': dict(incident_by_type),
            'incidents_by_level': dict(incident_by_level),
            'top_risk_ips': sorted(
                [(ip, score) for ip, score in self.threat_intel.reputation_scores.items()],
                key=lambda x: x[1]
            )[:10],
            'security_posture': self._calculate_security_posture()
        }
    
    def _calculate_security_posture(self) -> str:
        """Calculate overall security posture"""
        
        recent_incidents = [i for i in self.security_incidents 
                          if (datetime.now() - i.timestamp).total_seconds() < 3600]
        
        critical_count = len([i for i in recent_incidents if i.threat_level == ThreatLevel.CRITICAL])
        high_count = len([i for i in recent_incidents if i.threat_level == ThreatLevel.HIGH])
        
        if critical_count > 3:
            return "CRITICAL"
        elif critical_count > 0 or high_count > 5:
            return "HIGH_RISK"
        elif high_count > 0 or len(recent_incidents) > 10:
            return "ELEVATED"
        else:
            return "NORMAL"
    
    def export_security_report(self) -> str:
        """Export comprehensive security report"""
        
        dashboard = self.get_security_dashboard()
        
        # Add detailed incident data
        recent_incidents = [
            {
                'incident_id': i.incident_id,
                'event_type': i.event_type.value,
                'threat_level': i.threat_level.value,
                'timestamp': i.timestamp.isoformat(),
                'source_ip': i.source_ip,
                'description': i.description,
                'mitigated': i.mitigated,
                'impact_score': i.impact_score,
                'response_actions': [a.value for a in i.response_actions]
            }
            for i in self.security_incidents[-100:]
        ]
        
        report = {
            'dashboard': dashboard,
            'recent_incidents': recent_incidents,
            'blocked_entities': {
                'ips': list(self.blocked_ips),
                'users': list(self.quarantined_users)
            },
            'threat_intelligence': {
                'reputation_scores_count': len(self.threat_intel.reputation_scores),
                'attack_patterns_detected': len(self.threat_intel.attack_patterns)
            }
        }
        
        return json.dumps(report, indent=2, default=str)


# Factory function
def create_zero_trust_security_framework(config_path: Optional[str] = None) -> ZeroTrustSecurityFramework:
    """Create and configure zero-trust security framework"""
    config = ConfigManager(config_path)
    return ZeroTrustSecurityFramework(config)


# CLI interface
if __name__ == "__main__":
    import sys
    
    async def main():
        framework = create_zero_trust_security_framework()
        
        if len(sys.argv) > 1 and sys.argv[1] == "start":
            # Start monitoring
            print("Starting zero-trust security monitoring...")
            await framework.start_zero_trust_monitoring()
            
        elif len(sys.argv) > 1 and sys.argv[1] == "dashboard":
            # Show dashboard
            dashboard = framework.get_security_dashboard()
            print(json.dumps(dashboard, indent=2))
            
        elif len(sys.argv) > 1 and sys.argv[1] == "test":
            # Test access request
            context = SecurityContext(
                user_id="test_user",
                device_id="test_device",
                session_id="test_session",
                ip_address="192.168.1.100",
                user_agent="Test Agent",
                permissions={"data:read", "api:access"}
            )
            
            granted, trust, risks = await framework.evaluate_access_request(
                context, "data", "read", {"url": "/api/data"}
            )
            
            print(f"Access granted: {granted}")
            print(f"Trust score: {trust:.3f}")
            print(f"Risk factors: {risks}")
            
        elif len(sys.argv) > 1 and sys.argv[1] == "report":
            # Export report
            report = framework.export_security_report()
            with open("security_report.json", "w") as f:
                f.write(report)
            print("Security report saved to security_report.json")
            
        else:
            print("Usage:")
            print("  python zero_trust_security.py start      # Start monitoring")
            print("  python zero_trust_security.py dashboard  # Show dashboard")
            print("  python zero_trust_security.py test       # Test access request")
            print("  python zero_trust_security.py report     # Export security report")
    
    asyncio.run(main())