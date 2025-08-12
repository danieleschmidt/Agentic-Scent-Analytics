"""
Global compliance framework for GDPR, CCPA, PDPA and other regulations.
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import uuid


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA_SG = "pdpa_sg"  # Personal Data Protection Act (Singapore)
    PDPA_TH = "pdpa_th"  # Personal Data Protection Act (Thailand)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act (US)
    SOX = "sox"  # Sarbanes-Oxley Act (US)
    FDA_CFR21 = "fda_cfr21"  # FDA 21 CFR Part 11 (US Pharmaceuticals)
    EU_GMP = "eu_gmp"  # EU Good Manufacturing Practice


class DataProcessingPurpose(Enum):
    """Lawful purposes for data processing."""
    MANUFACTURING_QUALITY = "manufacturing_quality"
    PROCESS_OPTIMIZATION = "process_optimization"
    COMPLIANCE_MONITORING = "compliance_monitoring"
    SAFETY_ANALYSIS = "safety_analysis"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    REGULATORY_REPORTING = "regulatory_reporting"
    AUDIT_TRAIL = "audit_trail"
    SYSTEM_MONITORING = "system_monitoring"


@dataclass
class DataSubject:
    """Information about a data subject."""
    subject_id: str
    subject_type: str  # e.g., "employee", "visitor", "system_user"
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    consent_purposes: Set[DataProcessingPurpose] = field(default_factory=set)
    opt_out_requests: List[datetime] = field(default_factory=list)
    data_retention_period: Optional[timedelta] = None


@dataclass
class DataProcessingActivity:
    """Record of data processing activity."""
    activity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    data_controller: str = ""
    data_processor: str = ""
    purpose: DataProcessingPurpose = DataProcessingPurpose.SYSTEM_MONITORING
    data_types: List[str] = field(default_factory=list)
    data_subjects: List[str] = field(default_factory=list)
    legal_basis: str = ""
    retention_period: Optional[timedelta] = None
    security_measures: List[str] = field(default_factory=list)
    third_party_sharing: bool = False
    cross_border_transfer: bool = False
    transfer_safeguards: List[str] = field(default_factory=list)


@dataclass
class ComplianceViolation:
    """Record of a compliance violation."""
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    framework: ComplianceFramework
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_data_subjects: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    notification_required: bool = False
    notification_sent: bool = False
    resolved: bool = False
    resolution_date: Optional[datetime] = None


class ComplianceManager:
    """
    Global compliance manager for data protection and regulatory requirements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Active compliance frameworks
        self.active_frameworks = set(self.config.get('active_frameworks', [
            ComplianceFramework.GDPR,
            ComplianceFramework.FDA_CFR21
        ]))
        
        # Data processing registry
        self.processing_activities: List[DataProcessingActivity] = []
        self.data_subjects: Dict[str, DataSubject] = {}
        self.violations: List[ComplianceViolation] = []
        
        # Data retention policies
        self.retention_policies = self._load_retention_policies()
        
        # Privacy controls
        self.privacy_controls = self._initialize_privacy_controls()
        
        self.logger.info(f"Compliance manager initialized with frameworks: {[f.value for f in self.active_frameworks]}")
    
    def _load_retention_policies(self) -> Dict[str, timedelta]:
        """Load data retention policies."""
        # Default retention policies based on data type
        return {
            'sensor_readings': timedelta(days=2555),  # 7 years for manufacturing data
            'quality_records': timedelta(days=2555),  # 7 years for GMP compliance
            'audit_logs': timedelta(days=2555),  # 7 years for audit purposes
            'user_activity': timedelta(days=1095),  # 3 years for user data
            'system_metrics': timedelta(days=365),  # 1 year for system monitoring
            'personal_data': timedelta(days=1095),  # 3 years or upon consent withdrawal
            'session_data': timedelta(days=30),  # 30 days for session information
            'error_logs': timedelta(days=365)  # 1 year for error analysis
        }
    
    def _initialize_privacy_controls(self) -> Dict[str, Any]:
        """Initialize privacy control mechanisms."""
        return {
            'data_minimization': True,
            'purpose_limitation': True,
            'storage_limitation': True,
            'accuracy_requirement': True,
            'security_by_design': True,
            'transparency': True,
            'accountability': True,
            'pseudonymization': True,
            'encryption_at_rest': True,
            'encryption_in_transit': True
        }
    
    def register_data_processing(self, activity: DataProcessingActivity) -> str:
        """Register a data processing activity."""
        # Validate against active frameworks
        if not self._validate_processing_activity(activity):
            raise ValueError("Data processing activity does not comply with active frameworks")
        
        self.processing_activities.append(activity)
        self.logger.info(f"Registered data processing activity: {activity.purpose.value}")
        return activity.activity_id
    
    def register_data_subject(self, subject: DataSubject):
        """Register a data subject."""
        self.data_subjects[subject.subject_id] = subject
        self.logger.info(f"Registered data subject: {subject.subject_type}")
    
    def process_consent(self, subject_id: str, purposes: List[DataProcessingPurpose], 
                       consent_given: bool = True) -> bool:
        """Process consent for data processing."""
        if subject_id not in self.data_subjects:
            self.logger.error(f"Data subject {subject_id} not found")
            return False
        
        subject = self.data_subjects[subject_id]
        
        if consent_given:
            subject.consent_given = True
            subject.consent_date = datetime.now()
            subject.consent_purposes.update(purposes)
            self.logger.info(f"Consent granted for {subject_id}: {[p.value for p in purposes]}")
        else:
            # Consent withdrawal
            subject.consent_given = False
            subject.opt_out_requests.append(datetime.now())
            subject.consent_purposes.clear()
            self.logger.info(f"Consent withdrawn for {subject_id}")
            
            # Trigger data deletion if required
            self._handle_consent_withdrawal(subject_id)
        
        return True
    
    def check_processing_lawfulness(self, purpose: DataProcessingPurpose, 
                                  subject_ids: List[str]) -> Dict[str, bool]:
        """Check if data processing is lawful for given subjects."""
        results = {}
        
        for subject_id in subject_ids:
            if subject_id not in self.data_subjects:
                results[subject_id] = False
                continue
            
            subject = self.data_subjects[subject_id]
            
            # Check consent for GDPR-style frameworks
            if ComplianceFramework.GDPR in self.active_frameworks:
                lawful = (subject.consent_given and 
                         purpose in subject.consent_purposes)
            else:
                # Other frameworks may have different lawfulness criteria
                lawful = True  # Simplified for demo
            
            results[subject_id] = lawful
        
        return results
    
    def anonymize_data(self, data: Dict[str, Any], 
                      anonymization_method: str = "hash") -> Dict[str, Any]:
        """Anonymize personal data."""
        anonymized = data.copy()
        
        # Identify fields that need anonymization
        personal_fields = [
            'user_id', 'employee_id', 'email', 'name', 
            'phone', 'ip_address', 'device_id'
        ]
        
        for field in personal_fields:
            if field in anonymized:
                if anonymization_method == "hash":
                    anonymized[field] = self._hash_value(str(anonymized[field]))
                elif anonymization_method == "pseudonym":
                    anonymized[field] = self._generate_pseudonym(str(anonymized[field]))
                elif anonymization_method == "remove":
                    del anonymized[field]
        
        return anonymized
    
    def _hash_value(self, value: str) -> str:
        """Create a one-way hash of a value."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def _generate_pseudonym(self, value: str) -> str:
        """Generate a consistent pseudonym for a value."""
        hash_value = hashlib.md5(value.encode()).hexdigest()
        return f"USER_{hash_value[:8].upper()}"
    
    def check_data_retention(self) -> List[Dict[str, Any]]:
        """Check for data that should be deleted due to retention policies."""
        expired_data = []
        current_time = datetime.now()
        
        for activity in self.processing_activities:
            if not activity.retention_period:
                continue
            
            expiry_date = activity.timestamp + activity.retention_period
            if current_time > expiry_date:
                expired_data.append({
                    'activity_id': activity.activity_id,
                    'purpose': activity.purpose.value,
                    'expired_date': expiry_date.isoformat(),
                    'data_types': activity.data_types
                })
        
        return expired_data
    
    def handle_data_subject_request(self, subject_id: str, 
                                  request_type: str) -> Dict[str, Any]:
        """Handle data subject requests (GDPR Article 15-22)."""
        if subject_id not in self.data_subjects:
            return {'success': False, 'error': 'Data subject not found'}
        
        subject = self.data_subjects[subject_id]
        response = {'success': True, 'subject_id': subject_id}
        
        if request_type == "access":  # Right of access (Art. 15)
            response['data'] = {
                'personal_data': self._collect_personal_data(subject_id),
                'processing_activities': self._get_subject_processing_activities(subject_id),
                'consent_history': {
                    'consent_given': subject.consent_given,
                    'consent_date': subject.consent_date.isoformat() if subject.consent_date else None,
                    'purposes': [p.value for p in subject.consent_purposes]
                }
            }
        
        elif request_type == "rectification":  # Right to rectification (Art. 16)
            response['message'] = "Data rectification request logged"
            
        elif request_type == "erasure":  # Right to erasure (Art. 17)
            self._erase_subject_data(subject_id)
            response['message'] = "Data erasure initiated"
            
        elif request_type == "portability":  # Right to data portability (Art. 20)
            response['export_data'] = self._export_subject_data(subject_id)
            
        elif request_type == "restrict":  # Right to restrict processing (Art. 18)
            self._restrict_processing(subject_id)
            response['message'] = "Processing restriction applied"
            
        elif request_type == "object":  # Right to object (Art. 21)
            self.process_consent(subject_id, [], consent_given=False)
            response['message'] = "Processing objection recorded"
        
        self.logger.info(f"Handled {request_type} request for subject {subject_id}")
        return response
    
    def assess_privacy_impact(self, processing_activity: DataProcessingActivity) -> Dict[str, Any]:
        """Conduct Privacy Impact Assessment (PIA)."""
        assessment = {
            'activity_id': processing_activity.activity_id,
            'timestamp': datetime.now().isoformat(),
            'risk_level': 'low',
            'risks': [],
            'mitigation_measures': [],
            'recommendation': 'approved'
        }
        
        # Assess risks based on data types and processing purpose
        risk_score = 0
        
        if 'personal_data' in processing_activity.data_types:
            risk_score += 2
            assessment['risks'].append('Processing of personal data')
        
        if 'sensitive_personal_data' in processing_activity.data_types:
            risk_score += 4
            assessment['risks'].append('Processing of sensitive personal data')
        
        if processing_activity.cross_border_transfer:
            risk_score += 3
            assessment['risks'].append('Cross-border data transfer')
        
        if processing_activity.third_party_sharing:
            risk_score += 2
            assessment['risks'].append('Third-party data sharing')
        
        # Determine risk level
        if risk_score >= 7:
            assessment['risk_level'] = 'high'
            assessment['recommendation'] = 'requires_additional_measures'
        elif risk_score >= 4:
            assessment['risk_level'] = 'medium'
            assessment['recommendation'] = 'approved_with_conditions'
        
        # Suggest mitigation measures
        assessment['mitigation_measures'] = [
            'Implement data minimization',
            'Apply pseudonymization where possible',
            'Ensure encryption in transit and at rest',
            'Regular access reviews',
            'Data retention policy compliance'
        ]
        
        return assessment
    
    def generate_compliance_report(self, framework: ComplianceFramework,
                                 start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specified framework."""
        report = {
            'framework': framework.value,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'processing_activities': 0,
                'data_subjects': len(self.data_subjects),
                'violations': 0,
                'resolved_violations': 0
            },
            'details': {
                'processing_activities': [],
                'violations': [],
                'data_subject_requests': [],
                'retention_compliance': []
            }
        }
        
        # Filter activities by date range
        for activity in self.processing_activities:
            if start_date <= activity.timestamp <= end_date:
                report['summary']['processing_activities'] += 1
                report['details']['processing_activities'].append({
                    'activity_id': activity.activity_id,
                    'purpose': activity.purpose.value,
                    'data_types': activity.data_types,
                    'legal_basis': activity.legal_basis
                })
        
        # Filter violations by date range
        for violation in self.violations:
            if start_date <= violation.timestamp <= end_date:
                if violation.framework == framework:
                    report['summary']['violations'] += 1
                    if violation.resolved:
                        report['summary']['resolved_violations'] += 1
                    
                    report['details']['violations'].append({
                        'violation_id': violation.violation_id,
                        'type': violation.violation_type,
                        'severity': violation.severity,
                        'resolved': violation.resolved
                    })
        
        # Check retention compliance
        expired_data = self.check_data_retention()
        report['details']['retention_compliance'] = expired_data
        
        return report
    
    def _validate_processing_activity(self, activity: DataProcessingActivity) -> bool:
        """Validate processing activity against active frameworks."""
        # Basic validation rules
        if not activity.purpose:
            return False
        
        if not activity.data_controller:
            return False
        
        # Framework-specific validation
        for framework in self.active_frameworks:
            if not self._validate_against_framework(activity, framework):
                return False
        
        return True
    
    def _validate_against_framework(self, activity: DataProcessingActivity,
                                  framework: ComplianceFramework) -> bool:
        """Validate activity against specific compliance framework."""
        if framework == ComplianceFramework.GDPR:
            # GDPR requires legal basis
            return bool(activity.legal_basis)
        
        elif framework == ComplianceFramework.FDA_CFR21:
            # FDA 21 CFR Part 11 requires specific security measures
            required_measures = ['access_controls', 'audit_trail', 'data_integrity']
            return all(measure in activity.security_measures for measure in required_measures)
        
        # Add more framework-specific validations
        return True
    
    def _handle_consent_withdrawal(self, subject_id: str):
        """Handle consent withdrawal by initiating data deletion."""
        self.logger.info(f"Initiating data deletion for subject {subject_id}")
        # In production, this would trigger data deletion processes
        
    def _collect_personal_data(self, subject_id: str) -> Dict[str, Any]:
        """Collect all personal data for a subject."""
        # Mock implementation - in production would query all data stores
        return {
            'subject_id': subject_id,
            'collected_from': ['sensor_data', 'user_activity', 'audit_logs'],
            'data_categories': ['identification', 'activity_logs', 'preferences']
        }
    
    def _get_subject_processing_activities(self, subject_id: str) -> List[Dict[str, Any]]:
        """Get all processing activities involving a subject."""
        activities = []
        for activity in self.processing_activities:
            if subject_id in activity.data_subjects:
                activities.append({
                    'activity_id': activity.activity_id,
                    'purpose': activity.purpose.value,
                    'timestamp': activity.timestamp.isoformat()
                })
        return activities
    
    def _erase_subject_data(self, subject_id: str):
        """Erase all data for a subject (GDPR Right to Erasure)."""
        # Remove from data subjects registry
        if subject_id in self.data_subjects:
            del self.data_subjects[subject_id]
        
        # Mark processing activities for data deletion
        for activity in self.processing_activities:
            if subject_id in activity.data_subjects:
                activity.data_subjects.remove(subject_id)
        
        self.logger.info(f"Erased data for subject {subject_id}")
    
    def _export_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Export all data for a subject in portable format."""
        return {
            'subject_id': subject_id,
            'export_date': datetime.now().isoformat(),
            'personal_data': self._collect_personal_data(subject_id),
            'format': 'JSON',
            'version': '1.0'
        }
    
    def _restrict_processing(self, subject_id: str):
        """Restrict processing for a subject."""
        if subject_id in self.data_subjects:
            self.data_subjects[subject_id].consent_given = False
            self.logger.info(f"Restricted processing for subject {subject_id}")
    
    def log_violation(self, framework: ComplianceFramework, violation_type: str,
                     severity: str, description: str, **kwargs) -> str:
        """Log a compliance violation."""
        violation = ComplianceViolation(
            framework=framework,
            violation_type=violation_type,
            severity=severity,
            description=description,
            **kwargs
        )
        
        self.violations.append(violation)
        self.logger.warning(f"Compliance violation logged: {violation_type} ({severity})")
        
        return violation.violation_id
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        active_violations = [v for v in self.violations if not v.resolved]
        
        return {
            'active_frameworks': [f.value for f in self.active_frameworks],
            'processing_activities': len(self.processing_activities),
            'data_subjects': len(self.data_subjects),
            'total_violations': len(self.violations),
            'active_violations': len(active_violations),
            'compliance_score': self._calculate_compliance_score(),
            'last_assessment': datetime.now().isoformat()
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-1)."""
        if not self.violations:
            return 1.0
        
        active_violations = [v for v in self.violations if not v.resolved]
        if not active_violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.6, 'critical': 1.0}
        total_weight = sum(severity_weights.get(v.severity, 0.5) for v in active_violations)
        
        # Calculate score based on violations
        max_possible_weight = len(active_violations) * 1.0
        score = max(0.0, 1.0 - (total_weight / max_possible_weight))
        
        return score


# Global compliance manager
_compliance_manager = None


def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager instance."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager()
    return _compliance_manager