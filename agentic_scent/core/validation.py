"""Advanced data validation and sanitization for industrial safety."""

import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from .exceptions import ValidationError, create_error_with_code


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class DataQualityScore(Enum):
    """Data quality assessment."""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    CRITICAL = 1


@dataclass
class ValidationResult:
    """Result of validation with quality metrics."""
    is_valid: bool
    quality_score: DataQualityScore
    anomalies_detected: List[str] = None
    sanitized_data: Dict[str, Any] = None
    validation_warnings: List[str] = None
    temporal_consistency: Optional[float] = None
    statistical_outliers: List[str] = None
    
    def __post_init__(self):
        if self.anomalies_detected is None:
            self.anomalies_detected = []
        if self.validation_warnings is None:
            self.validation_warnings = []
        if self.statistical_outliers is None:
            self.statistical_outliers = []


class AdvancedDataValidator:
    """
    Industrial-grade data validation with statistical analysis and temporal consistency.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
        # Security patterns
        self.sql_injection_patterns = [
            r"(?i)(select|insert|update|delete|drop|create|alter|union|exec|execute)",
            r"(?i)';\s*(drop|delete|truncate|shutdown)",
            r"(?i)--\s*$",
            r"(?i)/\*.*\*/",
            r"(?i)xp_cmdshell",
            r"(?i)sp_executesql"
        ]
        
        self.xss_patterns = [
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)javascript:",
            r"(?i)on\w+\s*=",
            r"(?i)<iframe[^>]*>",
            r"(?i)eval\s*\(",
            r"(?i)document\.cookie"
        ]
        
        self.command_injection_patterns = [
            r"(?i)[;&|`$()]",
            r"(?i)rm\s+-rf",
            r"(?i)wget|curl",
            r"(?i)nc\s+-l",
            r"(?i)/bin/(sh|bash|csh|tcsh|zsh)"
        ]
        
        # Industrial sensor validation ranges
        self.sensor_ranges = {
            'temperature': (-50, 150),  # Celsius
            'humidity': (0, 100),  # %RH
            'pressure': (500, 1500),  # mbar
            'flow_rate': (0, 1000),  # L/min
            'ph': (0, 14),
            'conductivity': (0, 10000),  # µS/cm
            'dissolved_oxygen': (0, 50),  # mg/L
            'turbidity': (0, 1000),  # NTU
        }
        
        # Historical data for temporal analysis
        self.historical_readings: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def validate_comprehensive(self, data: Dict[str, Any], 
                             sensor_type: str = "e_nose",
                             context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Comprehensive validation with statistical analysis.
        
        Args:
            data: Raw sensor data
            sensor_type: Type of sensor generating the data
            context: Additional context for validation
            
        Returns:
            ValidationResult with quality assessment
        """
        validation_warnings = []
        anomalies_detected = []
        statistical_outliers = []
        
        try:
            # Step 1: Basic input validation
            sanitized_data = self._sanitize_input(data)
            
            # Step 2: Security validation
            security_result = self._validate_security(sanitized_data)
            if not security_result:
                return ValidationResult(
                    is_valid=False,
                    quality_score=DataQualityScore.CRITICAL,
                    anomalies_detected=["security_violation_detected"]
                )
            
            # Step 3: Industrial sensor validation
            sensor_result = self._validate_sensor_data(sanitized_data, sensor_type)
            if sensor_result['warnings']:
                validation_warnings.extend(sensor_result['warnings'])
            if sensor_result['anomalies']:
                anomalies_detected.extend(sensor_result['anomalies'])
            
            # Step 4: Statistical validation
            stats_result = self._validate_statistical_properties(sanitized_data)
            if stats_result['outliers']:
                statistical_outliers.extend(stats_result['outliers'])
            
            # Step 5: Temporal consistency check
            temporal_consistency = self._check_temporal_consistency(
                sanitized_data, sensor_type
            )
            
            if temporal_consistency < 0.7:
                anomalies_detected.append("temporal_inconsistency")
                validation_warnings.append(f"Temporal consistency score: {temporal_consistency:.2f}")
            
            # Step 6: Calculate quality score
            quality_score = self._calculate_quality_score(
                len(anomalies_detected),
                len(validation_warnings),
                len(statistical_outliers),
                temporal_consistency
            )
            
            # Determine if data is valid based on validation level
            is_valid = self._determine_validity(
                quality_score, anomalies_detected, validation_warnings
            )
            
            return ValidationResult(
                is_valid=is_valid,
                quality_score=quality_score,
                anomalies_detected=anomalies_detected,
                sanitized_data=sanitized_data,
                validation_warnings=validation_warnings,
                temporal_consistency=temporal_consistency,
                statistical_outliers=statistical_outliers
            )
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                quality_score=DataQualityScore.CRITICAL,
                anomalies_detected=[f"validation_exception: {str(e)}"]
            )
    
    def _sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and normalize input data."""
        if not isinstance(data, dict):
            raise ValidationError("Input must be a dictionary")
        
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize keys
            clean_key = re.sub(r'[^\w\-_.]', '', str(key))[:50]  # Limit key length
            
            # Sanitize values based on type
            if isinstance(value, str):
                # Remove null bytes and control characters
                clean_value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
                # Limit string length
                clean_value = clean_value[:1000] if len(clean_value) > 1000 else clean_value
                sanitized[clean_key] = clean_value
            
            elif isinstance(value, (int, float)):
                # Check for NaN/Inf values
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    sanitized[clean_key] = None
                else:
                    sanitized[clean_key] = value
            
            elif isinstance(value, (list, tuple)):
                # Sanitize array data
                clean_array = []
                for item in value:
                    if isinstance(item, (int, float)) and not (np.isnan(item) if isinstance(item, float) else False):
                        clean_array.append(item)
                    elif isinstance(item, str):
                        clean_item = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', item)[:100]
                        clean_array.append(clean_item)
                sanitized[clean_key] = clean_array
            
            else:
                sanitized[clean_key] = value
        
        return sanitized
    
    def _validate_security(self, data: Dict[str, Any]) -> bool:
        """Advanced security validation."""
        for key, value in data.items():
            if not isinstance(value, str):
                continue
            
            # Check for SQL injection
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, value):
                    self.logger.warning(f"SQL injection pattern detected in {key}: {pattern}")
                    return False
            
            # Check for XSS
            for pattern in self.xss_patterns:
                if re.search(pattern, value):
                    self.logger.warning(f"XSS pattern detected in {key}: {pattern}")
                    return False
            
            # Check for command injection
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                for pattern in self.command_injection_patterns:
                    if re.search(pattern, value):
                        self.logger.warning(f"Command injection pattern detected in {key}: {pattern}")
                        return False
        
        return True
    
    def _validate_sensor_data(self, data: Dict[str, Any], sensor_type: str) -> Dict[str, List[str]]:
        """Validate industrial sensor data ranges and patterns."""
        warnings = []
        anomalies = []
        
        # Validate against known sensor ranges
        for key, value in data.items():
            if key in self.sensor_ranges and isinstance(value, (int, float)):
                min_val, max_val = self.sensor_ranges[key]
                
                if value < min_val or value > max_val:
                    anomalies.append(f"{key}_out_of_range")
                    warnings.append(f"{key} value {value} outside valid range [{min_val}, {max_val}]")
                
                # Check for sudden spikes (>3 std deviations from historical mean)
                if key in self.historical_readings and len(self.historical_readings[key]) > 10:
                    historical_values = [reading[1] for reading in self.historical_readings[key][-20:]]
                    mean_val = np.mean(historical_values)
                    std_val = np.std(historical_values)
                    
                    if abs(value - mean_val) > 3 * std_val:
                        anomalies.append(f"{key}_statistical_outlier")
                        warnings.append(f"{key} value is {abs(value - mean_val)/std_val:.1f} std deviations from mean")
        
        # E-nose specific validation
        if sensor_type == "e_nose":
            self._validate_enose_data(data, warnings, anomalies)
        
        return {"warnings": warnings, "anomalies": anomalies}
    
    def _validate_enose_data(self, data: Dict[str, Any], warnings: List[str], anomalies: List[str]):
        """Specific validation for electronic nose sensors."""
        # Check for sensor readings array
        if 'sensor_readings' in data and isinstance(data['sensor_readings'], list):
            readings = data['sensor_readings']
            
            # Check array length (typical e-nose has 8-32 sensors)
            if len(readings) < 8 or len(readings) > 64:
                warnings.append(f"Unusual number of sensors: {len(readings)}")
            
            # Check for dead sensors (constant readings)
            readings_array = np.array(readings)
            if np.std(readings_array) < 0.001:
                anomalies.append("potential_dead_sensors")
            
            # Check for sensor drift (extreme values)
            if np.any(readings_array > 10000) or np.any(readings_array < -1000):
                anomalies.append("potential_sensor_drift")
    
    def _validate_statistical_properties(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Statistical validation of data properties."""
        outliers = []
        
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 3:
                try:
                    arr = np.array([v for v in value if isinstance(v, (int, float))])
                    if len(arr) > 3:
                        # Detect outliers using IQR method
                        Q1 = np.percentile(arr, 25)
                        Q3 = np.percentile(arr, 75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outlier_indices = np.where((arr < lower_bound) | (arr > upper_bound))[0]
                        if len(outlier_indices) > 0:
                            outliers.append(f"{key}_statistical_outliers")
                
                except Exception as e:
                    self.logger.debug(f"Statistical analysis failed for {key}: {e}")
        
        return {"outliers": outliers}
    
    def _check_temporal_consistency(self, data: Dict[str, Any], sensor_type: str) -> float:
        """Check temporal consistency of sensor readings."""
        if 'timestamp' not in data:
            return 1.0  # No timestamp to check
        
        try:
            current_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except:
            return 0.5  # Invalid timestamp format
        
        consistency_score = 1.0
        
        # Check if readings are from future (clock sync issues)
        if current_time > datetime.now() + timedelta(minutes=5):
            consistency_score *= 0.3
        
        # Check if readings are too old
        if current_time < datetime.now() - timedelta(hours=24):
            consistency_score *= 0.7
        
        # Update historical readings for this sensor
        for key, value in data.items():
            if key in self.sensor_ranges and isinstance(value, (int, float)):
                if key not in self.historical_readings:
                    self.historical_readings[key] = []
                
                # Keep last 100 readings
                self.historical_readings[key].append((current_time, value))
                if len(self.historical_readings[key]) > 100:
                    self.historical_readings[key] = self.historical_readings[key][-100:]
        
        return consistency_score
    
    def _calculate_quality_score(self, num_anomalies: int, num_warnings: int, 
                               num_outliers: int, temporal_consistency: float) -> DataQualityScore:
        """Calculate overall data quality score."""
        # Start with perfect score
        score = 5.0
        
        # Deduct for anomalies
        score -= num_anomalies * 0.8
        
        # Deduct for warnings
        score -= num_warnings * 0.3
        
        # Deduct for statistical outliers
        score -= num_outliers * 0.2
        
        # Factor in temporal consistency
        score *= temporal_consistency
        
        # Convert to enum
        if score >= 4.5:
            return DataQualityScore.EXCELLENT
        elif score >= 3.5:
            return DataQualityScore.GOOD
        elif score >= 2.5:
            return DataQualityScore.ACCEPTABLE
        elif score >= 1.5:
            return DataQualityScore.POOR
        else:
            return DataQualityScore.CRITICAL
    
    def _determine_validity(self, quality_score: DataQualityScore, 
                          anomalies: List[str], warnings: List[str]) -> bool:
        """Determine if data is valid based on validation level."""
        if self.validation_level == ValidationLevel.BASIC:
            return quality_score.value >= 2  # Accept POOR or better
        elif self.validation_level == ValidationLevel.STRICT:
            return quality_score.value >= 3 and len(anomalies) == 0  # Require ACCEPTABLE+ and no anomalies
        else:  # PARANOID
            return quality_score.value >= 4 and len(anomalies) == 0 and len(warnings) <= 1  # Require GOOD+ and minimal issues
    
    def validate_batch_consistency(self, batch_data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate consistency across a batch of sensor readings."""
        if not batch_data:
            return ValidationResult(
                is_valid=False,
                quality_score=DataQualityScore.CRITICAL,
                anomalies_detected=["empty_batch"]
            )
        
        batch_anomalies = []
        batch_warnings = []
        
        # Check batch size
        if len(batch_data) < 5:
            batch_warnings.append("small_batch_size")
        
        # Validate each reading
        valid_readings = 0
        quality_scores = []
        
        for i, reading in enumerate(batch_data):
            result = self.validate_comprehensive(reading)
            quality_scores.append(result.quality_score.value)
            
            if result.is_valid:
                valid_readings += 1
            else:
                batch_anomalies.extend([f"reading_{i}_{anomaly}" for anomaly in result.anomalies_detected])
        
        # Calculate batch quality
        validity_ratio = valid_readings / len(batch_data)
        avg_quality = np.mean(quality_scores)
        
        if validity_ratio < 0.8:
            batch_anomalies.append("low_batch_validity")
        
        batch_quality = DataQualityScore(max(1, min(5, int(avg_quality * validity_ratio))))
        
        return ValidationResult(
            is_valid=validity_ratio >= 0.8 and avg_quality >= 3.0,
            quality_score=batch_quality,
            anomalies_detected=batch_anomalies,
            validation_warnings=batch_warnings,
            temporal_consistency=validity_ratio
        )


# Backward compatibility
class DataValidator(AdvancedDataValidator):
    """Legacy validator for backward compatibility."""
    
    def __init__(self):
        super().__init__(ValidationLevel.BASIC)
    
    def validate_sensor_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy validation method."""
        result = self.validate_comprehensive(data)
        
        if not result.is_valid:
            if "security_violation" in result.anomalies_detected[0] if result.anomalies_detected else "":
                raise ValidationError("Input contains potentially malicious patterns")
            elif any("out_of_range" in anomaly for anomaly in result.anomalies_detected):
                raise ValidationError("Sensor reading out of valid range")
            else:
                raise ValidationError("Data validation failed")
        
        return result.sanitized_data or data