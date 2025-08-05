"""
Data validation and sanitization system.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from ..sensors.base import SensorReading, SensorType


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[Any] = None
    validation_score: float = 1.0  # 0-1, lower means more issues
    severity: ValidationSeverity = ValidationSeverity.INFO


class SensorReadingValidator:
    """
    Validates and sanitizes sensor readings.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define validation rules
        self.value_ranges = {
            SensorType.E_NOSE: (0.1, 50000.0),
            SensorType.TEMPERATURE: (-100.0, 300.0),
            SensorType.HUMIDITY: (0.0, 100.0),
            SensorType.PRESSURE: (500.0, 1500.0),
            SensorType.PH: (0.0, 14.0),
            SensorType.FLOW_RATE: (0.0, 10000.0)
        }
        
        # Statistical validation parameters
        self.max_std_dev_ratio = 5.0  # Max std dev as ratio of mean
        self.min_samples_for_stats = 3
        self.outlier_threshold = 3.0  # Standard deviations
        
        # Temporal validation
        self.max_value_change_rate = 0.5  # Max change per second as fraction of value
        self.min_reading_interval = timedelta(milliseconds=10)
        self.max_reading_interval = timedelta(minutes=10)
        
        # Historical data for trend analysis
        self.reading_history: Dict[str, List[SensorReading]] = {}
        self.max_history_size = 100
    
    def validate_reading(self, reading: SensorReading) -> ValidationResult:
        """
        Comprehensive validation of a sensor reading.
        
        Args:
            reading: Sensor reading to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        cleaned_data = None
        validation_score = 1.0
        
        # Basic structure validation
        struct_result = self._validate_structure(reading)
        errors.extend(struct_result["errors"])
        warnings.extend(struct_result["warnings"])
        validation_score *= struct_result["score"]
        
        # Value range validation
        range_result = self._validate_value_ranges(reading)
        errors.extend(range_result["errors"])
        warnings.extend(range_result["warnings"])
        validation_score *= range_result["score"]
        
        # Statistical validation
        stats_result = self._validate_statistics(reading)
        errors.extend(stats_result["errors"])
        warnings.extend(stats_result["warnings"])
        validation_score *= stats_result["score"]
        
        # Temporal validation
        temporal_result = self._validate_temporal_consistency(reading)
        errors.extend(temporal_result["errors"])
        warnings.extend(temporal_result["warnings"])
        validation_score *= temporal_result["score"]
        
        # Quality score validation
        quality_result = self._validate_quality_score(reading)
        errors.extend(quality_result["errors"])
        warnings.extend(quality_result["warnings"])
        validation_score *= quality_result["score"]
        
        # Data cleaning if minor issues found
        if warnings and not errors:
            cleaned_data = self._clean_reading(reading, warnings)
        
        # Determine severity
        if errors:
            if any("critical" in error.lower() for error in errors):
                severity = ValidationSeverity.CRITICAL
            else:
                severity = ValidationSeverity.ERROR
        elif warnings:
            severity = ValidationSeverity.WARNING
        else:
            severity = ValidationSeverity.INFO
        
        # Update history
        self._update_history(reading)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_data,
            validation_score=validation_score,
            severity=severity
        )
    
    def _validate_structure(self, reading: SensorReading) -> Dict[str, Any]:
        """Validate basic structure of the reading."""
        errors = []
        warnings = []
        score = 1.0
        
        # Check required fields
        if not reading.sensor_id:
            errors.append("Missing sensor_id")
            score *= 0.5
        
        if not isinstance(reading.sensor_id, str):
            errors.append("sensor_id must be string")
            score *= 0.8
        
        if not reading.values:
            errors.append("Missing sensor values")
            score *= 0.0
        elif not isinstance(reading.values, list):
            errors.append("Values must be a list")
            score *= 0.5
        
        if not reading.timestamp:
            warnings.append("Missing timestamp")
            score *= 0.9
        
        # Check timestamp validity
        if reading.timestamp:
            now = datetime.now()
            if reading.timestamp > now + timedelta(minutes=5):
                warnings.append("Timestamp is in the future")
                score *= 0.95
            elif reading.timestamp < now - timedelta(days=1):
                warnings.append("Timestamp is very old")
                score *= 0.95
        
        # Validate quality score
        if not (0.0 <= reading.quality_score <= 1.0):
            warnings.append("Quality score should be between 0 and 1")
            score *= 0.9
        
        return {"errors": errors, "warnings": warnings, "score": score}
    
    def _validate_value_ranges(self, reading: SensorReading) -> Dict[str, Any]:
        """Validate sensor values are within expected ranges."""
        errors = []
        warnings = []
        score = 1.0
        
        if not reading.values:
            return {"errors": errors, "warnings": warnings, "score": score}
        
        # Get expected range for sensor type
        expected_range = self.value_ranges.get(reading.sensor_type, (0.0, float('inf')))
        min_val, max_val = expected_range
        
        out_of_range = 0
        negative_values = 0
        zero_values = 0
        
        for i, value in enumerate(reading.values):
            if not isinstance(value, (int, float)):
                errors.append(f"Non-numeric value at index {i}: {value}")
                score *= 0.8
                continue
            
            if np.isnan(value) or np.isinf(value):
                errors.append(f"Invalid value at index {i}: {value}")
                score *= 0.8
                continue
            
            if value < min_val or value > max_val:
                out_of_range += 1
            
            if value < 0:
                negative_values += 1
            
            if value == 0:
                zero_values += 1
        
        # Evaluate out of range values
        if out_of_range > 0:
            ratio = out_of_range / len(reading.values)
            if ratio > 0.5:
                errors.append(f"{out_of_range} values out of expected range {expected_range}")
                score *= 0.5
            elif ratio > 0.1:
                warnings.append(f"{out_of_range} values out of expected range {expected_range}")
                score *= 0.8
        
        # Check for suspicious patterns
        if negative_values > len(reading.values) * 0.1:
            warnings.append(f"High number of negative values: {negative_values}")
            score *= 0.9
        
        if zero_values > len(reading.values) * 0.3:
            warnings.append(f"High number of zero values: {zero_values}")
            score *= 0.9
        
        return {"errors": errors, "warnings": warnings, "score": score}
    
    def _validate_statistics(self, reading: SensorReading) -> Dict[str, Any]:
        """Validate statistical properties of the values."""
        errors = []
        warnings = []
        score = 1.0
        
        if not reading.values or len(reading.values) < self.min_samples_for_stats:
            return {"errors": errors, "warnings": warnings, "score": score}
        
        values_array = np.array([v for v in reading.values if isinstance(v, (int, float))])
        
        if len(values_array) == 0:
            return {"errors": errors, "warnings": warnings, "score": score}
        
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        # Check for unrealistic standard deviation
        if mean_val > 0 and std_val / mean_val > self.max_std_dev_ratio:
            warnings.append(f"Excessive variation: std/mean ratio = {std_val/mean_val:.2f}")
            score *= 0.8
        
        # Check for constant values (may indicate sensor malfunction)  
        if std_val == 0 and len(values_array) > 5:
            warnings.append("All values are identical - possible sensor malfunction")
            score *= 0.7
        
        # Outlier detection using z-score
        if len(values_array) > 5:
            z_scores = np.abs((values_array - mean_val) / (std_val + 1e-6))
            outliers = np.sum(z_scores > self.outlier_threshold)
            
            if outliers > 0:
                outlier_ratio = outliers / len(values_array)
                if outlier_ratio > 0.2:
                    warnings.append(f"High number of outliers: {outliers} ({outlier_ratio:.1%})")
                    score *= 0.8
        
        return {"errors": errors, "warnings": warnings, "score": score}
    
    def _validate_temporal_consistency(self, reading: SensorReading) -> Dict[str, Any]:
        """Validate temporal consistency with previous readings."""
        errors = []
        warnings = []
        score = 1.0
        
        sensor_id = reading.sensor_id
        if sensor_id not in self.reading_history:
            return {"errors": errors, "warnings": warnings, "score": score}
        
        previous_readings = self.reading_history[sensor_id]
        if not previous_readings:
            return {"errors": errors, "warnings": warnings, "score": score}
        
        last_reading = previous_readings[-1]
        
        # Check time interval
        if reading.timestamp and last_reading.timestamp:
            time_diff = reading.timestamp - last_reading.timestamp
            
            if time_diff < self.min_reading_interval:
                warnings.append(f"Reading interval too short: {time_diff.total_seconds():.3f}s")
                score *= 0.9
            elif time_diff > self.max_reading_interval:
                warnings.append(f"Reading interval too long: {time_diff.total_seconds():.1f}s")
                score *= 0.95
        
        # Check value changes
        if (len(reading.values) == len(last_reading.values) and 
            len(reading.values) > 0 and reading.timestamp and last_reading.timestamp):
            
            time_delta = (reading.timestamp - last_reading.timestamp).total_seconds()
            if time_delta > 0:
                for i, (current, previous) in enumerate(zip(reading.values, last_reading.values)):
                    if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
                        if previous > 0:
                            change_rate = abs(current - previous) / previous / time_delta
                            if change_rate > self.max_value_change_rate:
                                warnings.append(
                                    f"Rapid value change in channel {i}: "
                                    f"{change_rate:.2f}/s ({previous:.1f} -> {current:.1f})"
                                )
                                score *= 0.9
        
        return {"errors": errors, "warnings": warnings, "score": score}
    
    def _validate_quality_score(self, reading: SensorReading) -> Dict[str, Any]:
        """Validate the quality score makes sense."""
        errors = []
        warnings = []
        score = 1.0
        
        # Check if quality score matches data characteristics
        if reading.quality_score > 0.8:
            # High quality score should have good data characteristics
            if reading.values:
                values_array = np.array([v for v in reading.values if isinstance(v, (int, float))])
                if len(values_array) > 0:
                    std_val = np.std(values_array)
                    mean_val = np.mean(values_array)
                    
                    # High quality but high variation might be inconsistent
                    if mean_val > 0 and std_val / mean_val > 0.5:
                        warnings.append("High quality score but high data variation")
                        score *= 0.95
        
        elif reading.quality_score < 0.3:
            # Low quality score - should flag for attention
            warnings.append(f"Low quality score: {reading.quality_score:.3f}")
            score *= 0.9
        
        return {"errors": errors, "warnings": warnings, "score": score}
    
    def _clean_reading(self, reading: SensorReading, warnings: List[str]) -> SensorReading:
        """Clean and repair minor issues in sensor reading."""
        # Create a copy of the reading
        cleaned_reading = SensorReading(
            sensor_id=reading.sensor_id,
            sensor_type=reading.sensor_type,
            values=reading.values.copy(),
            timestamp=reading.timestamp,
            metadata=reading.metadata.copy(),
            quality_score=reading.quality_score
        )
        
        # Clean values
        cleaned_values = []
        for value in reading.values:
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                # Clamp to reasonable ranges
                sensor_range = self.value_ranges.get(reading.sensor_type, (0.0, float('inf')))
                clamped_value = max(sensor_range[0], min(sensor_range[1], value))
                cleaned_values.append(clamped_value)
            else:
                # Replace invalid values with interpolated values or zeros
                cleaned_values.append(0.0)
        
        cleaned_reading.values = cleaned_values
        
        # Adjust quality score based on cleaning
        cleaning_penalty = len(warnings) * 0.05
        cleaned_reading.quality_score = max(0.1, reading.quality_score - cleaning_penalty)
        
        # Add cleaning metadata
        cleaned_reading.metadata["cleaned"] = True
        cleaned_reading.metadata["cleaning_warnings"] = warnings
        cleaned_reading.metadata["original_quality_score"] = reading.quality_score
        
        return cleaned_reading
    
    def _update_history(self, reading: SensorReading):
        """Update reading history for temporal validation."""
        sensor_id = reading.sensor_id
        
        if sensor_id not in self.reading_history:
            self.reading_history[sensor_id] = []
        
        self.reading_history[sensor_id].append(reading)
        
        # Limit history size
        if len(self.reading_history[sensor_id]) > self.max_history_size:
            self.reading_history[sensor_id] = self.reading_history[sensor_id][-self.max_history_size:]
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_readings = sum(len(history) for history in self.reading_history.values())
        
        return {
            "total_readings_processed": total_readings,
            "sensors_monitored": len(self.reading_history),
            "average_history_size": total_readings / len(self.reading_history) if self.reading_history else 0,
            "validation_rules": {
                "value_ranges": dict(self.value_ranges),
                "max_std_dev_ratio": self.max_std_dev_ratio,
                "outlier_threshold": self.outlier_threshold,
                "max_value_change_rate": self.max_value_change_rate
            }
        }


class InputSanitizer:
    """
    Sanitizes user inputs and API parameters.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define patterns for different input types
        self.patterns = {
            "sensor_id": re.compile(r'^[a-zA-Z0-9_-]{1,50}$'),
            "agent_id": re.compile(r'^[a-zA-Z0-9_-]{1,50}$'),
            "batch_id": re.compile(r'^[a-zA-Z0-9_-]{1,100}$'),
            "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            "url": re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            "filename": re.compile(r'^[a-zA-Z0-9._-]{1,255}$')
        }
    
    def sanitize_string(self, value: str, input_type: str = "general", 
                       max_length: int = 1000) -> Tuple[str, List[str]]:
        """
        Sanitize string input.
        
        Args:
            value: String to sanitize
            input_type: Type of input for specific validation
            max_length: Maximum allowed length
            
        Returns:
            Tuple of (sanitized_string, warnings)
        """
        warnings = []
        
        if not isinstance(value, str):
            value = str(value)
            warnings.append("Converted non-string input to string")
        
        # Remove null bytes
        if '\x00' in value:
            value = value.replace('\x00', '')
            warnings.append("Removed null bytes")
        
        # Trim whitespace
        original_length = len(value)
        value = value.strip()
        if len(value) != original_length:
            warnings.append("Trimmed whitespace")
        
        # Length validation
        if len(value) > max_length:
            value = value[:max_length]
            warnings.append(f"Truncated to {max_length} characters")
        
        # Pattern validation
        if input_type in self.patterns:
            pattern = self.patterns[input_type]
            if not pattern.match(value):
                warnings.append(f"Input doesn't match expected pattern for {input_type}")
        
        return value, warnings
    
    def sanitize_numeric(self, value: Union[int, float, str], 
                        min_val: Optional[float] = None, 
                        max_val: Optional[float] = None) -> Tuple[float, List[str]]:
        """
        Sanitize numeric input.
        
        Args:
            value: Numeric value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Tuple of (sanitized_number, warnings)
        """
        warnings = []
        
        # Convert to float
        try:
            if isinstance(value, str):
                # Remove common non-numeric characters
                cleaned = re.sub(r'[^\d.-]', '', value)
                numeric_value = float(cleaned)
                if cleaned != value:
                    warnings.append("Removed non-numeric characters")
            else:
                numeric_value = float(value)
        except (ValueError, TypeError):
            numeric_value = 0.0
            warnings.append("Could not convert to number, using 0.0")
        
        # Check for special values
        if np.isnan(numeric_value):
            numeric_value = 0.0
            warnings.append("Converted NaN to 0.0")
        elif np.isinf(numeric_value):
            numeric_value = 0.0
            warnings.append("Converted infinity to 0.0")
        
        # Range validation
        if min_val is not None and numeric_value < min_val:
            numeric_value = min_val
            warnings.append(f"Clamped to minimum value: {min_val}")
        
        if max_val is not None and numeric_value > max_val:
            numeric_value = max_val
            warnings.append(f"Clamped to maximum value: {max_val}")
        
        return numeric_value, warnings
    
    def sanitize_list(self, value: Any, max_length: int = 1000, 
                     item_type: type = str) -> Tuple[List[Any], List[str]]:
        """
        Sanitize list input.
        
        Args:
            value: List to sanitize
            max_length: Maximum allowed length
            item_type: Expected type of list items
            
        Returns:
            Tuple of (sanitized_list, warnings)
        """
        warnings = []
        
        # Convert to list if needed
        if not isinstance(value, list):
            if isinstance(value, (tuple, set)):
                value = list(value)
                warnings.append("Converted to list")
            else:
                value = [value]
                warnings.append("Wrapped single value in list")
        
        # Length validation
        if len(value) > max_length:
            value = value[:max_length]
            warnings.append(f"Truncated to {max_length} items")
        
        # Type validation and conversion
        sanitized_items = []
        for i, item in enumerate(value):
            try:
                if item_type == str:
                    sanitized_item, item_warnings = self.sanitize_string(str(item))
                    warnings.extend([f"Item {i}: {w}" for w in item_warnings])
                elif item_type in (int, float):
                    sanitized_item, item_warnings = self.sanitize_numeric(item)
                    warnings.extend([f"Item {i}: {w}" for w in item_warnings])
                else:
                    sanitized_item = item_type(item)
                
                sanitized_items.append(sanitized_item)
            except (ValueError, TypeError):
                warnings.append(f"Could not convert item {i} to {item_type.__name__}, skipping")
        
        return sanitized_items, warnings