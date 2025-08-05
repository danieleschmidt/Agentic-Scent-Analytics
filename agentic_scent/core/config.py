"""
Configuration management with validation and security.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import hashlib


@dataclass 
class SecurityConfig:
    """Security configuration."""
    enable_encryption: bool = True
    encryption_key_path: Optional[str] = None
    audit_trail_enabled: bool = True
    require_authentication: bool = False
    max_failed_attempts: int = 3
    session_timeout_minutes: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    enable_prometheus: bool = True
    prometheus_port: int = 8090
    enable_health_checks: bool = True
    health_check_interval: int = 30
    alert_webhook_url: Optional[str] = None
    log_level: str = "INFO"
    max_log_size_mb: int = 100
    log_retention_days: int = 30


@dataclass
class PerformanceConfig:
    """Performance and scaling configuration."""
    max_concurrent_analyses: int = 10
    sensor_read_timeout: float = 5.0
    agent_response_timeout: float = 30.0
    cache_size_mb: int = 512
    enable_async_processing: bool = True
    batch_processing_size: int = 100


@dataclass
class ValidationConfig:
    """Data validation configuration."""
    enable_input_validation: bool = True
    sensor_value_range: Dict[str, tuple] = field(default_factory=lambda: {
        "e_nose": (0.1, 10000.0),
        "temperature": (-50.0, 200.0),
        "humidity": (0.0, 100.0),
        "pressure": (800.0, 1200.0)
    })
    required_metadata_fields: List[str] = field(default_factory=lambda: ["timestamp", "sensor_id"])
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 3.0  # Standard deviations


@dataclass
class IntegrationConfig:
    """External system integration configuration."""
    mes_enabled: bool = False
    mes_endpoint: Optional[str] = None
    mes_api_key: Optional[str] = None
    scada_enabled: bool = False
    scada_protocol: str = "OPC_UA"
    scada_endpoint: Optional[str] = None
    cloud_enabled: bool = False
    cloud_provider: str = "aws"
    cloud_region: str = "us-east-1"


@dataclass
class AgenticScentConfig:
    """Main configuration class."""
    version: str = "0.1.0"
    environment: str = "development"  # development, staging, production
    site_id: str = "default"
    production_lines: List[str] = field(default_factory=list)
    
    # Sub-configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    # Configuration metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    config_hash: Optional[str] = None


class ConfigManager:
    """
    Configuration manager with validation, security, and persistence.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".agentic_scent" / "config.json"
        self.logger = logging.getLogger(__name__)
        self._config: Optional[AgenticScentConfig] = None
        
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> AgenticScentConfig:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Validate config hash if present
                stored_hash = config_dict.pop('config_hash', None)
                if stored_hash:
                    calculated_hash = self._calculate_config_hash(config_dict)
                    if stored_hash != calculated_hash:
                        self.logger.warning("Configuration hash mismatch - config may have been tampered with")
                
                # Parse datetime fields
                if config_dict.get('created_at'):
                    config_dict['created_at'] = datetime.fromisoformat(config_dict['created_at'])
                if config_dict.get('updated_at'):
                    config_dict['updated_at'] = datetime.fromisoformat(config_dict['updated_at'])
                
                # Reconstruct nested dataclasses
                config = self._dict_to_config(config_dict)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                self.logger.info("Using default configuration")
                config = AgenticScentConfig()
        else:
            self.logger.info("No configuration file found, creating default")
            config = AgenticScentConfig()
            config.created_at = datetime.now()
        
        # Validate configuration
        self._validate_config(config)
        
        # Update timestamps and hash
        config.updated_at = datetime.now()
        config.config_hash = self._calculate_config_hash(asdict(config))
        
        self._config = config
        return config
    
    def save_config(self, config: AgenticScentConfig):
        """Save configuration to file."""
        try:
            config.updated_at = datetime.now()
            config_dict = asdict(config)
            
            # Convert datetime objects to ISO format
            if config_dict.get('created_at'):
                config_dict['created_at'] = config.created_at.isoformat()
            if config_dict.get('updated_at'):
                config_dict['updated_at'] = config.updated_at.isoformat()
            
            # Calculate and store hash
            config_dict['config_hash'] = self._calculate_config_hash(config_dict)
            
            # Write to file with backup
            backup_path = self.config_path.with_suffix('.json.bak')
            if self.config_path.exists():
                self.config_path.rename(backup_path)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            # Remove backup on successful write
            if backup_path.exists():
                backup_path.unlink()
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            self._config = config
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            # Restore backup if available
            if backup_path.exists():
                backup_path.rename(self.config_path)
            raise
    
    def get_config(self) -> AgenticScentConfig:
        """Get current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, **kwargs) -> AgenticScentConfig:
        """Update configuration values."""
        config = self.get_config()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
        
        self._validate_config(config)
        self.save_config(config)
        return config
    
    def reset_config(self) -> AgenticScentConfig:
        """Reset to default configuration."""
        config = AgenticScentConfig()
        config.created_at = datetime.now()
        self.save_config(config)
        return config
    
    def _validate_config(self, config: AgenticScentConfig):
        """Validate configuration values."""
        errors = []
        
        # Validate environment
        if config.environment not in ["development", "staging", "production"]:
            errors.append(f"Invalid environment: {config.environment}")
        
        # Validate monitoring config
        if config.monitoring.prometheus_port < 1024 or config.monitoring.prometheus_port > 65535:
            errors.append(f"Invalid Prometheus port: {config.monitoring.prometheus_port}")
        
        if config.monitoring.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append(f"Invalid log level: {config.monitoring.log_level}")
        
        # Validate performance config
        if config.performance.max_concurrent_analyses < 1:
            errors.append("max_concurrent_analyses must be at least 1")
        
        if config.performance.sensor_read_timeout <= 0:
            errors.append("sensor_read_timeout must be positive")
        
        # Validate security config
        if config.security.max_failed_attempts < 1:
            errors.append("max_failed_attempts must be at least 1")
        
        if config.security.session_timeout_minutes < 1:
            errors.append("session_timeout_minutes must be at least 1")
        
        # Validate integration config
        if config.integration.mes_enabled and not config.integration.mes_endpoint:
            errors.append("mes_endpoint required when MES is enabled")
        
        if config.integration.scada_enabled and not config.integration.scada_endpoint:
            errors.append("scada_endpoint required when SCADA is enabled")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AgenticScentConfig:
        """Convert dictionary to configuration object."""
        # Handle nested dataclasses
        if 'security' in config_dict and isinstance(config_dict['security'], dict):
            config_dict['security'] = SecurityConfig(**config_dict['security'])
        
        if 'monitoring' in config_dict and isinstance(config_dict['monitoring'], dict):
            config_dict['monitoring'] = MonitoringConfig(**config_dict['monitoring'])
        
        if 'performance' in config_dict and isinstance(config_dict['performance'], dict):
            config_dict['performance'] = PerformanceConfig(**config_dict['performance'])
        
        if 'validation' in config_dict and isinstance(config_dict['validation'], dict):
            config_dict['validation'] = ValidationConfig(**config_dict['validation'])
        
        if 'integration' in config_dict and isinstance(config_dict['integration'], dict):
            config_dict['integration'] = IntegrationConfig(**config_dict['integration'])
        
        return AgenticScentConfig(**config_dict)
    
    def _calculate_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of configuration."""
        # Remove hash field itself and timestamps for consistent hashing
        config_copy = config_dict.copy()
        config_copy.pop('config_hash', None)
        config_copy.pop('created_at', None)
        config_copy.pop('updated_at', None)
        
        config_str = json.dumps(config_copy, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        # Define environment variable mappings
        env_mappings = {
            'AGENTIC_SCENT_SITE_ID': 'site_id',
            'AGENTIC_SCENT_ENVIRONMENT': 'environment',
            'AGENTIC_SCENT_LOG_LEVEL': 'monitoring.log_level',
            'AGENTIC_SCENT_PROMETHEUS_PORT': 'monitoring.prometheus_port',
            'AGENTIC_SCENT_MAX_CONCURRENT': 'performance.max_concurrent_analyses',
            'AGENTIC_SCENT_ENABLE_ENCRYPTION': 'security.enable_encryption',
            'AGENTIC_SCENT_MES_ENDPOINT': 'integration.mes_endpoint',
            'AGENTIC_SCENT_MES_API_KEY': 'integration.mes_api_key',
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle type conversion
                if config_path.endswith('_port') or config_path.endswith('_concurrent'):
                    try:
                        value = int(value)
                    except ValueError:
                        self.logger.warning(f"Invalid integer value for {env_var}: {value}")
                        continue
                elif config_path.endswith('_encryption'):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                overrides[config_path] = value
        
        return overrides
    
    def apply_environment_overrides(self, config: AgenticScentConfig) -> AgenticScentConfig:
        """Apply environment variable overrides to configuration."""
        overrides = self.get_environment_overrides()
        
        for config_path, value in overrides.items():
            self._set_nested_config_value(config, config_path, value)
        
        return config
    
    def _set_nested_config_value(self, config: AgenticScentConfig, path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        parts = path.split('.')
        obj = config
        
        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                self.logger.warning(f"Configuration path not found: {path}")
                return
        
        # Set the final value
        final_key = parts[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
            self.logger.info(f"Applied environment override: {path} = {value}")
        else:
            self.logger.warning(f"Configuration key not found: {path}")


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> AgenticScentConfig:
    """Get current configuration."""
    return get_config_manager().get_config()