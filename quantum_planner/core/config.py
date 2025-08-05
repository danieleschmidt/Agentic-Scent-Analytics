"""Configuration management for quantum task planner."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path


@dataclass
class PlannerConfig:
    # Quantum optimization parameters
    max_iterations: int = 1000
    convergence_threshold: float = 0.001
    quantum_annealing_strength: float = 0.5
    entanglement_factor: float = 0.2
    
    # Scheduling parameters
    time_horizon_days: int = 30
    resource_buffer_percent: float = 0.1
    priority_weight: float = 1.0
    deadline_weight: float = 2.0
    
    # Performance parameters  
    max_concurrent_tasks: int = 10
    task_timeout_seconds: int = 3600
    retry_attempts: int = 3
    cache_size: int = 1000
    
    # Agent parameters
    max_agents: int = 5
    load_balancing_enabled: bool = True
    consensus_threshold: float = 0.7
    
    # I18n and localization
    default_language: str = "en"
    supported_languages: list = field(default_factory=lambda: ["en", "es", "fr", "de", "ja", "zh"])
    timezone: str = "UTC"
    
    # Security and compliance
    encryption_enabled: bool = True
    audit_logging: bool = True
    data_retention_days: int = 365
    
    # Storage configuration
    storage_backend: str = "sqlite"  # sqlite, postgresql, redis
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # API configuration
    api_host: str = "localhost"
    api_port: int = 8000
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Development and debugging
    debug_mode: bool = False
    log_level: str = "INFO"
    profile_performance: bool = False
    
    def __post_init__(self):
        # Validate configuration
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0 <= self.convergence_threshold <= 1:
            raise ValueError("convergence_threshold must be between 0 and 1")
        if not 0 <= self.quantum_annealing_strength <= 1:
            raise ValueError("quantum_annealing_strength must be between 0 and 1")
        if self.time_horizon_days <= 0:
            raise ValueError("time_horizon_days must be positive")
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        env_mapping = {
            "QUANTUM_PLANNER_MAX_ITERATIONS": ("max_iterations", int),
            "QUANTUM_PLANNER_CONVERGENCE_THRESHOLD": ("convergence_threshold", float),
            "QUANTUM_PLANNER_ANNEALING_STRENGTH": ("quantum_annealing_strength", float),
            "QUANTUM_PLANNER_TIME_HORIZON": ("time_horizon_days", int),
            "QUANTUM_PLANNER_MAX_CONCURRENT": ("max_concurrent_tasks", int),
            "QUANTUM_PLANNER_DEBUG": ("debug_mode", bool),
            "QUANTUM_PLANNER_LOG_LEVEL": ("log_level", str),
            "QUANTUM_PLANNER_DATABASE_URL": ("database_url", str),
            "QUANTUM_PLANNER_REDIS_URL": ("redis_url", str),
            "QUANTUM_PLANNER_API_HOST": ("api_host", str),
            "QUANTUM_PLANNER_API_PORT": ("api_port", int),
        }
        
        for env_var, (attr_name, attr_type) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                if attr_type == bool:
                    value = value.lower() in ("true", "1", "yes", "on")
                else:
                    value = attr_type(value)
                setattr(self, attr_name, value)
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration based on backend."""
        if self.storage_backend == "postgresql":
            return {
                "url": self.database_url or "postgresql://user:pass@localhost/quantum_planner",
                "pool_size": 20,
                "max_overflow": 30,
                "pool_timeout": 30,
            }
        elif self.storage_backend == "redis":
            return {
                "url": self.redis_url or "redis://localhost:6379/0",
                "decode_responses": True,
                "max_connections": 20,
            }
        else:  # sqlite
            return {
                "url": self.database_url or "sqlite:///quantum_planner.db",
                "check_same_thread": False,
            }
    
    def get_quantum_params(self) -> Dict[str, float]:
        """Get quantum algorithm parameters."""
        return {
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "annealing_strength": self.quantum_annealing_strength,
            "entanglement_factor": self.entanglement_factor,
        }
    
    def get_scheduling_params(self) -> Dict[str, Any]:
        """Get scheduling parameters."""
        return {
            "time_horizon_days": self.time_horizon_days,
            "resource_buffer_percent": self.resource_buffer_percent,
            "priority_weight": self.priority_weight,
            "deadline_weight": self.deadline_weight,
            "max_concurrent_tasks": self.max_concurrent_tasks,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    @classmethod
    def create_default(cls) -> "PlannerConfig":
        """Create default configuration."""
        config = cls()
        config.load_from_env()
        return config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})