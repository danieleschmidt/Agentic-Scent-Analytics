"""
Quantum-Inspired Task Planner - Advanced task optimization using quantum computing principles.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__email__ = "contact@terragonlabs.com"

from .core.planner import QuantumTaskPlanner
from .core.task import Task, TaskDependency
from .algorithms.quantum_optimizer import QuantumOptimizer
from .algorithms.scheduler import QuantumScheduler
from .agents.task_agent import TaskAgent
from .agents.coordinator import TaskCoordinator

__all__ = [
    "QuantumTaskPlanner",
    "Task",
    "TaskDependency", 
    "QuantumOptimizer",
    "QuantumScheduler",
    "TaskAgent",
    "TaskCoordinator",
]