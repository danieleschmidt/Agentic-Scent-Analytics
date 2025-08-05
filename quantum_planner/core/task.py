"""Task models and dependencies using quantum-inspired structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any
from datetime import datetime, timedelta
import uuid


class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskDependency:
    task_id: str
    dependency_type: str = "finish_to_start"
    lag: timedelta = field(default_factory=lambda: timedelta(0))
    
    def __post_init__(self):
        if self.dependency_type not in ["finish_to_start", "start_to_start", "finish_to_finish", "start_to_finish"]:
            raise ValueError(f"Invalid dependency type: {self.dependency_type}")


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    
    # Quantum-inspired properties
    amplitude: float = 1.0  # Quantum amplitude for superposition states
    phase: float = 0.0      # Quantum phase for interference patterns
    entangled_tasks: Set[str] = field(default_factory=set)
    
    # Temporal properties  
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    earliest_start: Optional[datetime] = None
    latest_finish: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_finish: Optional[datetime] = None
    
    # Dependencies and resources
    dependencies: List[TaskDependency] = field(default_factory=list)
    resources_required: Dict[str, float] = field(default_factory=dict)
    success_probability: float = 1.0
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Task-{self.id[:8]}"
            
    def add_dependency(self, task_id: str, dependency_type: str = "finish_to_start", lag: timedelta = None):
        if lag is None:
            lag = timedelta(0)
        self.dependencies.append(TaskDependency(task_id, dependency_type, lag))
        self.updated_at = datetime.now()
    
    def add_entanglement(self, task_id: str):
        self.entangled_tasks.add(task_id)
        self.updated_at = datetime.now()
    
    def set_quantum_state(self, amplitude: float, phase: float):
        self.amplitude = max(0.0, min(1.0, amplitude))
        self.phase = phase % (2 * 3.14159)
        self.updated_at = datetime.now()
    
    def calculate_quantum_priority(self) -> float:
        base_priority = self.priority.value
        quantum_factor = self.amplitude * abs(complex(self.amplitude, self.phase))
        entanglement_boost = len(self.entangled_tasks) * 0.1
        return base_priority * quantum_factor + entanglement_boost
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        if self.status != TaskStatus.PENDING:
            return False
            
        for dep in self.dependencies:
            if dep.task_id not in completed_tasks:
                return False
                
        return True
    
    def can_start_at(self, time: datetime, completed_tasks: Set[str]) -> bool:
        if not self.is_ready(completed_tasks):
            return False
            
        if self.earliest_start and time < self.earliest_start:
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "amplitude": self.amplitude,
            "phase": self.phase,
            "entangled_tasks": list(self.entangled_tasks),
            "estimated_duration": self.estimated_duration.total_seconds(),
            "earliest_start": self.earliest_start.isoformat() if self.earliest_start else None,
            "latest_finish": self.latest_finish.isoformat() if self.latest_finish else None,
            "dependencies": [
                {
                    "task_id": dep.task_id,
                    "type": dep.dependency_type,
                    "lag": dep.lag.total_seconds()
                }
                for dep in self.dependencies
            ],
            "resources_required": self.resources_required,
            "success_probability": self.success_probability,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        task = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            priority=TaskPriority(data["priority"]),
            status=TaskStatus(data["status"]),
            amplitude=data["amplitude"],
            phase=data["phase"],
            entangled_tasks=set(data["entangled_tasks"]),
            estimated_duration=timedelta(seconds=data["estimated_duration"]),
            earliest_start=datetime.fromisoformat(data["earliest_start"]) if data["earliest_start"] else None,
            latest_finish=datetime.fromisoformat(data["latest_finish"]) if data["latest_finish"] else None,
            resources_required=data["resources_required"],
            success_probability=data["success_probability"],
            tags=set(data["tags"]),
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        for dep_data in data["dependencies"]:
            task.dependencies.append(TaskDependency(
                task_id=dep_data["task_id"],
                dependency_type=dep_data["type"],
                lag=timedelta(seconds=dep_data["lag"])
            ))
            
        return task