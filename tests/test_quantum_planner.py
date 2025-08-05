"""Comprehensive tests for quantum task planner."""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

from quantum_planner.core.planner import QuantumTaskPlanner
from quantum_planner.core.task import Task, TaskPriority, TaskStatus
from quantum_planner.core.config import PlannerConfig
from quantum_planner.agents.task_agent import TaskAgent
from quantum_planner.algorithms.quantum_optimizer import QuantumOptimizer
from quantum_planner.security.validator import TaskValidator


class TestQuantumTaskPlanner:
    """Test cases for QuantumTaskPlanner."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PlannerConfig(
            max_iterations=100,
            max_concurrent_tasks=5,
            debug_mode=True
        )
    
    @pytest.fixture
    def planner(self, config):
        """Create test planner instance."""
        return QuantumTaskPlanner(config)
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        return Task(
            name="Test Task",
            description="A test task for unit testing",
            priority=TaskPriority.MEDIUM,
            estimated_duration=timedelta(minutes=30),
            success_probability=0.9
        )
    
    @pytest.mark.asyncio
    async def test_planner_initialization(self, config):
        """Test planner initializes correctly."""
        planner = QuantumTaskPlanner(config)
        
        assert planner.config == config
        assert len(planner.tasks) == 0
        assert len(planner.completed_tasks) == 0
        assert not planner.is_running
    
    @pytest.mark.asyncio
    async def test_add_task(self, planner, sample_task):
        """Test adding tasks to planner."""
        task_id = await planner.add_task(sample_task)
        
        assert task_id == sample_task.id
        assert task_id in planner.tasks
        assert planner.tasks[task_id] == sample_task
    
    @pytest.mark.asyncio
    async def test_remove_task(self, planner, sample_task):
        """Test removing tasks from planner."""
        await planner.add_task(sample_task)
        
        result = await planner.remove_task(sample_task.id)
        assert result is True
        assert sample_task.id not in planner.tasks
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, planner):
        """Test detection of circular dependencies."""
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2")
        
        await planner.add_task(task1)
        await planner.add_task(task2)
        
        # Add dependency: task2 -> task1
        task2.add_dependency(task1.id)
        await planner.update_task(task2.id, dependencies=task2.dependencies)
        
        # Try to add circular dependency: task1 -> task2
        task1.add_dependency(task2.id)
        
        with pytest.raises(ValueError, match="circular dependency"):
            await planner.add_task(task1)
    
    @pytest.mark.asyncio
    async def test_optimize_schedule(self, planner):
        """Test schedule optimization."""
        # Create test tasks
        tasks = []
        for i in range(5):
            task = Task(
                name=f"Test Task {i}",
                priority=TaskPriority.MEDIUM,
                estimated_duration=timedelta(minutes=30),
                amplitude=0.8,
                phase=i * 0.5
            )
            tasks.append(task)
            await planner.add_task(task)
        
        # Add some dependencies
        tasks[1].add_dependency(tasks[0].id)
        tasks[2].add_dependency(tasks[1].id)
        
        schedule = await planner.optimize_schedule()
        
        assert isinstance(schedule, dict)
        assert len(schedule) <= len(tasks)
        
        # Check schedule respects dependencies
        if tasks[0].id in schedule and tasks[1].id in schedule:
            assert schedule[tasks[0].id] <= schedule[tasks[1].id]
    
    @pytest.mark.asyncio
    async def test_get_ready_tasks(self, planner):
        """Test getting ready tasks."""
        task1 = Task(name="Task 1", priority=TaskPriority.HIGH)
        task2 = Task(name="Task 2", priority=TaskPriority.MEDIUM)
        task3 = Task(name="Task 3", priority=TaskPriority.LOW)
        
        # Task 3 depends on Task 1
        task3.add_dependency(task1.id)
        
        await planner.add_task(task1)
        await planner.add_task(task2)
        await planner.add_task(task3)
        
        ready_tasks = await planner.get_ready_tasks()
        
        # Only task1 and task2 should be ready
        ready_ids = [t.id for t in ready_tasks]
        assert task1.id in ready_ids
        assert task2.id in ready_ids
        assert task3.id not in ready_ids
        
        # Should be sorted by quantum priority (higher first)
        assert ready_tasks[0].priority.value >= ready_tasks[1].priority.value
    
    @pytest.mark.asyncio
    async def test_critical_path(self, planner):
        """Test critical path calculation."""
        # Create task chain: A -> B -> C
        task_a = Task(name="Task A", estimated_duration=timedelta(hours=2))
        task_b = Task(name="Task B", estimated_duration=timedelta(hours=3))
        task_c = Task(name="Task C", estimated_duration=timedelta(hours=1))
        
        task_b.add_dependency(task_a.id)
        task_c.add_dependency(task_b.id)
        
        await planner.add_task(task_a)
        await planner.add_task(task_b)
        await planner.add_task(task_c)
        
        critical_path = await planner.get_critical_path()
        
        assert len(critical_path) >= 2  # Should include at least 2 tasks
        # Critical path should be in dependency order
        assert critical_path.index(task_a.id) < critical_path.index(task_b.id)


class TestQuantumOptimizer:
    """Test cases for QuantumOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create test optimizer."""
        params = {
            "max_iterations": 50,
            "convergence_threshold": 0.01,
            "annealing_strength": 0.5,
            "entanglement_factor": 0.2
        }
        return QuantumOptimizer(params)
    
    @pytest.fixture
    def sample_task_data(self):
        """Create sample optimization data."""
        import numpy as np
        
        n_tasks = 3
        return {
            "tasks": [
                Task(name=f"Task {i}", priority=TaskPriority.MEDIUM)
                for i in range(n_tasks)
            ],
            "task_indices": {f"task_{i}": i for i in range(n_tasks)},
            "adjacency_matrix": np.eye(n_tasks),
            "priorities": np.array([2.0, 3.0, 1.0]),
            "durations": np.array([1800, 3600, 900]),  # seconds
            "n_tasks": n_tasks
        }
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer.max_iterations == 50
        assert optimizer.convergence_threshold == 0.01
        assert optimizer.best_energy == float('inf')
        assert optimizer.quantum_state is None
        
    @pytest.mark.asyncio
    async def test_optimization(self, optimizer, sample_task_data):
        """Test optimization process."""
        result = await optimizer.optimize(sample_task_data)
        
        assert "priority_assignment" in result
        assert "final_energy" in result
        assert "iterations" in result
        assert "converged" in result
        
        priority_assignment = result["priority_assignment"]
        assert len(priority_assignment) == sample_task_data["n_tasks"]
        
        # Check all tasks have priority assignments
        for task in sample_task_data["tasks"]:
            assert task.id in priority_assignment
            assignment = priority_assignment[task.id]
            assert "quantum_priority" in assignment
            assert "amplitude" in assignment
            assert "phase" in assignment


class TestTaskAgent:
    """Test cases for TaskAgent."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PlannerConfig(debug_mode=True)
    
    @pytest.fixture
    def agent(self, config):
        """Create test agent."""
        return TaskAgent(config=config)
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task."""
        return Task(
            name="Agent Test Task",
            estimated_duration=timedelta(seconds=1),
            success_probability=1.0,
            resources_required={"cpu": 0.5}
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.agent_id is not None
        assert agent.state.value == "idle"
        assert agent.quantum_efficiency == 1.0
        assert agent.tasks_completed == 0
    
    @pytest.mark.asyncio
    async def test_agent_start_stop(self, agent):
        """Test agent start and stop."""
        await agent.start()
        assert agent.state.value == "idle"
        
        await agent.stop()
        assert agent.state.value == "maintenance"
    
    @pytest.mark.asyncio
    async def test_task_assignment(self, agent, sample_task):
        """Test task assignment to agent."""
        await agent.start()
        
        # Test can execute task
        can_execute = agent._can_execute_task(sample_task)
        assert can_execute is True
        
        # Test task assignment
        success = await agent.assign_task(sample_task)
        assert success is True
        
        # Wait a bit for task processing
        await asyncio.sleep(0.2)
        
        await agent.stop()
    
    def test_load_factor_calculation(self, agent):
        """Test load factor calculation."""
        # Initially idle, load should be low
        load = agent.get_load_factor()
        assert load >= 0
        
        # Simulate resource usage
        agent.resource_usage = {"cpu": 0.8, "memory": 0.6}
        load_with_resources = agent.get_load_factor()
        assert load_with_resources > load
    
    def test_capability_management(self, agent):
        """Test agent capability management."""
        capabilities = {"python": 0.9, "data_analysis": 0.7, "ml": 0.8}
        
        agent.set_capabilities(capabilities)
        assert agent.capabilities == capabilities
        
        agent.add_capability("testing", 0.95)
        assert agent.capabilities["testing"] == 0.95


class TestTaskValidator:
    """Test cases for TaskValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create test validator."""
        return TaskValidator()
    
    @pytest.fixture
    def valid_task(self):
        """Create valid task for testing."""
        return Task(
            name="Valid Task",
            description="A properly formed task",
            priority=TaskPriority.MEDIUM,
            estimated_duration=timedelta(hours=2),
            success_probability=0.85,
            amplitude=0.8,
            phase=1.5
        )
    
    def test_valid_task_validation(self, validator, valid_task):
        """Test validation of valid task."""
        errors = validator.validate_task(valid_task)
        assert len(errors) == 0
    
    def test_invalid_task_validation(self, validator):
        """Test validation catches invalid data."""
        invalid_task = Task(
            name="",  # Empty name
            description="A" * 3000,  # Too long description
            amplitude=1.5,  # Invalid amplitude > 1.0
            phase=-1.0,  # Invalid phase < 0
            success_probability=1.5  # Invalid probability > 1.0
        )
        
        errors = validator.validate_task(invalid_task)
        assert len(errors) > 0
        
        # Check specific error types
        error_fields = [e.field for e in errors]
        assert any("name" in field for field in error_fields)
        assert any("amplitude" in field for field in error_fields)
        assert any("success_probability" in field for field in error_fields)
    
    def test_task_sanitization(self, validator):
        """Test task data sanitization."""
        dirty_task = Task(
            name="<script>alert('hack')</script>Clean Name",
            description="Normal description with\x00control chars",
            amplitude=1.2,  # Will be clamped to 1.0
            success_probability=-0.1  # Will be clamped to 0.0
        )
        
        clean_task = validator.sanitize_task(dirty_task)
        
        # Check HTML entities are escaped
        assert "<script>" not in clean_task.name
        assert "&lt;script&gt;" in clean_task.name
        
        # Check control characters removed
        assert "\x00" not in clean_task.description
        
        # Check numeric bounds
        assert clean_task.amplitude == 1.0
        assert clean_task.success_probability == 0.0
    
    def test_circular_dependency_validation(self, validator):
        """Test circular dependency detection."""
        task = Task(name="Self-dependent task")
        task.add_dependency(task.id)  # Self-dependency
        
        errors = validator.validate_task(task)
        assert len(errors) > 0
        
        # Should detect self-dependency
        assert any("cannot depend on itself" in e.message for e in errors)
    
    def test_data_integrity_hash(self, validator, valid_task):
        """Test data integrity hash calculation."""
        hash1 = validator.calculate_data_integrity_hash(valid_task)
        hash2 = validator.calculate_data_integrity_hash(valid_task)
        
        # Same task should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        
        # Modified task should produce different hash
        valid_task.name = "Modified Name"
        hash3 = validator.calculate_data_integrity_hash(valid_task)
        assert hash3 != hash1
        
        # Verify integrity
        assert validator.verify_data_integrity(valid_task, hash3)
        assert not validator.verify_data_integrity(valid_task, hash1)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete task planning and execution workflow."""
        config = PlannerConfig(
            max_iterations=50,
            max_concurrent_tasks=3,
            debug_mode=True
        )
        
        planner = QuantumTaskPlanner(config)
        
        # Create test tasks with dependencies
        task1 = Task(name="Setup", priority=TaskPriority.HIGH, 
                    estimated_duration=timedelta(seconds=0.1))
        task2 = Task(name="Process", priority=TaskPriority.MEDIUM,
                    estimated_duration=timedelta(seconds=0.1))
        task3 = Task(name="Cleanup", priority=TaskPriority.LOW,
                    estimated_duration=timedelta(seconds=0.1))
        
        task2.add_dependency(task1.id)
        task3.add_dependency(task2.id)
        
        # Add tasks to planner
        await planner.add_task(task1)
        await planner.add_task(task2)
        await planner.add_task(task3)
        
        # Create and register agents
        agents = []
        for i in range(2):
            agent = TaskAgent(config=config)
            await planner.coordinator.register_agent(agent)
            agents.append(agent)
        
        # Execute workflow
        results = await planner.execute_schedule()
        
        # Verify results
        assert results["status"] in ["completed", "no_tasks"]
        if results["completed"] > 0:
            assert results["completed"] <= 3
            assert "metrics" in results
        
        # Check final status
        status = await planner.get_status()
        assert status["total_tasks"] == 3
        
        # Cleanup agents
        for agent in agents:
            await planner.coordinator.unregister_agent(agent.agent_id)
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load."""
        config = PlannerConfig(
            max_iterations=100,
            max_concurrent_tasks=10,
            debug_mode=False
        )
        
        planner = QuantumTaskPlanner(config)
        
        # Create many tasks
        tasks = []
        for i in range(20):
            task = Task(
                name=f"Load Test Task {i}",
                priority=TaskPriority(i % 4 + 1),
                estimated_duration=timedelta(milliseconds=50),
                amplitude=0.5 + (i % 5) * 0.1,
                phase=i * 0.3
            )
            tasks.append(task)
            await planner.add_task(task)
        
        # Add some dependencies to create complexity
        for i in range(1, min(10, len(tasks))):
            tasks[i].add_dependency(tasks[i-1].id)
        
        # Measure optimization time
        start_time = datetime.now()
        schedule = await planner.optimize_schedule()
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete optimization in reasonable time
        assert optimization_time < 5.0  # 5 seconds max
        assert len(schedule) > 0
        
        # Verify schedule quality
        scheduled_tasks = len(schedule)
        total_tasks = len(tasks)
        scheduling_ratio = scheduled_tasks / total_tasks
        
        # Should schedule most tasks
        assert scheduling_ratio >= 0.7  # At least 70% of tasks scheduled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])