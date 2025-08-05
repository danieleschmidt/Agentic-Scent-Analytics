"""Task coordination and multi-agent orchestration."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict
import numpy as np

from ..core.task import Task, TaskStatus
from ..core.config import PlannerConfig
from .task_agent import TaskAgent, AgentState
from .load_balancer import LoadBalancer
from .consensus import ConsensusEngine


logger = logging.getLogger(__name__)


class TaskCoordinator:
    """Coordinates task execution across multiple agents with quantum-inspired optimization."""
    
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.agents: Dict[str, TaskAgent] = {}
        self.load_balancer = LoadBalancer()
        self.consensus_engine = ConsensusEngine(config.consensus_threshold)
        
        # Coordination state
        self.active_tasks: Dict[str, str] = {}  # task_id -> agent_id
        self.task_assignments: Dict[str, List[str]] = defaultdict(list)  # agent_id -> [task_ids]
        self.execution_results: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.coordination_metrics = {
            "tasks_coordinated": 0,
            "successful_completions": 0,
            "failed_executions": 0,
            "average_coordination_time": 0.0,
            "load_balance_efficiency": 1.0
        }
        
        # Quantum entanglement network
        self.entanglement_network: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info("TaskCoordinator initialized")
    
    async def register_agent(self, agent: TaskAgent) -> bool:
        """Register a new agent with the coordinator."""
        if agent.agent_id in self.agents:
            return False
        
        self.agents[agent.agent_id] = agent
        await agent.start()
        
        # Initialize in load balancer
        self.load_balancer.register_agent(agent.agent_id, agent.get_load_factor)
        
        logger.info(f"Registered agent {agent.agent_id}, total agents: {len(self.agents)}")
        return True
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the coordinator."""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        await agent.stop()
        
        # Remove from load balancer
        self.load_balancer.unregister_agent(agent_id)
        
        # Handle ongoing tasks
        await self._handle_agent_removal(agent_id)
        
        del self.agents[agent_id]
        logger.info(f"Unregistered agent {agent_id}, remaining agents: {len(self.agents)}")
        return True
    
    async def execute_tasks(
        self, 
        schedule: Dict[str, datetime], 
        tasks: Dict[str, Task],
        completed_tasks: Set[str],
        running_tasks: Set[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute scheduled tasks using multi-agent coordination."""
        logger.info(f"Coordinating execution of {len(schedule)} tasks")
        
        if not self.agents:
            raise Exception("No agents available for task execution")
        
        self.coordination_metrics["tasks_coordinated"] += len(schedule)
        
        # Group tasks by scheduled time
        time_groups = self._group_tasks_by_time(schedule, tasks)
        
        # Execute tasks in temporal order
        all_results = {}
        for scheduled_time, task_group in sorted(time_groups.items()):
            # Wait until scheduled time
            await self._wait_until_time(scheduled_time)
            
            # Execute task group with quantum coordination
            group_results = await self._execute_task_group(
                task_group, tasks, completed_tasks, running_tasks
            )
            
            all_results.update(group_results)
            
            # Update completed tasks
            for task_id, result in group_results.items():
                if result["success"]:
                    completed_tasks.add(task_id)
            
            # Apply quantum entanglement effects
            await self._apply_entanglement_effects(task_group, group_results)
        
        # Update coordination metrics
        self._update_coordination_metrics(all_results)
        
        logger.info(f"Coordination completed, {len(all_results)} tasks processed")
        return all_results
    
    def _group_tasks_by_time(
        self, 
        schedule: Dict[str, datetime], 
        tasks: Dict[str, Task]
    ) -> Dict[datetime, List[str]]:
        """Group tasks by their scheduled execution time."""
        time_groups = defaultdict(list)
        
        for task_id, scheduled_time in schedule.items():
            if task_id in tasks:
                # Round to nearest minute for grouping
                rounded_time = scheduled_time.replace(second=0, microsecond=0)
                time_groups[rounded_time].append(task_id)
        
        return time_groups
    
    async def _wait_until_time(self, target_time: datetime):
        """Wait until the target execution time."""
        current_time = datetime.now()
        if target_time > current_time:
            wait_seconds = (target_time - current_time).total_seconds()
            # Cap wait time for simulation
            wait_seconds = min(wait_seconds, 1.0)
            await asyncio.sleep(wait_seconds)
    
    async def _execute_task_group(
        self,
        task_ids: List[str],
        tasks: Dict[str, Task],
        completed_tasks: Set[str],
        running_tasks: Set[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute a group of tasks scheduled for the same time."""
        logger.info(f"Executing task group with {len(task_ids)} tasks")
        
        # Filter ready tasks
        ready_tasks = []
        for task_id in task_ids:
            task = tasks.get(task_id)
            if task and task.is_ready(completed_tasks) and task_id not in running_tasks:
                ready_tasks.append(task_id)
        
        if not ready_tasks:
            return {}
        
        # Quantum-enhanced task assignment
        assignments = await self._quantum_task_assignment(ready_tasks, tasks)
        
        # Execute tasks concurrently
        execution_futures = []
        for task_id, agent_id in assignments.items():
            task = tasks[task_id]
            agent = self.agents[agent_id]
            
            # Track assignment
            self.active_tasks[task_id] = agent_id
            self.task_assignments[agent_id].append(task_id)
            running_tasks.add(task_id)
            
            # Start execution
            future = asyncio.create_task(
                self._execute_single_task(task, agent, task_id)
            )
            execution_futures.append((task_id, future))
        
        # Wait for completions
        results = {}
        for task_id, future in execution_futures:
            try:
                result = await future
                results[task_id] = result
            except Exception as e:
                logger.error(f"Task {task_id} execution failed: {e}")
                results[task_id] = {
                    "success": False,
                    "error": str(e),
                    "start_time": datetime.now(),
                    "finish_time": datetime.now()
                }
            finally:
                # Clean up tracking
                self.active_tasks.pop(task_id, None)
                running_tasks.discard(task_id)
                agent_id = assignments.get(task_id)
                if agent_id and task_id in self.task_assignments[agent_id]:
                    self.task_assignments[agent_id].remove(task_id)
        
        return results
    
    async def _quantum_task_assignment(
        self, 
        task_ids: List[str], 
        tasks: Dict[str, Task]
    ) -> Dict[str, str]:
        """Assign tasks to agents using quantum-inspired optimization."""
        if not task_ids or not self.agents:
            return {}
        
        # Get available agents
        available_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.state == AgentState.IDLE
        ]
        
        if not available_agents:
            # Use load balancing for busy agents
            available_agents = list(self.agents.keys())
        
        # Calculate quantum assignment matrix
        assignment_matrix = self._calculate_assignment_matrix(task_ids, available_agents, tasks)
        
        # Solve assignment problem using quantum-inspired algorithm
        assignments = await self._solve_quantum_assignment(
            task_ids, available_agents, assignment_matrix
        )
        
        logger.info(f"Quantum assignment: {len(assignments)} tasks assigned")
        return assignments
    
    def _calculate_assignment_matrix(
        self,
        task_ids: List[str],
        agent_ids: List[str], 
        tasks: Dict[str, Task]
    ) -> np.ndarray:
        """Calculate quantum assignment cost matrix."""
        n_tasks = len(task_ids)
        n_agents = len(agent_ids)
        
        # Initialize cost matrix
        cost_matrix = np.zeros((n_tasks, n_agents))
        
        for i, task_id in enumerate(task_ids):
            task = tasks[task_id]
            
            for j, agent_id in enumerate(agent_ids):
                agent = self.agents[agent_id]
                
                # Base cost (lower is better)
                base_cost = 1.0
                
                # Load factor cost
                load_cost = agent.get_load_factor() * 0.5
                
                # Capability matching bonus
                capability_bonus = self._calculate_capability_match(task, agent)
                
                # Quantum efficiency bonus
                efficiency_bonus = agent.quantum_efficiency * 0.3
                
                # Entanglement bonus
                entanglement_bonus = self._calculate_entanglement_bonus(task_id, agent_id, tasks)
                
                total_cost = base_cost + load_cost - capability_bonus - efficiency_bonus - entanglement_bonus
                cost_matrix[i][j] = max(0.1, total_cost)  # Ensure positive costs
        
        return cost_matrix
    
    def _calculate_capability_match(self, task: Task, agent: TaskAgent) -> float:
        """Calculate how well agent capabilities match task requirements."""
        required_capabilities = task.metadata.get("required_capabilities", {})
        if not required_capabilities:
            return 0.0
        
        match_score = 0.0
        for capability, required_level in required_capabilities.items():
            agent_level = agent.capabilities.get(capability, 0)
            if agent_level >= required_level:
                match_score += min(1.0, agent_level / required_level)
        
        return match_score / len(required_capabilities)
    
    def _calculate_entanglement_bonus(self, task_id: str, agent_id: str, tasks: Dict[str, Task]) -> float:
        """Calculate bonus for quantum entanglement effects."""
        task = tasks[task_id]
        agent = self.agents[agent_id]
        
        entanglement_bonus = 0.0
        
        # Check if task is entangled with tasks already assigned to this agent
        for entangled_task_id in task.entangled_tasks:
            if entangled_task_id in self.active_tasks:
                assigned_agent = self.active_tasks[entangled_task_id]
                if assigned_agent == agent_id:
                    entanglement_bonus += 0.2
        
        # Check agent entanglement network
        for partner_id in agent.entanglement_partners:
            if partner_id in self.active_tasks.values():
                entanglement_bonus += 0.1
        
        return entanglement_bonus
    
    async def _solve_quantum_assignment(
        self,
        task_ids: List[str],
        agent_ids: List[str],
        cost_matrix: np.ndarray
    ) -> Dict[str, str]:
        """Solve assignment problem using quantum-inspired algorithm."""
        n_tasks, n_agents = cost_matrix.shape
        
        if n_tasks == 0 or n_agents == 0:
            return {}
        
        # Use quantum superposition to explore multiple assignment possibilities
        assignments = {}
        
        if n_tasks <= n_agents:
            # More agents than tasks - use Hungarian-style assignment
            assignments = self._hungarian_assignment(task_ids, agent_ids, cost_matrix)
        else:
            # More tasks than agents - use load balancing
            assignments = await self._load_balanced_assignment(task_ids, agent_ids, cost_matrix)
        
        return assignments
    
    def _hungarian_assignment(
        self, 
        task_ids: List[str], 
        agent_ids: List[str], 
        cost_matrix: np.ndarray
    ) -> Dict[str, str]:
        """Simplified Hungarian algorithm for optimal assignment."""
        assignments = {}
        
        # Greedy assignment for simplicity (in real implementation, use scipy.optimize.linear_sum_assignment)
        used_agents = set()
        
        # Sort tasks by minimum cost
        task_costs = [(i, np.min(cost_matrix[i])) for i in range(len(task_ids))]
        task_costs.sort(key=lambda x: x[1])
        
        for task_idx, _ in task_costs:
            # Find best available agent
            best_agent_idx = None
            best_cost = float('inf')
            
            for agent_idx in range(len(agent_ids)):
                if agent_idx not in used_agents and cost_matrix[task_idx][agent_idx] < best_cost:
                    best_cost = cost_matrix[task_idx][agent_idx]
                    best_agent_idx = agent_idx
            
            if best_agent_idx is not None:
                assignments[task_ids[task_idx]] = agent_ids[best_agent_idx]
                used_agents.add(best_agent_idx)
        
        return assignments
    
    async def _load_balanced_assignment(
        self,
        task_ids: List[str],
        agent_ids: List[str], 
        cost_matrix: np.ndarray
    ) -> Dict[str, str]:
        """Load-balanced assignment when there are more tasks than agents."""
        assignments = {}
        agent_loads = {agent_id: 0 for agent_id in agent_ids}
        
        # Sort tasks by priority
        task_priorities = []
        for i, task_id in enumerate(task_ids):
            # Use minimum cost as priority indicator
            priority = -np.min(cost_matrix[i])
            task_priorities.append((i, task_id, priority))
        
        task_priorities.sort(key=lambda x: x[2], reverse=True)
        
        # Assign tasks to least loaded suitable agents
        for task_idx, task_id, _ in task_priorities:
            # Find agent with minimum combined cost and load
            best_agent_idx = None
            best_score = float('inf')
            
            for agent_idx, agent_id in enumerate(agent_ids):
                cost = cost_matrix[task_idx][agent_idx]
                load = agent_loads[agent_id]
                combined_score = cost + load * 0.5
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_agent_idx = agent_idx
            
            if best_agent_idx is not None:
                agent_id = agent_ids[best_agent_idx]
                assignments[task_id] = agent_id
                agent_loads[agent_id] += 1
        
        return assignments
    
    async def _execute_single_task(
        self, 
        task: Task, 
        agent: TaskAgent, 
        task_id: str
    ) -> Dict[str, Any]:
        """Execute a single task on a specific agent."""
        start_time = datetime.now()
        
        try:
            # Assign task to agent
            success = await agent.assign_task(task)
            if not success:
                raise Exception(f"Agent {agent.agent_id} could not accept task")
            
            # Wait for task completion
            timeout = task.estimated_duration.total_seconds() * 2  # 2x timeout
            timeout = min(timeout, self.config.task_timeout_seconds)
            
            # Poll for completion
            while task.status in [TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING]:
                await asyncio.sleep(0.1)
                
                if (datetime.now() - start_time).total_seconds() > timeout:
                    task.status = TaskStatus.FAILED
                    raise asyncio.TimeoutError(f"Task {task_id} timed out after {timeout}s")
            
            finish_time = task.actual_finish or datetime.now()
            success = task.status == TaskStatus.COMPLETED
            
            return {
                "success": success,
                "start_time": start_time,
                "finish_time": finish_time,
                "agent_id": agent.agent_id,
                "execution_time": (finish_time - start_time).total_seconds()
            }
            
        except Exception as e:
            finish_time = datetime.now()
            return {
                "success": False,
                "error": str(e),
                "start_time": start_time,
                "finish_time": finish_time,
                "agent_id": agent.agent_id,
                "execution_time": (finish_time - start_time).total_seconds()
            }
    
    async def _apply_entanglement_effects(
        self, 
        task_ids: List[str], 
        results: Dict[str, Dict[str, Any]]
    ):
        """Apply quantum entanglement effects between related tasks."""
        # Update agent entanglement based on task outcomes
        for task_id in task_ids:
            result = results.get(task_id, {})
            agent_id = result.get("agent_id")
            
            if agent_id and result.get("success"):
                agent = self.agents.get(agent_id)
                if agent:
                    # Increase quantum efficiency for successful tasks
                    agent.quantum_efficiency = min(1.0, agent.quantum_efficiency * 1.01)
                    
                    # Strengthen entanglements
                    for partner_id in agent.entanglement_partners:
                        partner = self.agents.get(partner_id)
                        if partner:
                            partner.quantum_efficiency = min(1.0, partner.quantum_efficiency * 1.005)
    
    async def _handle_agent_removal(self, agent_id: str):
        """Handle ongoing tasks when an agent is removed."""
        # Find tasks assigned to the removed agent
        affected_tasks = [
            task_id for task_id, assigned_agent in self.active_tasks.items()
            if assigned_agent == agent_id
        ]
        
        # Reassign or cancel affected tasks
        for task_id in affected_tasks:
            logger.warning(f"Task {task_id} affected by agent {agent_id} removal")
            # In a real implementation, would reassign to other agents
            self.active_tasks.pop(task_id, None)
    
    def _update_coordination_metrics(self, results: Dict[str, Dict[str, Any]]):
        """Update coordination performance metrics."""
        if not results:
            return
        
        successful = len([r for r in results.values() if r.get("success", False)])
        failed = len(results) - successful
        
        self.coordination_metrics["successful_completions"] += successful
        self.coordination_metrics["failed_executions"] += failed
        
        # Calculate load balance efficiency
        if self.agents:
            load_factors = [agent.get_load_factor() for agent in self.agents.values()]
            load_variance = np.var(load_factors) if len(load_factors) > 1 else 0
            self.coordination_metrics["load_balance_efficiency"] = max(0.1, 1.0 - load_variance)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current coordination status."""
        agent_statuses = {
            agent_id: agent.get_status() 
            for agent_id, agent in self.agents.items()
        }
        
        return {
            "total_agents": len(self.agents),
            "active_tasks": len(self.active_tasks),
            "coordination_metrics": self.coordination_metrics,
            "agent_statuses": agent_statuses,
            "load_balancer_status": self.load_balancer.get_status(),
            "entanglement_network": {
                agent_id: list(entangled) 
                for agent_id, entangled in self.entanglement_network.items()
            }
        }
    
    async def create_entanglement(self, agent_id1: str, agent_id2: str) -> bool:
        """Create quantum entanglement between two agents."""
        if agent_id1 not in self.agents or agent_id2 not in self.agents:
            return False
        
        agent1 = self.agents[agent_id1]
        agent2 = self.agents[agent_id2]
        
        agent1.add_entanglement(agent_id2)
        agent2.add_entanglement(agent_id1)
        
        self.entanglement_network[agent_id1].add(agent_id2)
        self.entanglement_network[agent_id2].add(agent_id1)
        
        logger.info(f"Created entanglement between agents {agent_id1} and {agent_id2}")
        return True
    
    async def break_entanglement(self, agent_id1: str, agent_id2: str) -> bool:
        """Break quantum entanglement between two agents."""
        if agent_id1 not in self.agents or agent_id2 not in self.agents:
            return False
        
        agent1 = self.agents[agent_id1]
        agent2 = self.agents[agent_id2]
        
        agent1.remove_entanglement(agent_id2)
        agent2.remove_entanglement(agent_id1)
        
        self.entanglement_network[agent_id1].discard(agent_id2)
        self.entanglement_network[agent_id2].discard(agent_id1)
        
        logger.info(f"Broke entanglement between agents {agent_id1} and {agent_id2}")
        return True