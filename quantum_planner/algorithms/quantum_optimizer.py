"""Quantum-inspired optimization algorithms for task scheduling."""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import math


logger = logging.getLogger(__name__)


class QuantumOptimizer:
    """Quantum-inspired optimizer using superposition and entanglement principles."""
    
    def __init__(self, params: Dict[str, Any]):
        self.max_iterations = params.get("max_iterations", 1000)
        self.convergence_threshold = params.get("convergence_threshold", 0.001)
        self.annealing_strength = params.get("annealing_strength", 0.5)
        self.entanglement_factor = params.get("entanglement_factor", 0.2)
        
        # Quantum state variables
        self.quantum_state = None
        self.energy_history = []
        self.best_solution = None
        self.best_energy = float('inf')
        
        logger.info(f"QuantumOptimizer initialized with {self.max_iterations} max iterations")
    
    async def optimize(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main optimization routine using quantum-inspired algorithms."""
        logger.info("Starting quantum optimization process")
        
        n_tasks = task_data["n_tasks"]
        if n_tasks == 0:
            return {}
        
        # Initialize quantum state
        self.quantum_state = self._initialize_quantum_state(n_tasks)
        
        # Run quantum annealing optimization
        for iteration in range(self.max_iterations):
            # Apply quantum gates and evolution
            await self._quantum_evolution_step(task_data, iteration)
            
            # Measure current energy (cost function)
            current_energy = self._calculate_energy(task_data)
            self.energy_history.append(current_energy)
            
            # Update best solution
            if current_energy < self.best_energy:
                self.best_energy = current_energy
                self.best_solution = self.quantum_state.copy()
            
            # Check convergence
            if iteration > 50 and self._check_convergence():
                logger.info(f"Converged after {iteration} iterations")
                break
            
            # Adaptive annealing schedule
            if iteration % 100 == 0:
                await self._adapt_parameters(iteration)
        
        # Extract classical solution from quantum state
        solution = self._extract_solution(task_data)
        
        logger.info(f"Optimization completed with energy {self.best_energy}")
        return solution
    
    def _initialize_quantum_state(self, n_tasks: int) -> np.ndarray:
        """Initialize quantum state with superposition of all possible task orderings."""
        # Create quantum state vector representing task priorities
        state = np.random.random(n_tasks) + 1j * np.random.random(n_tasks)
        
        # Normalize to unit vector (quantum state requirement)
        state = state / np.linalg.norm(state)
        
        return state
    
    async def _quantum_evolution_step(self, task_data: Dict[str, Any], iteration: int):
        """Evolve quantum state using quantum gates and operations."""
        n_tasks = task_data["n_tasks"]
        
        # Apply rotation gates based on task priorities
        priorities = task_data["priorities"]
        for i in range(n_tasks):
            rotation_angle = priorities[i] * math.pi / 4
            self._apply_rotation_gate(i, rotation_angle)
        
        # Apply entanglement operations
        await self._apply_entanglement(task_data)
        
        # Apply quantum annealing
        temperature = self._calculate_temperature(iteration)
        self._apply_annealing(temperature)
    
    def _apply_rotation_gate(self, qubit_index: int, angle: float):
        """Apply rotation gate to specific qubit (task)."""
        rotation_matrix = np.array([
            [np.cos(angle/2), -1j*np.sin(angle/2)],
            [-1j*np.sin(angle/2), np.cos(angle/2)]
        ])
        
        # Apply rotation to the quantum state component
        real_part = self.quantum_state[qubit_index].real
        imag_part = self.quantum_state[qubit_index].imag
        
        state_vector = np.array([real_part, imag_part])
        rotated = rotation_matrix @ state_vector
        
        self.quantum_state[qubit_index] = rotated[0] + 1j * rotated[1]
    
    async def _apply_entanglement(self, task_data: Dict[str, Any]):
        """Apply entanglement operations based on task dependencies."""
        tasks = task_data["tasks"]
        task_indices = task_data["task_indices"]
        
        for task in tasks:
            if task.entangled_tasks:
                task_idx = task_indices[task.id]
                
                for entangled_id in task.entangled_tasks:
                    if entangled_id in task_indices:
                        entangled_idx = task_indices[entangled_id]
                        
                        # Create entanglement by correlating quantum states
                        entanglement_strength = self.entanglement_factor
                        
                        # Apply CNOT-like operation for entanglement
                        control_state = self.quantum_state[task_idx]
                        target_state = self.quantum_state[entangled_idx]
                        
                        # Entangle states
                        new_target = target_state * (1 - entanglement_strength) + \
                                   control_state * entanglement_strength
                        
                        self.quantum_state[entangled_idx] = new_target
    
    def _apply_annealing(self, temperature: float):
        """Apply quantum annealing to escape local minima."""
        noise_strength = self.annealing_strength * temperature
        
        # Add quantum noise proportional to temperature
        noise = np.random.normal(0, noise_strength, len(self.quantum_state)) + \
                1j * np.random.normal(0, noise_strength, len(self.quantum_state))
        
        self.quantum_state += noise
        
        # Renormalize quantum state
        self.quantum_state = self.quantum_state / np.linalg.norm(self.quantum_state)
    
    def _calculate_temperature(self, iteration: int) -> float:
        """Calculate annealing temperature schedule."""
        # Exponential cooling schedule
        initial_temp = 1.0
        cooling_rate = 0.99
        return initial_temp * (cooling_rate ** iteration)
    
    def _calculate_energy(self, task_data: Dict[str, Any]) -> float:
        """Calculate energy (cost) of current quantum state."""
        n_tasks = task_data["n_tasks"]
        priorities = task_data["priorities"]
        durations = task_data["durations"]
        adjacency_matrix = task_data["adjacency_matrix"]
        
        # Convert quantum amplitudes to classical priorities
        classical_priorities = np.abs(self.quantum_state) ** 2
        
        # Cost components
        priority_cost = -np.sum(priorities * classical_priorities)  # Minimize negative priority
        duration_cost = np.sum(durations * classical_priorities)    # Minimize duration
        
        # Dependency violation cost
        dependency_cost = 0
        for i in range(n_tasks):
            for j in range(n_tasks):
                if adjacency_matrix[i][j] > 0:  # j depends on i
                    if classical_priorities[j] > classical_priorities[i]:
                        dependency_cost += 1000  # Heavy penalty for violations
        
        total_energy = priority_cost + 0.1 * duration_cost + dependency_cost
        return total_energy
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.energy_history) < 10:
            return False
        
        recent_energies = self.energy_history[-10:]
        energy_variance = np.var(recent_energies)
        
        return energy_variance < self.convergence_threshold
    
    async def _adapt_parameters(self, iteration: int):
        """Adapt optimization parameters based on progress."""
        progress = iteration / self.max_iterations
        
        # Reduce annealing strength over time
        self.annealing_strength *= 0.99
        
        # Increase entanglement factor if stuck
        if len(self.energy_history) > 100:
            recent_improvement = self.energy_history[-100] - self.energy_history[-1]
            if recent_improvement < 0.01:
                self.entanglement_factor = min(0.5, self.entanglement_factor * 1.1)
    
    def _extract_solution(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract classical solution from quantum state."""
        tasks = task_data["tasks"]
        
        # Convert quantum amplitudes to task priorities
        quantum_priorities = np.abs(self.quantum_state) ** 2
        
        # Create priority assignment
        priority_assignment = {}
        for i, task in enumerate(tasks):
            priority_assignment[task.id] = {
                "quantum_priority": float(quantum_priorities[i]),
                "phase": float(np.angle(self.quantum_state[i])),
                "amplitude": float(np.abs(self.quantum_state[i])),
                "order_hint": float(quantum_priorities[i] * task.priority.value)
            }
        
        return {
            "priority_assignment": priority_assignment,
            "final_energy": self.best_energy,
            "iterations": len(self.energy_history),
            "converged": self._check_convergence(),
            "quantum_state": self.quantum_state.tolist() if self.quantum_state is not None else []
        }


class QuantumGeneticOptimizer(QuantumOptimizer):
    """Quantum-inspired genetic algorithm for task optimization."""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.population_size = params.get("population_size", 50)
        self.mutation_rate = params.get("mutation_rate", 0.1)
        self.crossover_rate = params.get("crossover_rate", 0.8)
        self.population = []
    
    async def optimize(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genetic algorithm with quantum-inspired operations."""
        n_tasks = task_data["n_tasks"]
        if n_tasks == 0:
            return {}
        
        # Initialize population
        self.population = [
            self._initialize_quantum_state(n_tasks) 
            for _ in range(self.population_size)
        ]
        
        for generation in range(self.max_iterations):
            # Evaluate fitness
            fitness_scores = [
                -self._calculate_energy_for_individual(individual, task_data)
                for individual in self.population
            ]
            
            # Track best solution
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > -self.best_energy:
                self.best_energy = -fitness_scores[best_idx]
                self.best_solution = self.population[best_idx].copy()
            
            # Selection and reproduction
            new_population = []
            for _ in range(self.population_size):
                # Tournament selection
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                # Quantum crossover
                offspring = self._quantum_crossover(parent1, parent2)
                
                # Quantum mutation
                offspring = self._quantum_mutation(offspring)
                
                new_population.append(offspring)
            
            self.population = new_population
            
            if generation % 100 == 0:
                logger.info(f"Generation {generation}, best fitness: {max(fitness_scores)}")
        
        self.quantum_state = self.best_solution
        return self._extract_solution(task_data)
    
    def _calculate_energy_for_individual(self, individual: np.ndarray, task_data: Dict[str, Any]) -> float:
        """Calculate energy for a specific individual."""
        original_state = self.quantum_state
        self.quantum_state = individual
        energy = self._calculate_energy(task_data)
        self.quantum_state = original_state
        return energy
    
    def _tournament_selection(self, fitness_scores: List[float]) -> np.ndarray:
        """Tournament selection for genetic algorithm."""
        tournament_size = 3
        tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        winner_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
        return self.population[winner_idx].copy()
    
    def _quantum_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Quantum-inspired crossover operation."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy()
        
        # Quantum superposition crossover
        alpha = np.random.random()
        offspring = alpha * parent1 + (1 - alpha) * parent2
        
        # Normalize quantum state
        offspring = offspring / np.linalg.norm(offspring)
        
        return offspring
    
    def _quantum_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Quantum-inspired mutation operation."""
        if np.random.random() > self.mutation_rate:
            return individual
        
        # Apply random quantum gate rotations
        mutation_strength = 0.1
        for i in range(len(individual)):
            if np.random.random() < 0.3:  # 30% chance to mutate each qubit
                rotation_angle = np.random.normal(0, mutation_strength)
                self._apply_rotation_gate_to_individual(individual, i, rotation_angle)
        
        # Renormalize
        individual = individual / np.linalg.norm(individual)
        return individual
    
    def _apply_rotation_gate_to_individual(self, individual: np.ndarray, qubit_index: int, angle: float):
        """Apply rotation gate to individual in population."""
        rotation_matrix = np.array([
            [np.cos(angle/2), -1j*np.sin(angle/2)],
            [-1j*np.sin(angle/2), np.cos(angle/2)]
        ])
        
        real_part = individual[qubit_index].real
        imag_part = individual[qubit_index].imag
        
        state_vector = np.array([real_part, imag_part])
        rotated = rotation_matrix @ state_vector
        
        individual[qubit_index] = rotated[0] + 1j * rotated[1]