#!/usr/bin/env python3
"""
Adaptive Learning System for Autonomous Industrial AI
Implements continuous learning, adaptation, and evolution capabilities
for manufacturing quality control and process optimization.
"""

import asyncio
import numpy as np
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics
import pickle
import threading
from pathlib import Path

from .exceptions import AgenticScentError
from .quantum_intelligence import QuantumIntelligenceFramework


class LearningMode(Enum):
    """Learning modes for the adaptive system."""
    PASSIVE = "passive"           # Learn from observations only
    ACTIVE = "active"             # Actively seek learning opportunities
    REINFORCEMENT = "reinforcement"  # Learn from rewards and penalties
    EVOLUTIONARY = "evolutionary"   # Evolve through genetic algorithms
    QUANTUM = "quantum"           # Quantum-inspired learning


class AdaptationStrategy(Enum):
    """Strategies for system adaptation."""
    CONSERVATIVE = "conservative"  # Small, gradual changes
    AGGRESSIVE = "aggressive"      # Rapid, large changes
    BALANCED = "balanced"          # Balanced approach
    CONTEXTUAL = "contextual"      # Context-dependent adaptation
    QUANTUM_COHERENT = "quantum_coherent"  # Quantum-coherent adaptation


@dataclass
class LearningExperience:
    """Represents a single learning experience."""
    timestamp: datetime
    context: Dict[str, Any]
    action_taken: Dict[str, Any]
    outcome: Dict[str, Any]
    reward: float
    confidence: float
    learning_mode: LearningMode
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationRule:
    """Rule for system adaptation."""
    condition: str                 # Condition triggering adaptation
    action: str                   # Adaptation action to take
    parameters: Dict[str, Any]    # Action parameters
    confidence_threshold: float   # Minimum confidence to apply rule
    success_rate: float = 0.0     # Historical success rate
    usage_count: int = 0          # Number of times applied


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_speed: float
    energy_efficiency: float
    adaptation_success_rate: float
    learning_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


class ExperienceBuffer:
    """Buffer for storing and managing learning experiences."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.indices_by_context = defaultdict(list)
        self.indices_by_outcome = defaultdict(list)
        
    def add_experience(self, experience: LearningExperience):
        """Add a new learning experience."""
        index = len(self.experiences)
        self.experiences.append(experience)
        
        # Index by context for fast retrieval
        for key, value in experience.context.items():
            context_key = f"{key}:{value}"
            self.indices_by_context[context_key].append(index)
            
        # Index by outcome
        outcome_key = str(experience.outcome.get('result', 'unknown'))
        self.indices_by_outcome[outcome_key].append(index)
        
    def get_similar_experiences(self, context: Dict[str, Any], limit: int = 10) -> List[LearningExperience]:
        """Retrieve experiences with similar context."""
        similar_indices = set()
        
        for key, value in context.items():
            context_key = f"{key}:{value}"
            similar_indices.update(self.indices_by_context.get(context_key, []))
            
        # Return most recent similar experiences
        similar_experiences = []
        for idx in sorted(similar_indices, reverse=True)[:limit]:
            if idx < len(self.experiences):
                similar_experiences.append(self.experiences[idx])
                
        return similar_experiences
    
    def get_successful_experiences(self, threshold: float = 0.7) -> List[LearningExperience]:
        """Get experiences with high success rates."""
        return [exp for exp in self.experiences if exp.reward > threshold]
    
    def get_recent_experiences(self, hours: int = 24) -> List[LearningExperience]:
        """Get experiences from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [exp for exp in self.experiences if exp.timestamp > cutoff]


class PatternRecognition:
    """Advanced pattern recognition for learning optimization."""
    
    def __init__(self):
        self.patterns = {}
        self.pattern_success_rates = {}
        self.temporal_patterns = deque(maxlen=1000)
        
    async def detect_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Detect patterns in learning experiences."""
        patterns = {
            'temporal': await self._detect_temporal_patterns(experiences),
            'contextual': await self._detect_contextual_patterns(experiences),
            'outcome': await self._detect_outcome_patterns(experiences),
            'adaptation': await self._detect_adaptation_patterns(experiences)
        }
        
        return patterns
    
    async def _detect_temporal_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Detect temporal patterns in experiences."""
        if len(experiences) < 10:
            return {}
            
        # Analyze time-based trends
        timestamps = [exp.timestamp for exp in experiences]
        rewards = [exp.reward for exp in experiences]
        
        # Calculate moving averages
        window_size = min(10, len(rewards))
        moving_avg = []
        for i in range(window_size - 1, len(rewards)):
            avg = statistics.mean(rewards[i - window_size + 1:i + 1])
            moving_avg.append(avg)
            
        # Detect trends
        if len(moving_avg) >= 2:
            trend = "improving" if moving_avg[-1] > moving_avg[0] else "declining"
        else:
            trend = "stable"
            
        return {
            'trend': trend,
            'moving_average': moving_avg[-5:] if moving_avg else [],
            'volatility': statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        }
    
    async def _detect_contextual_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Detect patterns in context-outcome relationships."""
        context_outcomes = defaultdict(list)
        
        for exp in experiences:
            context_signature = self._create_context_signature(exp.context)
            context_outcomes[context_signature].append(exp.reward)
            
        # Find best-performing contexts
        best_contexts = {}
        for context_sig, rewards in context_outcomes.items():
            if len(rewards) >= 3:  # Minimum sample size
                avg_reward = statistics.mean(rewards)
                best_contexts[context_sig] = {
                    'average_reward': avg_reward,
                    'sample_count': len(rewards),
                    'consistency': 1.0 - (statistics.stdev(rewards) / max(0.1, avg_reward))
                }
                
        return best_contexts
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature string for context comparison."""
        # Simplified signature - could be enhanced with more sophisticated hashing
        items = sorted(context.items())
        return "|".join(f"{k}:{v}" for k, v in items if isinstance(v, (str, int, float, bool)))
    
    async def _detect_outcome_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Detect patterns in outcomes and their predictors."""
        if len(experiences) < 5:
            return {}
            
        # Analyze outcome distribution
        outcomes = [exp.outcome.get('result', 'unknown') for exp in experiences]
        outcome_counts = defaultdict(int)
        for outcome in outcomes:
            outcome_counts[outcome] += 1
            
        # Find most common outcomes
        total_outcomes = len(outcomes)
        outcome_patterns = {
            outcome: {
                'frequency': count / total_outcomes,
                'count': count
            }
            for outcome, count in outcome_counts.items()
        }
        
        return outcome_patterns
    
    async def _detect_adaptation_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Detect patterns in adaptation effectiveness."""
        adaptation_results = defaultdict(list)
        
        for exp in experiences:
            if 'adaptation' in exp.metadata:
                adaptation_type = exp.metadata['adaptation'].get('type', 'unknown')
                adaptation_results[adaptation_type].append(exp.reward)
                
        # Calculate adaptation effectiveness
        adaptation_effectiveness = {}
        for adapt_type, rewards in adaptation_results.items():
            if len(rewards) >= 2:
                effectiveness = statistics.mean(rewards)
                adaptation_effectiveness[adapt_type] = {
                    'effectiveness': effectiveness,
                    'sample_count': len(rewards),
                    'variance': statistics.variance(rewards) if len(rewards) > 1 else 0.0
                }
                
        return adaptation_effectiveness


class GeneticEvolution:
    """Genetic algorithm for evolving system parameters."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        
    async def evolve_parameters(self, 
                               parameter_space: Dict[str, Tuple[float, float]],
                               fitness_function: Callable,
                               generations: int = 20) -> Dict[str, float]:
        """Evolve optimal parameters using genetic algorithm."""
        
        # Initialize population if empty
        if not self.population:
            self.population = self._initialize_population(parameter_space)
            
        best_fitness = float('-inf')
        
        for gen in range(generations):
            # Evaluate fitness for each individual
            fitnesses = []
            for individual in self.population:
                try:
                    if asyncio.iscoroutinefunction(fitness_function):
                        fitness = await fitness_function(individual)
                    else:
                        fitness = fitness_function(individual)
                    fitnesses.append(fitness)
                except Exception as e:
                    logging.error(f"Error evaluating fitness: {e}")
                    fitnesses.append(float('-inf'))
                    
            # Track best individual
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                self.best_individual = self.population[max_fitness_idx].copy()
                
            self.fitness_history.append(best_fitness)
            
            # Selection, crossover, and mutation
            new_population = []
            
            # Elitism - keep best individuals
            elite_count = max(1, self.population_size // 10)
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
                
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(self.population, fitnesses)
                parent2 = self._tournament_selection(self.population, fitnesses)
                
                # Crossover
                child = self._crossover(parent1, parent2, parameter_space)
                
                # Mutation
                child = self._mutate(child, parameter_space)
                
                new_population.append(child)
                
            self.population = new_population
            self.generation += 1
            
            # Small delay for async cooperation
            if gen % 5 == 0:
                await asyncio.sleep(0.001)
                
        return self.best_individual or {}
    
    def _initialize_population(self, parameter_space: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in parameter_space.items():
                individual[param] = np.random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def _tournament_selection(self, population: List[Dict[str, float]], fitnesses: List[float]) -> Dict[str, float]:
        """Tournament selection for parent selection."""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float], 
                   parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Crossover between two parents."""
        child = {}
        for param in parameter_space.keys():
            if np.random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def _mutate(self, individual: Dict[str, float], 
                parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Mutate an individual."""
        mutated = individual.copy()
        for param, (min_val, max_val) in parameter_space.items():
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                sigma = (max_val - min_val) * 0.1  # 10% of range
                mutation = np.random.normal(0, sigma)
                mutated[param] = np.clip(mutated[param] + mutation, min_val, max_val)
        return mutated


class AdaptiveLearningSystem:
    """
    Complete adaptive learning system for autonomous industrial AI.
    Integrates experience buffer, pattern recognition, genetic evolution,
    and quantum intelligence for continuous improvement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Core components
        self.experience_buffer = ExperienceBuffer(
            max_size=self.config.get('experience_buffer_size', 10000)
        )
        self.pattern_recognition = PatternRecognition()
        self.genetic_evolution = GeneticEvolution(
            population_size=self.config.get('population_size', 30),
            mutation_rate=self.config.get('mutation_rate', 0.1)
        )
        
        # Quantum intelligence integration
        self.quantum_intelligence = None
        if self.config.get('enable_quantum', True):
            try:
                self.quantum_intelligence = QuantumIntelligenceFramework(self.config)
            except Exception as e:
                self.logger.warning(f"Quantum intelligence not available: {e}")
                
        # Learning state
        self.learning_mode = LearningMode.BALANCED
        self.adaptation_strategy = AdaptationStrategy.BALANCED
        self.adaptation_rules: List[AdaptationRule] = []
        self.performance_history: List[PerformanceMetrics] = []
        
        # Learning parameters
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.exploration_rate = self.config.get('exploration_rate', 0.1)
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.1)
        
        # Persistence
        self.save_path = Path(self.config.get('save_path', './adaptive_learning_state.pkl'))
        
        # Background learning thread
        self.learning_active = False
        self.learning_thread = None
        
    async def initialize(self):
        """Initialize the adaptive learning system."""
        self.logger.info("Initializing Adaptive Learning System")
        
        # Initialize quantum intelligence if available
        if self.quantum_intelligence:
            await self.quantum_intelligence.initialize()
            
        # Load previous state if available
        await self._load_state()
        
        # Start background learning
        await self._start_background_learning()
        
        self.logger.info("Adaptive Learning System initialized")
        
    async def _start_background_learning(self):
        """Start background learning thread."""
        self.learning_active = True
        self.learning_thread = threading.Thread(
            target=self._background_learning_loop,
            daemon=True
        )
        self.learning_thread.start()
        
    def _background_learning_loop(self):
        """Background learning loop for continuous adaptation."""
        while self.learning_active:
            try:
                # Periodic learning and adaptation
                asyncio.run(self._periodic_learning())
                threading.Event().wait(300)  # 5-minute intervals
                
            except Exception as e:
                self.logger.error(f"Background learning error: {e}")
                threading.Event().wait(60)  # Wait 1 minute before retry
                
    async def _periodic_learning(self):
        """Perform periodic learning and adaptation."""
        try:
            # Analyze recent experiences
            recent_experiences = self.experience_buffer.get_recent_experiences(hours=1)
            
            if len(recent_experiences) >= 5:
                # Detect patterns
                patterns = await self.pattern_recognition.detect_patterns(recent_experiences)
                
                # Adapt based on patterns
                await self._adapt_from_patterns(patterns)
                
                # Update adaptation rules
                await self._update_adaptation_rules(recent_experiences)
                
                # Evolve parameters if needed
                if len(recent_experiences) >= 20:
                    await self._evolutionary_adaptation(recent_experiences)
                    
        except Exception as e:
            self.logger.error(f"Periodic learning failed: {e}")
            
    async def learn_from_experience(self, 
                                  context: Dict[str, Any],
                                  action: Dict[str, Any],
                                  outcome: Dict[str, Any],
                                  reward: float,
                                  confidence: float = 1.0,
                                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Learn from a new experience.
        
        Args:
            context: Situation context
            action: Action taken
            outcome: Result of the action
            reward: Reward/penalty for the outcome
            confidence: Confidence in the experience quality
            metadata: Additional metadata
            
        Returns:
            Whether learning was successful
        """
        try:
            # Create experience record
            experience = LearningExperience(
                timestamp=datetime.now(),
                context=context,
                action_taken=action,
                outcome=outcome,
                reward=reward,
                confidence=confidence,
                learning_mode=self.learning_mode,
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.experience_buffer.add_experience(experience)
            
            # Immediate learning if quantum intelligence is available
            if self.quantum_intelligence:
                await self._quantum_learning(experience)
                
            # Check for immediate adaptation needs
            if abs(reward) > self.adaptation_threshold:
                await self._immediate_adaptation(experience)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Learning from experience failed: {e}")
            return False
            
    async def _quantum_learning(self, experience: LearningExperience):
        """Apply quantum learning to experience."""
        if not self.quantum_intelligence:
            return
            
        try:
            # Prepare problem data for quantum processing
            problem_data = {
                'context': experience.context,
                'action': experience.action_taken,
                'outcome': experience.outcome,
                'reward': experience.reward
            }
            
            # Process with quantum intelligence
            quantum_insight = await self.quantum_intelligence.process_intelligent_decision(
                problem_data
            )
            
            # Update learning parameters based on quantum insight
            if quantum_insight.get('confidence', 0) > 0.7:
                if experience.reward > 0.5:
                    self.learning_rate = min(0.1, self.learning_rate * 1.05)
                else:
                    self.learning_rate = max(0.001, self.learning_rate * 0.95)
                    
        except Exception as e:
            self.logger.error(f"Quantum learning failed: {e}")
            
    async def _immediate_adaptation(self, experience: LearningExperience):
        """Apply immediate adaptation based on experience."""
        try:
            # Find similar past experiences
            similar_experiences = self.experience_buffer.get_similar_experiences(
                experience.context, limit=5
            )
            
            if len(similar_experiences) >= 2:
                # Calculate trend in similar contexts
                rewards = [exp.reward for exp in similar_experiences]
                recent_rewards = rewards[-3:]  # Most recent 3
                
                if len(recent_rewards) >= 2:
                    trend = statistics.mean(recent_rewards)
                    
                    # Adapt strategy based on trend
                    if trend < -0.3:  # Poor performance
                        await self._switch_adaptation_strategy()
                    elif trend > 0.5:  # Good performance
                        # Reinforce current strategy
                        pass
                        
        except Exception as e:
            self.logger.error(f"Immediate adaptation failed: {e}")
            
    async def _switch_adaptation_strategy(self):
        """Switch to a different adaptation strategy."""
        strategies = list(AdaptationStrategy)
        current_idx = strategies.index(self.adaptation_strategy)
        next_idx = (current_idx + 1) % len(strategies)
        
        old_strategy = self.adaptation_strategy
        self.adaptation_strategy = strategies[next_idx]
        
        self.logger.info(f"Switched adaptation strategy from {old_strategy.value} to {self.adaptation_strategy.value}")
        
    async def _adapt_from_patterns(self, patterns: Dict[str, Any]):
        """Adapt system based on detected patterns."""
        try:
            # Temporal pattern adaptation
            temporal_patterns = patterns.get('temporal', {})
            if temporal_patterns.get('trend') == 'declining':
                self.exploration_rate = min(0.3, self.exploration_rate * 1.2)
            elif temporal_patterns.get('trend') == 'improving':
                self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
                
            # Contextual pattern adaptation
            contextual_patterns = patterns.get('contextual', {})
            for context_sig, pattern_info in contextual_patterns.items():
                if pattern_info['average_reward'] > 0.8 and pattern_info['sample_count'] >= 5:
                    # Create or update adaptation rule for high-performing contexts
                    await self._create_adaptation_rule(context_sig, pattern_info)
                    
        except Exception as e:
            self.logger.error(f"Pattern adaptation failed: {e}")
            
    async def _create_adaptation_rule(self, context_signature: str, pattern_info: Dict[str, Any]):
        """Create an adaptation rule based on successful patterns."""
        # Check if rule already exists
        existing_rule = None
        for rule in self.adaptation_rules:
            if context_signature in rule.condition:
                existing_rule = rule
                break
                
        if existing_rule:
            # Update existing rule
            existing_rule.success_rate = pattern_info['average_reward']
            existing_rule.usage_count += 1
        else:
            # Create new rule
            new_rule = AdaptationRule(
                condition=f"context_matches:{context_signature}",
                action="apply_successful_pattern",
                parameters={'pattern_info': pattern_info},
                confidence_threshold=0.7,
                success_rate=pattern_info['average_reward']
            )
            self.adaptation_rules.append(new_rule)
            
    async def _update_adaptation_rules(self, experiences: List[LearningExperience]):
        """Update adaptation rules based on recent experiences."""
        for rule in self.adaptation_rules:
            # Find experiences that match this rule's condition
            matching_experiences = []
            for exp in experiences:
                if self._rule_matches_experience(rule, exp):
                    matching_experiences.append(exp)
                    
            if matching_experiences:
                # Update rule success rate
                rewards = [exp.reward for exp in matching_experiences]
                rule.success_rate = statistics.mean(rewards)
                rule.usage_count += len(matching_experiences)
                
    def _rule_matches_experience(self, rule: AdaptationRule, experience: LearningExperience) -> bool:
        """Check if an adaptation rule matches an experience."""
        # Simplified rule matching - could be enhanced with more sophisticated logic
        if "context_matches:" in rule.condition:
            context_sig = rule.condition.split("context_matches:")[1]
            experience_sig = self.pattern_recognition._create_context_signature(experience.context)
            return context_sig == experience_sig
            
        return False
        
    async def _evolutionary_adaptation(self, experiences: List[LearningExperience]):
        """Apply evolutionary adaptation to system parameters."""
        try:
            # Define parameter space for evolution
            parameter_space = {
                'learning_rate': (0.001, 0.1),
                'exploration_rate': (0.01, 0.5),
                'adaptation_threshold': (0.05, 0.3)
            }
            
            # Define fitness function based on recent performance
            def fitness_function(params: Dict[str, float]) -> float:
                # Simulate performance with these parameters
                # In a real implementation, this would test the parameters
                reward_sum = sum(exp.reward for exp in experiences[-10:])
                consistency = 1.0 - statistics.stdev([exp.reward for exp in experiences[-10:]])
                return reward_sum + consistency
                
            # Evolve parameters
            evolved_params = await self.genetic_evolution.evolve_parameters(
                parameter_space, fitness_function, generations=10
            )
            
            # Apply evolved parameters
            if evolved_params:
                self.learning_rate = evolved_params.get('learning_rate', self.learning_rate)
                self.exploration_rate = evolved_params.get('exploration_rate', self.exploration_rate)
                self.adaptation_threshold = evolved_params.get('adaptation_threshold', self.adaptation_threshold)
                
                self.logger.info(f"Applied evolutionary adaptation: {evolved_params}")
                
        except Exception as e:
            self.logger.error(f"Evolutionary adaptation failed: {e}")
            
    async def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status."""
        return {
            'learning_mode': self.learning_mode.value,
            'adaptation_strategy': self.adaptation_strategy.value,
            'parameters': {
                'learning_rate': self.learning_rate,
                'exploration_rate': self.exploration_rate,
                'adaptation_threshold': self.adaptation_threshold
            },
            'experience_buffer': {
                'total_experiences': len(self.experience_buffer.experiences),
                'recent_experiences': len(self.experience_buffer.get_recent_experiences()),
                'successful_experiences': len(self.experience_buffer.get_successful_experiences())
            },
            'adaptation_rules': len(self.adaptation_rules),
            'genetic_evolution': {
                'generation': self.genetic_evolution.generation,
                'best_fitness': max(self.genetic_evolution.fitness_history) if self.genetic_evolution.fitness_history else 0.0
            },
            'quantum_intelligence_active': self.quantum_intelligence is not None,
            'background_learning_active': self.learning_active
        }
        
    async def save_state(self):
        """Save learning system state to disk."""
        try:
            state = {
                'experience_buffer': self.experience_buffer,
                'adaptation_rules': self.adaptation_rules,
                'performance_history': self.performance_history,
                'learning_parameters': {
                    'learning_rate': self.learning_rate,
                    'exploration_rate': self.exploration_rate,
                    'adaptation_threshold': self.adaptation_threshold
                },
                'genetic_evolution_state': {
                    'generation': self.genetic_evolution.generation,
                    'fitness_history': self.genetic_evolution.fitness_history,
                    'best_individual': self.genetic_evolution.best_individual
                }
            }
            
            with open(self.save_path, 'wb') as f:
                pickle.dump(state, f)
                
            self.logger.info(f"Learning state saved to {self.save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save learning state: {e}")
            
    async def _load_state(self):
        """Load learning system state from disk."""
        try:
            if self.save_path.exists():
                with open(self.save_path, 'rb') as f:
                    state = pickle.load(f)
                    
                # Restore state
                self.experience_buffer = state.get('experience_buffer', self.experience_buffer)
                self.adaptation_rules = state.get('adaptation_rules', [])
                self.performance_history = state.get('performance_history', [])
                
                # Restore parameters
                params = state.get('learning_parameters', {})
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.exploration_rate = params.get('exploration_rate', self.exploration_rate)
                self.adaptation_threshold = params.get('adaptation_threshold', self.adaptation_threshold)
                
                # Restore genetic evolution state
                genetic_state = state.get('genetic_evolution_state', {})
                self.genetic_evolution.generation = genetic_state.get('generation', 0)
                self.genetic_evolution.fitness_history = genetic_state.get('fitness_history', [])
                self.genetic_evolution.best_individual = genetic_state.get('best_individual')
                
                self.logger.info(f"Learning state loaded from {self.save_path}")
                
        except Exception as e:
            self.logger.warning(f"Could not load learning state: {e}")
            
    async def shutdown(self):
        """Shutdown the adaptive learning system."""
        self.learning_active = False
        
        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)
            
        # Save current state
        await self.save_state()
        
        self.logger.info("Adaptive Learning System shutdown completed")


# Factory function
def create_adaptive_learning_system(config: Optional[Dict[str, Any]] = None) -> AdaptiveLearningSystem:
    """Create and return an adaptive learning system instance."""
    return AdaptiveLearningSystem(config)


# Learning decorator
def adaptive_learning(learning_config: Optional[Dict[str, Any]] = None):
    """Decorator to add adaptive learning to any function."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            learning_system = create_adaptive_learning_system(learning_config)
            await learning_system.initialize()
            
            try:
                # Execute function
                start_time = datetime.now()
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Learn from execution
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
                
                action = {'executed': func.__name__}
                outcome = {'success': True, 'execution_time': execution_time}
                reward = 1.0 - min(1.0, execution_time)  # Faster execution = higher reward
                
                await learning_system.learn_from_experience(
                    context, action, outcome, reward
                )
                
                return result
                
            except Exception as e:
                # Learn from failure
                context = {'function': func.__name__, 'error_type': type(e).__name__}
                action = {'executed': func.__name__}
                outcome = {'success': False, 'error': str(e)}
                reward = -1.0
                
                await learning_system.learn_from_experience(
                    context, action, outcome, reward
                )
                
                raise
                
            finally:
                await learning_system.shutdown()
                
        return wrapper
    return decorator