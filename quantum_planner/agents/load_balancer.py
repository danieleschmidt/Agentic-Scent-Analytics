"""Load balancing system for agent coordination."""

import logging
from typing import Dict, List, Callable, Optional, Any
from collections import defaultdict
import statistics
import numpy as np


logger = logging.getLogger(__name__)


class LoadBalancer:
    """Quantum-inspired load balancer for task agents."""
    
    def __init__(self):
        self.agents: Dict[str, Callable[[], float]] = {}  # agent_id -> load_factor_func
        self.load_history: Dict[str, List[float]] = defaultdict(list)
        self.rebalance_threshold = 0.3  # Load imbalance threshold
        self.history_size = 100
        
        logger.info("LoadBalancer initialized")
    
    def register_agent(self, agent_id: str, load_factor_func: Callable[[], float]):
        """Register an agent for load balancing."""
        self.agents[agent_id] = load_factor_func
        self.load_history[agent_id] = []
        logger.info(f"Registered agent {agent_id} for load balancing")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from load balancing."""
        self.agents.pop(agent_id, None)
        self.load_history.pop(agent_id, None)
        logger.info(f"Unregistered agent {agent_id} from load balancing")
    
    def get_least_loaded_agent(self) -> Optional[str]:
        """Get the agent with the lowest current load."""
        if not self.agents:
            return None
        
        current_loads = self._get_current_loads()
        if not current_loads:
            return None
        
        return min(current_loads.items(), key=lambda x: x[1])[0]
    
    def get_load_balanced_agents(self, count: int) -> List[str]:
        """Get specified number of agents with balanced load distribution."""
        if not self.agents or count <= 0:
            return []
        
        current_loads = self._get_current_loads()
        if not current_loads:
            return []
        
        # Sort agents by load (ascending)
        sorted_agents = sorted(current_loads.items(), key=lambda x: x[1])
        
        # Select agents with round-robin and load consideration
        selected = []
        available_agents = list(sorted_agents)
        
        for i in range(min(count, len(available_agents))):
            # Quantum-inspired selection with preference for lower loads
            weights = self._calculate_selection_weights([load for _, load in available_agents])
            
            if weights:
                # Weighted random selection
                selected_idx = np.random.choice(len(available_agents), p=weights)
                selected.append(available_agents[selected_idx][0])
                available_agents.pop(selected_idx)
            
        return selected
    
    def _get_current_loads(self) -> Dict[str, float]:
        """Get current load factors for all agents."""
        current_loads = {}
        
        for agent_id, load_func in self.agents.items():
            try:
                load = load_func()
                current_loads[agent_id] = load
                
                # Update load history
                self.load_history[agent_id].append(load)
                if len(self.load_history[agent_id]) > self.history_size:
                    self.load_history[agent_id].pop(0)
                    
            except Exception as e:
                logger.error(f"Failed to get load for agent {agent_id}: {e}")
                current_loads[agent_id] = float('inf')  # Treat as overloaded
        
        return current_loads
    
    def _calculate_selection_weights(self, loads: List[float]) -> List[float]:
        """Calculate selection weights based on inverse load."""
        if not loads:
            return []
        
        # Quantum-inspired weight calculation
        # Higher weights for lower loads
        max_load = max(loads) if loads else 1.0
        inverse_loads = [max_load - load + 0.1 for load in loads]  # Add small constant
        
        # Normalize to probabilities
        total = sum(inverse_loads)
        weights = [w / total for w in inverse_loads] if total > 0 else [1.0 / len(loads)] * len(loads)
        
        return weights
    
    def check_rebalancing_needed(self) -> bool:
        """Check if load rebalancing is needed."""
        current_loads = self._get_current_loads()
        if len(current_loads) < 2:
            return False
        
        loads = list(current_loads.values())
        mean_load = statistics.mean(loads)
        std_load = statistics.stdev(loads) if len(loads) > 1 else 0
        
        # Check if standard deviation exceeds threshold
        coefficient_of_variation = std_load / mean_load if mean_load > 0 else 0
        
        return coefficient_of_variation > self.rebalance_threshold
    
    def get_rebalancing_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for load rebalancing."""
        if not self.check_rebalancing_needed():
            return []
        
        current_loads = self._get_current_loads()
        if len(current_loads) < 2:
            return []
        
        loads = list(current_loads.items())
        loads.sort(key=lambda x: x[1])  # Sort by load
        
        suggestions = []
        
        # Suggest moving work from high-load to low-load agents
        high_load_agents = loads[-len(loads)//3:]  # Top third
        low_load_agents = loads[:len(loads)//3]    # Bottom third
        
        for high_agent_id, high_load in high_load_agents:
            for low_agent_id, low_load in low_load_agents:
                if high_load - low_load > self.rebalance_threshold:
                    suggestions.append({
                        "from_agent": high_agent_id,
                        "to_agent": low_agent_id,
                        "load_difference": high_load - low_load,
                        "priority": "high" if high_load - low_load > 2 * self.rebalance_threshold else "medium"
                    })
        
        return suggestions
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        current_loads = self._get_current_loads()
        
        if not current_loads:
            return {"status": "no_agents"}
        
        loads = list(current_loads.values())
        
        # Calculate statistics
        stats = {
            "total_agents": len(current_loads),
            "average_load": statistics.mean(loads),
            "median_load": statistics.median(loads),
            "min_load": min(loads),
            "max_load": max(loads),
            "load_std": statistics.stdev(loads) if len(loads) > 1 else 0,
            "rebalancing_needed": self.check_rebalancing_needed(),
            "agent_loads": current_loads
        }
        
        # Calculate load balance efficiency
        if stats["average_load"] > 0:
            coefficient_of_variation = stats["load_std"] / stats["average_load"]
            stats["balance_efficiency"] = max(0, 1 - coefficient_of_variation)
        else:
            stats["balance_efficiency"] = 1.0
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get current load balancer status."""
        return {
            "registered_agents": len(self.agents),
            "load_statistics": self.get_load_statistics(),
            "rebalancing_suggestions": self.get_rebalancing_suggestions(),
            "configuration": {
                "rebalance_threshold": self.rebalance_threshold,
                "history_size": self.history_size
            }
        }