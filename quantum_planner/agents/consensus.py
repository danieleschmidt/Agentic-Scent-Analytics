"""Consensus engine for multi-agent decision making."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict


logger = logging.getLogger(__name__)


class ConsensusMethod(Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    QUANTUM_VOTING = "quantum_voting"


class ConsensusEngine:
    """Quantum-inspired consensus system for agent coordination."""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.voting_history: List[Dict[str, Any]] = []
        self.agent_reliability: Dict[str, float] = defaultdict(lambda: 1.0)
        
        logger.info(f"ConsensusEngine initialized with threshold {threshold}")
    
    async def reach_consensus(
        self,
        agents: List[str],
        proposals: Dict[str, Any],
        method: ConsensusMethod = ConsensusMethod.QUANTUM_VOTING,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Reach consensus among agents on proposals."""
        if not agents or not proposals:
            return {"consensus_reached": False, "reason": "no_agents_or_proposals"}
        
        logger.info(f"Seeking consensus among {len(agents)} agents using {method.value}")
        
        # Apply consensus method
        if method == ConsensusMethod.MAJORITY:
            result = self._majority_consensus(agents, proposals)
        elif method == ConsensusMethod.WEIGHTED:
            result = self._weighted_consensus(agents, proposals, weights or {})
        elif method == ConsensusMethod.UNANIMOUS:
            result = self._unanimous_consensus(agents, proposals)
        else:  # QUANTUM_VOTING
            result = await self._quantum_consensus(agents, proposals, weights or {})
        
        # Record voting history
        self._record_consensus(agents, proposals, result, method)
        
        return result
    
    def _majority_consensus(self, agents: List[str], proposals: Dict[str, Any]) -> Dict[str, Any]:
        """Simple majority voting consensus."""
        votes = defaultdict(int)
        
        for agent_id in agents:
            agent_vote = proposals.get(agent_id, {}).get("vote")
            if agent_vote:
                votes[agent_vote] += 1
        
        if not votes:
            return {"consensus_reached": False, "reason": "no_votes"}
        
        # Find majority
        total_votes = sum(votes.values())
        majority_threshold = total_votes / 2
        
        for option, count in votes.items():
            if count > majority_threshold:
                return {
                    "consensus_reached": True,
                    "decision": option,
                    "support_ratio": count / total_votes,
                    "method": "majority",
                    "vote_counts": dict(votes)
                }
        
        return {
            "consensus_reached": False,
            "reason": "no_majority",
            "vote_counts": dict(votes)
        }
    
    def _weighted_consensus(
        self, 
        agents: List[str], 
        proposals: Dict[str, Any], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Weighted voting consensus based on agent weights."""
        vote_weights = defaultdict(float)
        total_weight = 0
        
        for agent_id in agents:
            agent_weight = weights.get(agent_id, 1.0)
            agent_vote = proposals.get(agent_id, {}).get("vote")
            
            if agent_vote:
                vote_weights[agent_vote] += agent_weight
                total_weight += agent_weight
        
        if total_weight == 0:
            return {"consensus_reached": False, "reason": "no_weighted_votes"}
        
        # Find weighted majority
        for option, weight in vote_weights.items():
            support_ratio = weight / total_weight
            if support_ratio >= self.threshold:
                return {
                    "consensus_reached": True,
                    "decision": option,
                    "support_ratio": support_ratio,
                    "method": "weighted",
                    "vote_weights": dict(vote_weights)
                }
        
        return {
            "consensus_reached": False,
            "reason": "insufficient_weighted_support",
            "vote_weights": dict(vote_weights)
        }
    
    def _unanimous_consensus(self, agents: List[str], proposals: Dict[str, Any]) -> Dict[str, Any]:
        """Unanimous consensus requirement."""
        votes = set()
        
        for agent_id in agents:
            agent_vote = proposals.get(agent_id, {}).get("vote")
            if agent_vote:
                votes.add(agent_vote)
        
        if len(votes) == 1:
            decision = list(votes)[0]
            return {
                "consensus_reached": True,
                "decision": decision,
                "support_ratio": 1.0,
                "method": "unanimous"
            }
        
        return {
            "consensus_reached": False,
            "reason": "not_unanimous",
            "vote_options": list(votes)
        }
    
    async def _quantum_consensus(
        self, 
        agents: List[str], 
        proposals: Dict[str, Any], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Quantum-inspired consensus using superposition and interference."""
        logger.info("Applying quantum consensus algorithm")
        
        # Extract vote options and confidences
        vote_data = []
        for agent_id in agents:
            proposal = proposals.get(agent_id, {})
            vote = proposal.get("vote")
            confidence = proposal.get("confidence", 1.0)
            reasoning_strength = len(proposal.get("reasoning", "")) / 100.0  # Normalize
            
            if vote:
                vote_data.append({
                    "agent_id": agent_id,
                    "vote": vote,
                    "confidence": confidence,
                    "reasoning_strength": reasoning_strength,
                    "weight": weights.get(agent_id, 1.0),
                    "reliability": self.agent_reliability[agent_id]
                })
        
        if not vote_data:
            return {"consensus_reached": False, "reason": "no_quantum_votes"}
        
        # Create quantum state representation
        vote_options = list(set(v["vote"] for v in vote_data))
        n_options = len(vote_options)
        
        if n_options == 1:
            # Unanimous case
            return {
                "consensus_reached": True,
                "decision": vote_options[0],
                "support_ratio": 1.0,
                "method": "quantum_unanimous"
            }
        
        # Initialize quantum amplitudes for each option
        quantum_amplitudes = np.zeros(n_options, dtype=complex)
        
        for vote_info in vote_data:
            option_idx = vote_options.index(vote_info["vote"])
            
            # Calculate quantum amplitude
            base_amplitude = vote_info["confidence"] * vote_info["weight"] * vote_info["reliability"]
            
            # Add quantum phase based on reasoning strength
            phase = vote_info["reasoning_strength"] * np.pi
            
            quantum_amplitudes[option_idx] += base_amplitude * np.exp(1j * phase)
        
        # Apply quantum interference
        interfered_amplitudes = self._apply_quantum_interference(quantum_amplitudes)
        
        # Measure quantum state (collapse to classical probabilities)
        probabilities = np.abs(interfered_amplitudes) ** 2
        
        # Normalize probabilities
        total_prob = np.sum(probabilities)
        if total_prob > 0:
            probabilities = probabilities / total_prob
        else:
            probabilities = np.ones(n_options) / n_options
        
        # Find highest probability option
        max_prob_idx = np.argmax(probabilities)
        winning_option = vote_options[max_prob_idx]
        support_ratio = probabilities[max_prob_idx]
        
        # Check if meets consensus threshold
        consensus_reached = support_ratio >= self.threshold
        
        return {
            "consensus_reached": consensus_reached,
            "decision": winning_option if consensus_reached else None,
            "support_ratio": float(support_ratio),
            "method": "quantum_voting",
            "quantum_probabilities": {
                vote_options[i]: float(probabilities[i]) 
                for i in range(n_options)
            },
            "quantum_amplitudes": {
                vote_options[i]: {
                    "amplitude": float(np.abs(interfered_amplitudes[i])),
                    "phase": float(np.angle(interfered_amplitudes[i]))
                }
                for i in range(n_options)
            }
        }
    
    def _apply_quantum_interference(self, amplitudes: np.ndarray) -> np.ndarray:
        """Apply quantum interference effects to amplitudes."""
        n = len(amplitudes)
        
        # Create interference matrix (simplified)
        interference_matrix = np.eye(n, dtype=complex)
        
        # Add constructive/destructive interference based on amplitude similarities
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate interference strength based on amplitude correlation
                amplitude_diff = np.abs(amplitudes[i] - amplitudes[j])
                interference_strength = np.exp(-amplitude_diff) * 0.1
                
                # Apply interference (simplified model)
                interference_matrix[i, j] = interference_strength
                interference_matrix[j, i] = interference_strength.conjugate()
        
        # Apply interference transformation
        interfered = interference_matrix @ amplitudes
        
        return interfered
    
    def _record_consensus(
        self,
        agents: List[str],
        proposals: Dict[str, Any],
        result: Dict[str, Any],
        method: ConsensusMethod
    ):
        """Record consensus voting history."""
        record = {
            "timestamp": logger.handlers[0].formatter.formatTime(logger.makeRecord("", 0, "", 0, "", (), None)) if logger.handlers else "unknown",
            "agents": agents,
            "proposals": proposals,
            "result": result,
            "method": method.value
        }
        
        self.voting_history.append(record)
        
        # Limit history size
        if len(self.voting_history) > 1000:
            self.voting_history.pop(0)
        
        # Update agent reliability based on consensus outcomes
        self._update_agent_reliability(agents, proposals, result)
    
    def _update_agent_reliability(
        self,
        agents: List[str],
        proposals: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Update agent reliability scores based on consensus outcomes."""
        if not result.get("consensus_reached"):
            return
        
        winning_decision = result.get("decision")
        if not winning_decision:
            return
        
        # Update reliability for agents based on their votes
        for agent_id in agents:
            agent_vote = proposals.get(agent_id, {}).get("vote")
            current_reliability = self.agent_reliability[agent_id]
            
            if agent_vote == winning_decision:
                # Reward correct votes
                self.agent_reliability[agent_id] = min(2.0, current_reliability * 1.01)
            else:
                # Penalize incorrect votes (but not too harshly)
                self.agent_reliability[agent_id] = max(0.1, current_reliability * 0.99)
    
    def get_agent_reliability(self, agent_id: str) -> float:
        """Get reliability score for an agent."""
        return self.agent_reliability[agent_id]
    
    def set_agent_reliability(self, agent_id: str, reliability: float):
        """Set reliability score for an agent."""
        self.agent_reliability[agent_id] = max(0.1, min(2.0, reliability))
        logger.info(f"Set agent {agent_id} reliability to {reliability}")
    
    def get_consensus_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get consensus voting history."""
        history = self.voting_history
        if limit:
            history = history[-limit:]
        return history
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get consensus system statistics."""
        if not self.voting_history:
            return {"status": "no_history"}
        
        total_votes = len(self.voting_history)
        successful_consensus = len([h for h in self.voting_history if h["result"].get("consensus_reached")])
        
        method_counts = defaultdict(int)
        for record in self.voting_history:
            method_counts[record["method"]] += 1
        
        return {
            "total_consensus_attempts": total_votes,
            "successful_consensus": successful_consensus,
            "success_rate": successful_consensus / total_votes if total_votes > 0 else 0,
            "method_usage": dict(method_counts),
            "average_support_ratio": np.mean([
                r["result"].get("support_ratio", 0) 
                for r in self.voting_history 
                if r["result"].get("consensus_reached")
            ]) if successful_consensus > 0 else 0,
            "agent_reliabilities": dict(self.agent_reliability)
        }
    
    def reset_history(self):
        """Reset consensus voting history."""
        self.voting_history.clear()
        self.agent_reliability.clear()
        logger.info("Consensus history reset")