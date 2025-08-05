"""
Multi-agent orchestrator for coordinating intelligent agents.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .base import BaseAgent, AnalysisResult, AgentCapability
from ..sensors.base import SensorReading


class CommunicationProtocol(Enum):
    """Communication protocols between agents."""
    ALERT_ESCALATION = "alert_escalation"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    CONSENSUS_BUILDING = "consensus_building"
    MAINTENANCE_COORDINATION = "maintenance_coordination"


@dataclass
class AgentMessage:
    """Message between agents."""
    sender_id: str
    receiver_id: str
    protocol: CommunicationProtocol
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical


@dataclass
class ConsensusDecision:
    """Result of multi-agent consensus."""
    decision: str
    confidence: float
    participating_agents: List[str]
    voting_results: Dict[str, Any]
    reasoning: str


class AgentOrchestrator:
    """
    Orchestrates multiple intelligent agents and manages their interactions.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.communication_protocols: Dict[str, List[str]] = {}
        self.message_queue: List[AgentMessage] = []
        self.is_monitoring = False
        self.logger = logging.getLogger("orchestrator")
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name} ({agent.__class__.__name__})")
    
    def unregister_agent(self, name: str):
        """Unregister an agent."""
        if name in self.agents:
            del self.agents[name]
            self.logger.info(f"Unregistered agent: {name}")
    
    def define_communication_protocol(self, protocols: Dict[str, List[str]]):
        """
        Define communication protocols between agents.
        
        Args:
            protocols: Dict mapping protocol names to lists of participating agents
        """
        self.communication_protocols.update(protocols)
        self.logger.info(f"Defined communication protocols: {list(protocols.keys())}")
    
    async def start_autonomous_monitoring(self):
        """Start autonomous monitoring with all registered agents."""
        if not self.agents:
            raise ValueError("No agents registered for monitoring")
        
        self.is_monitoring = True
        self.logger.info("Starting autonomous monitoring")
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
        
        # Start message processing
        message_task = asyncio.create_task(self._process_messages())
        
        # Keep monitoring running
        try:
            await message_task
        except asyncio.CancelledError:
            self.logger.info("Monitoring stopped")
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """Stop monitoring and cleanup."""
        self.is_monitoring = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        self.logger.info("Autonomous monitoring stopped")
    
    async def coordinate_analysis(self, sensor_reading: SensorReading) -> Dict[str, AnalysisResult]:
        """
        Coordinate analysis across multiple agents for a sensor reading.
        
        Args:
            sensor_reading: Sensor reading to analyze
            
        Returns:
            Dict mapping agent names to their analysis results
        """
        results = {}
        
        # Run analysis in parallel across all active agents
        tasks = []
        agent_names = []
        
        for name, agent in self.agents.items():
            if agent.is_active:
                task = asyncio.create_task(agent.analyze(sensor_reading))
                tasks.append(task)
                agent_names.append(name)
        
        if tasks:
            analyses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for name, analysis in zip(agent_names, analyses):
                if isinstance(analysis, Exception):
                    self.logger.error(f"Agent {name} analysis failed: {analysis}")
                    results[name] = None
                else:
                    results[name] = analysis
                    
                    # Trigger communication protocols if anomaly detected
                    if analysis and hasattr(analysis, 'anomaly_detected') and analysis.anomaly_detected:
                        await self._trigger_alert_escalation(name, analysis)
        
        return results
    
    async def build_consensus(self, agents: List[str], decision_prompt: str, 
                            voting_mechanism: str = "weighted_confidence") -> ConsensusDecision:
        """
        Build consensus among specified agents for a decision.
        
        Args:
            agents: List of agent names to participate in consensus
            decision_prompt: The decision that needs to be made
            voting_mechanism: How to weight votes ('simple_majority', 'weighted_confidence')
            
        Returns:
            ConsensusDecision with the consensus result
        """
        participating_agents = [name for name in agents if name in self.agents]
        
        if not participating_agents:
            raise ValueError("No valid agents specified for consensus")
        
        # Collect votes/opinions from each agent
        votes = {}
        
        for agent_name in participating_agents:
            agent = self.agents[agent_name]
            
            # Get agent's analysis/opinion (simplified implementation)
            # In practice, this would be more sophisticated
            recent_analyses = agent.get_analysis_history(limit=5)
            
            if recent_analyses:
                avg_confidence = sum(a.confidence for a in recent_analyses) / len(recent_analyses)
                anomaly_rate = sum(1 for a in recent_analyses if a.anomaly_detected) / len(recent_analyses)
                
                votes[agent_name] = {
                    "decision": "reject" if anomaly_rate > 0.5 else "approve",
                    "confidence": avg_confidence,
                    "reasoning": f"Based on {len(recent_analyses)} recent analyses"
                }
            else:
                votes[agent_name] = {
                    "decision": "abstain",
                    "confidence": 0.5,
                    "reasoning": "Insufficient data for decision"
                }
        
        # Calculate consensus based on voting mechanism
        if voting_mechanism == "weighted_confidence":
            consensus_result = self._weighted_confidence_consensus(votes)
        else:
            consensus_result = self._simple_majority_consensus(votes)
        
        # Generate reasoning
        reasoning = self._generate_consensus_reasoning(votes, consensus_result)
        
        return ConsensusDecision(
            decision=consensus_result["decision"],
            confidence=consensus_result["confidence"],
            participating_agents=participating_agents,
            voting_results=votes,
            reasoning=reasoning
        )
    
    async def configure_for_product(self, product_spec: Dict[str, Any], 
                                  quality_targets: Dict[str, float],
                                  regulatory_requirements: List[str]):
        """
        Configure agents for a specific product.
        
        Args:
            product_spec: Product specification
            quality_targets: Quality targets for the product
            regulatory_requirements: Regulatory requirements to meet
        """
        self.logger.info(f"Configuring for product: {product_spec.get('name', 'Unknown')}")
        
        # Update agent configurations based on product requirements
        for agent_name, agent in self.agents.items():
            # This would update agent-specific configurations
            # For now, just log the configuration
            self.logger.info(f"Configured agent {agent_name} for product requirements")
    
    def on_quality_event(self, callback: Callable):
        """Register callback for quality events."""
        if "quality_event" not in self.event_callbacks:
            self.event_callbacks["quality_event"] = []
        self.event_callbacks["quality_event"].append(callback)
    
    async def _process_messages(self):
        """Process inter-agent messages."""
        while self.is_monitoring:
            if self.message_queue:
                # Sort messages by priority
                self.message_queue.sort(key=lambda m: m.priority, reverse=True)
                
                # Process highest priority message
                message = self.message_queue.pop(0)
                await self._handle_message(message)
            
            await asyncio.sleep(0.1)  # Check for messages every 100ms
    
    async def _handle_message(self, message: AgentMessage):
        """Handle a specific inter-agent message."""
        self.logger.debug(f"Processing message: {message.protocol.value} from {message.sender_id} to {message.receiver_id}")
        
        if message.protocol == CommunicationProtocol.ALERT_ESCALATION:
            await self._handle_alert_escalation(message)
        elif message.protocol == CommunicationProtocol.KNOWLEDGE_SHARING:
            await self._handle_knowledge_sharing(message)
        elif message.protocol == CommunicationProtocol.CONSENSUS_BUILDING:
            await self._handle_consensus_building(message)
        # Add more protocol handlers as needed
    
    async def _trigger_alert_escalation(self, agent_name: str, analysis: AnalysisResult):
        """Trigger alert escalation protocol."""
        if "alert_escalation" in self.communication_protocols:
            target_agents = self.communication_protocols["alert_escalation"]
            
            for target_agent in target_agents:
                if target_agent != agent_name and target_agent in self.agents:
                    message = AgentMessage(
                        sender_id=agent_name,
                        receiver_id=target_agent,
                        protocol=CommunicationProtocol.ALERT_ESCALATION,
                        content={
                            "analysis": analysis.__dict__,
                            "alert_type": "quality_anomaly"
                        },
                        timestamp=datetime.now(),
                        priority=3  # High priority
                    )
                    self.message_queue.append(message)
    
    async def _handle_alert_escalation(self, message: AgentMessage):
        """Handle alert escalation message."""
        # Notify relevant systems about the alert
        self.logger.warning(f"Alert escalation from {message.sender_id}: {message.content.get('alert_type')}")
        
        # Trigger quality event callbacks
        if "quality_event" in self.event_callbacks:
            for callback in self.event_callbacks["quality_event"]:
                try:
                    await callback(message.content)
                except Exception as e:
                    self.logger.error(f"Error in quality event callback: {e}")
    
    async def _handle_knowledge_sharing(self, message: AgentMessage):
        """Handle knowledge sharing between agents."""
        # Implementation for knowledge sharing protocol
        pass
    
    async def _handle_consensus_building(self, message: AgentMessage):
        """Handle consensus building message."""
        # Implementation for consensus building protocol
        pass
    
    def _weighted_confidence_consensus(self, votes: Dict[str, Dict]) -> Dict[str, Any]:
        """Build consensus using weighted confidence voting."""
        total_weight = 0
        weighted_decisions = {"approve": 0, "reject": 0, "abstain": 0}
        
        for agent_name, vote in votes.items():
            confidence = vote["confidence"]
            decision = vote["decision"]
            
            weighted_decisions[decision] += confidence
            total_weight += confidence
        
        # Find decision with highest weighted score
        if total_weight > 0:
            best_decision = max(weighted_decisions, key=weighted_decisions.get)
            consensus_confidence = weighted_decisions[best_decision] / total_weight
        else:
            best_decision = "abstain"
            consensus_confidence = 0.0
        
        return {
            "decision": best_decision,
            "confidence": consensus_confidence
        }
    
    def _simple_majority_consensus(self, votes: Dict[str, Dict]) -> Dict[str, Any]:
        """Build consensus using simple majority voting."""
        decision_counts = {"approve": 0, "reject": 0, "abstain": 0}
        
        for vote in votes.values():
            decision_counts[vote["decision"]] += 1
        
        best_decision = max(decision_counts, key=decision_counts.get)
        consensus_confidence = decision_counts[best_decision] / len(votes)
        
        return {
            "decision": best_decision,
            "confidence": consensus_confidence
        }
    
    def _generate_consensus_reasoning(self, votes: Dict[str, Dict], 
                                    consensus_result: Dict[str, Any]) -> str:
        """Generate reasoning for consensus decision."""
        decision = consensus_result["decision"]
        confidence = consensus_result["confidence"]
        
        reasoning = f"Consensus reached: {decision} (confidence: {confidence:.2f}). "
        
        # Add summary of individual votes
        approve_count = sum(1 for v in votes.values() if v["decision"] == "approve")
        reject_count = sum(1 for v in votes.values() if v["decision"] == "reject")
        abstain_count = sum(1 for v in votes.values() if v["decision"] == "abstain")
        
        reasoning += f"Votes: {approve_count} approve, {reject_count} reject, {abstain_count} abstain."
        
        return reasoning