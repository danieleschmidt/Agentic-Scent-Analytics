#!/usr/bin/env python3
"""
Multi-agent coordination demonstration.
Shows how multiple agents work together for comprehensive monitoring.
"""

import asyncio
import logging
from agentic_scent import ScentAnalyticsFactory, QualityControlAgent, AgentOrchestrator
from agentic_scent.agents.base import AgentConfig, AgentCapability, MockLLMAgent

logging.basicConfig(level=logging.INFO)

def create_agent(agent_type: str, focus: str, sensors: list, knowledge: str) -> MockLLMAgent:
    """Create a specialized agent."""
    config = AgentConfig(
        agent_id=f"{agent_type}_{focus}",
        capabilities=[AgentCapability.ANOMALY_DETECTION, AgentCapability.ROOT_CAUSE_ANALYSIS],
        confidence_threshold=0.7
    )
    return MockLLMAgent(config)

async def main():
    """Multi-agent demonstration."""
    print("🤖 Agentic Scent Analytics - Multi-Agent Demo")
    print("=" * 50)
    
    # Initialize factory system
    factory = ScentAnalyticsFactory(
        production_line='pharma_tablet_production',
        e_nose_config={
            'sensors': ['MOS', 'PID', 'EC', 'QCM'],
            'sampling_rate': 5,
            'channels': 32
        },
        site_id='pharma_plant_01'
    )
    
    # Initialize multi-agent orchestrator
    orchestrator = AgentOrchestrator()
    
    # Create specialized agents
    agents = {
        'inlet_monitor': create_agent(
            type='quality_control',
            focus='raw_material_inspection',
            sensors=['e_nose_array_1', 'moisture_sensor'],
            knowledge='raw_material_specs.yaml'
        ),
        
        'process_monitor': create_agent(
            type='process_control',
            focus='reaction_monitoring',
            sensors=['e_nose_array_2', 'temperature_probes'],
            knowledge='reaction_kinetics.db'
        ),
        
        'packaging_inspector': create_agent(
            type='quality_control',
            focus='final_product_verification',
            sensors=['e_nose_array_3', 'vision_system'],
            knowledge='product_standards.json'
        ),
        
        'maintenance_predictor': create_agent(
            type='predictive_maintenance',
            focus='equipment_health',
            sensors='all',
            knowledge='equipment_history.db'
        )
    }
    
    # Register agents with orchestrator
    for name, agent in agents.items():
        orchestrator.register_agent(name, agent)
        factory.register_agent(agent)
    
    # Define inter-agent communication protocols
    orchestrator.define_communication_protocol({
        'alert_escalation': ['inlet_monitor', 'process_monitor', 'packaging_inspector'],
        'maintenance_coordination': list(agents.keys()),
        'knowledge_sharing': list(agents.keys())
    })
    
    print(f"🏭 Factory: {factory.config.production_line}")
    print(f"🤖 Agents deployed: {len(agents)}")
    print(f"📡 Communication protocols: {len(orchestrator.communication_protocols)}")
    print()
    
    # Start all agents
    print("🚀 Starting multi-agent system...")
    for agent in agents.values():
        await agent.start()
    
    # Simulate coordinated monitoring
    print("📊 Coordinated analysis (5 readings):")
    print("-" * 50)
    
    reading_count = 0
    async for reading in factory.sensor_stream():
        reading_count += 1
        
        print(f"\n📅 Reading {reading_count} - {reading.timestamp.strftime('%H:%M:%S')}")
        
        # Coordinate analysis across all agents
        analyses = await orchestrator.coordinate_analysis(reading)
        
        # Display results from each agent
        for agent_name, analysis in analyses.items():
            if analysis:
                status = "⚠️  ALERT" if analysis.anomaly_detected else "✅ OK"
                print(f"  {agent_name:20} | {status} | Confidence: {analysis.confidence:.3f}")
            else:
                print(f"  {agent_name:20} | ❌ ERROR | Analysis failed")
        
        # Check for consensus if any anomalies detected
        anomaly_agents = [name for name, analysis in analyses.items() 
                         if analysis and analysis.anomaly_detected]
        
        if anomaly_agents:
            print(f"\n  🔍 Anomaly detected by: {', '.join(anomaly_agents)}")
            
            # Build consensus for quality decision
            try:
                consensus = await orchestrator.build_consensus(
                    agents=list(agents.keys()),
                    decision_prompt="Should this batch be flagged for quality review?",
                    voting_mechanism="weighted_confidence"
                )
                
                print(f"  🏛️  Consensus: {consensus.decision.upper()} (confidence: {consensus.confidence:.3f})")
                print(f"  💭 Reasoning: {consensus.reasoning[:100]}...")
                
            except Exception as e:
                print(f"  ❌ Consensus failed: {e}")
        
        # Stop after 5 readings for demo
        if reading_count >= 5:
            break
    
    # Stop monitoring
    await factory.stop_monitoring()
    await orchestrator.stop_monitoring()
    
    print("\n🏁 Multi-agent monitoring completed")
    
    # Generate summary report
    print("\n📈 Multi-Agent Summary:")
    print("-" * 30)
    
    total_analyses = 0
    total_anomalies = 0
    
    for agent_name, agent in agents.items():
        history = agent.get_analysis_history()
        anomaly_count = sum(1 for h in history if h.anomaly_detected)
        
        print(f"  {agent_name:20} | Analyses: {len(history)} | Anomalies: {anomaly_count}")
        
        total_analyses += len(history)
        total_anomalies += anomaly_count
    
    print(f"\n  📊 System Performance:")
    print(f"    Total analyses: {total_analyses}")
    print(f"    Total anomalies: {total_anomalies}")
    if total_analyses > 0:
        anomaly_rate = (total_anomalies / total_analyses) * 100
        print(f"    Anomaly rate: {anomaly_rate:.1f}%")
    
    print(f"    System status: {'⚠️  Requires attention' if total_anomalies > 0 else '✅ Normal operation'}")

if __name__ == "__main__":
    asyncio.run(main())