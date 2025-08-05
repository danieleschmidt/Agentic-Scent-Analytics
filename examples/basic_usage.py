#!/usr/bin/env python3
"""
Basic usage example for Agentic Scent Analytics.
This demonstrates the core functionality with mock sensors.
"""

import asyncio
import logging
from agentic_scent import ScentAnalyticsFactory, QualityControlAgent

logging.basicConfig(level=logging.INFO)

async def main():
    """Basic usage demonstration."""
    print("🔬 Agentic Scent Analytics - Basic Usage Example")
    print("=" * 50)
    
    # Initialize factory analytics system
    factory = ScentAnalyticsFactory(
        production_line='demo_tablet_coating',
        e_nose_config={
            'sensors': ['MOS', 'PID', 'EC', 'QCM'],
            'sampling_rate': 10,  # Hz
            'channels': 32
        },
        site_id='demo_plant'
    )
    
    # Create quality control agent
    qc_agent = QualityControlAgent(
        llm_model='gpt-4',
        knowledge_base='demo_quality_standards.db',
        alert_threshold=0.95
    )
    
    # Register agent with factory
    factory.register_agent(qc_agent)
    
    print(f"📍 Factory: {factory.config.site_id} - {factory.config.production_line}")
    print(f"🤖 Agents registered: {len(factory.agents)}")
    print(f"🔧 Sensors available: {list(factory.sensors.keys())}")
    print()
    
    # Start monitoring
    print("🚀 Starting monitoring...")
    await qc_agent.start()
    
    # Simulate real-time monitoring for a short period
    print("📊 Real-time monitoring (10 readings):")
    print("-" * 40)
    
    reading_count = 0
    async for reading in factory.sensor_stream():
        reading_count += 1
        
        # Agent analyzes scent pattern
        analysis = await qc_agent.analyze(reading)
        
        if analysis:
            print(f"Reading {reading_count}:")
            print(f"  Timestamp: {reading.timestamp.strftime('%H:%M:%S')}")
            print(f"  Confidence: {analysis.confidence:.3f}")
            print(f"  Anomaly: {'⚠️  YES' if analysis.anomaly_detected else '✅ NO'}")
            
            if analysis.anomaly_detected:
                print(f"  🔍 Suspected cause: {getattr(analysis, 'root_cause', 'Unknown')}")
                print(f"  💡 Action: {getattr(analysis, 'recommended_action', 'Monitor')}")
        
        print()
        
        # Stop after 10 readings for demo
        if reading_count >= 10:
            break
    
    # Stop monitoring
    await factory.stop_monitoring()
    await qc_agent.stop()
    
    print("🏁 Monitoring completed")
    print("\n📈 Analysis Summary:")
    print(f"  Total readings: {reading_count}")
    
    # Get analysis history
    history = qc_agent.get_analysis_history()
    if history:
        anomaly_count = sum(1 for h in history if h.anomaly_detected)
        avg_confidence = sum(h.confidence for h in history) / len(history)
        
        print(f"  Anomalies detected: {anomaly_count}")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  System status: {'⚠️  Attention needed' if anomaly_count > 0 else '✅ Normal operation'}")

if __name__ == "__main__":
    asyncio.run(main())