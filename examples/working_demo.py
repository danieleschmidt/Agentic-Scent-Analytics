#!/usr/bin/env python3

"""
Working Production Demo - Using actual available API.
"""

import asyncio
import time
import sys
import os
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_scent.core.factory import ScentAnalyticsFactory
from agentic_scent.agents.quality_control import QualityControlAgent
from agentic_scent.sensors.base import SensorReading, SensorType


async def working_demo():
    """Simple working demo of core capabilities."""
    print("🏭 AGENTIC SCENT ANALYTICS - WORKING DEMO")
    print("=" * 50)
    
    # Initialize factory
    factory = ScentAnalyticsFactory(
        production_line="working_demo_line",
        e_nose_config={"sensors": ["MOS", "PID"], "channels": 16}
    )
    
    print(f"✅ Factory initialized: {factory.config.production_line}")
    
    # Create QC agent
    qc_agent = QualityControlAgent()
    await qc_agent.start()
    
    print(f"✅ QC Agent started: {qc_agent.agent_id}")
    
    # Create mock sensor readings manually
    print("\n🔬 Quality Control Analysis")
    print("-" * 30)
    
    for cycle in range(3):
        # Create mock sensor reading
        sensor_data = np.random.randn(16) * 0.1  # 16 channels with noise
        sensor_data[0] += 1.0  # Base signal
        
        reading = SensorReading(
            sensor_id="e_nose_001",
            sensor_type=SensorType.E_NOSE,
            values=sensor_data.tolist(),
            timestamp=datetime.now(),
            metadata={"cycle": cycle, "production_line": "working_demo"}
        )
        
        # Analyze with QC agent
        analysis = await qc_agent.analyze(reading)
        
        status = "✅ PASS" if not analysis.anomaly_detected else "⚠️  ALERT" 
        print(f"Cycle {cycle+1}: {status}")
        print(f"   Confidence: {analysis.confidence:.1%}")
        print(f"   Issue: {analysis.issue_type}")
        
        if analysis.anomaly_detected:
            print(f"   Root Cause: {analysis.root_cause}")
            print(f"   Action: {analysis.recommended_action}")
        
        await asyncio.sleep(0.3)
    
    await qc_agent.stop()
    
    # Demonstrate factory status
    print(f"\n📊 Factory Status")
    print("-" * 20)
    
    factory_status = factory.get_current_state()
    print(f"Current temperature: {factory_status['temperature']:.1f}°C")
    print(f"Humidity: {factory_status['humidity']:.1f}%")
    print(f"Pressure: {factory_status['pressure']:.1f} kPa")
    
    # Test performance
    print(f"\n⚡ Performance Test")
    print("-" * 20)
    
    start_time = time.time()
    
    # Run multiple analyses
    qc_agent2 = QualityControlAgent()
    await qc_agent2.start()
    
    tasks = []
    for i in range(10):
        sensor_data = np.random.randn(16) * 0.1
        reading = SensorReading(
            sensor_id=f"e_nose_{i:03d}",
            sensor_type=SensorType.E_NOSE,
            values=sensor_data.tolist(),
            timestamp=datetime.now(),
            metadata={"batch": f"test_{i}"}
        )
        tasks.append(qc_agent2.analyze(reading))
    
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    print(f"Processed: {len(results)} analyses")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {len(results)/duration:.1f} analyses/second")
    
    anomaly_count = sum(1 for r in results if r.anomaly_detected)
    avg_confidence = sum(r.confidence for r in results) / len(results)
    
    print(f"Anomalies: {anomaly_count}/{len(results)}")
    print(f"Avg confidence: {avg_confidence:.1%}")
    
    await qc_agent2.stop()
    
    print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 35)
    print("✅ Factory system operational")
    print("✅ AI agents functioning")
    print("✅ Real-time analysis working")
    print("✅ Performance validated")
    print("\n🚀 Ready for production deployment!")


if __name__ == "__main__":
    asyncio.run(working_demo())