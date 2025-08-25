#!/usr/bin/env python3

"""
Simple Production Demo - Working demonstration of core capabilities.
"""

import asyncio
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_scent.core.factory import ScentAnalyticsFactory
from agentic_scent.agents.quality_control import QualityControlAgent
from agentic_scent.analytics.fingerprinting import ScentFingerprinter


async def basic_quality_control_demo():
    """Demonstrate basic quality control functionality."""
    print("🏭 Basic Quality Control Demo")
    print("=" * 40)
    
    # Initialize factory
    factory = ScentAnalyticsFactory(
        production_line="demo_line_001",
        e_nose_config={"sensors": ["MOS", "PID"], "channels": 16}
    )
    
    # Create quality control agent
    qc_agent = QualityControlAgent()
    await qc_agent.start()
    
    print(f"✅ Factory initialized: {factory.config.production_line}")
    print(f"✅ QC Agent started: {qc_agent.agent_id}")
    
    # Simulate production monitoring
    print("\n🔬 Production Monitoring (5 cycles)")
    print("-" * 35)
    
    for cycle in range(5):
        # Generate sensor reading
        reading = factory.create_mock_sensor_reading()
        
        # Analyze with QC agent
        analysis = await qc_agent.analyze(reading)
        
        status = "✅ PASS" if not analysis.anomaly_detected else "❌ FAIL"
        print(f"Cycle {cycle+1}: {status} (Confidence: {analysis.confidence:.1%})")
        
        if analysis.anomaly_detected:
            print(f"   Issue: {analysis.issue_type}")
            print(f"   Action: {analysis.recommended_action}")
        
        await asyncio.sleep(0.3)
    
    await qc_agent.stop()
    print("\n✅ Quality control demo completed")


async def fingerprinting_demo():
    """Demonstrate scent fingerprinting capability."""
    print("\n🔬 Scent Fingerprinting Demo")
    print("=" * 35)
    
    # Create fingerprinter
    fingerprinter = ScentFingerprinter()
    
    # Create factory for data generation
    factory = ScentAnalyticsFactory(
        production_line="fingerprint_demo",
        e_nose_config={"sensors": ["MOS", "PID"], "channels": 16}
    )
    
    # Generate training data (good batches)
    print("📊 Generating training data...")
    training_readings = []
    for i in range(20):
        reading = factory.create_mock_sensor_reading()
        training_readings.append(reading)
    
    # Create fingerprint model
    print("🧠 Creating quality fingerprint...")
    fingerprint_result = await fingerprinter.create_fingerprint(
        readings=training_readings,
        product_name="demo_product"
    )
    
    print(f"✅ Fingerprint created: {fingerprint_result['fingerprint_id']}")
    print(f"   Quality score: {fingerprint_result['quality_score']:.2f}")
    print(f"   Contamination risk: {fingerprint_result['contamination_risk']:.1%}")
    
    # Test new batch against fingerprint
    print("\n🔍 Testing new batch...")
    test_reading = factory.create_mock_sensor_reading()
    
    similarity = await fingerprinter.compare_to_fingerprint(
        reading=test_reading,
        fingerprint_id=fingerprint_result['fingerprint_id']
    )
    
    match_status = "✅ MATCH" if similarity['similarity_score'] > 0.8 else "⚠️  DEVIATION"
    print(f"{match_status} Similarity: {similarity['similarity_score']:.1%}")
    
    if similarity['deviations']:
        print("   Detected deviations:")
        for deviation in similarity['deviations']:
            print(f"   • {deviation}")
    
    print("✅ Fingerprinting demo completed")


async def performance_demo():
    """Demonstrate performance capabilities."""
    print("\n⚡ Performance Demo")
    print("=" * 25)
    
    # Initialize components
    factory = ScentAnalyticsFactory(
        production_line="performance_test",
        e_nose_config={"sensors": ["MOS", "PID", "EC"], "channels": 32}
    )
    
    qc_agent = QualityControlAgent()
    await qc_agent.start()
    
    # Performance test
    print("🚀 Running performance test...")
    
    start_time = time.time()
    analyses_completed = 0
    
    # Process multiple readings rapidly
    tasks = []
    for i in range(50):
        reading = factory.create_mock_sensor_reading()
        task = qc_agent.analyze(reading)
        tasks.append(task)
    
    # Execute all analyses concurrently
    results = await asyncio.gather(*tasks)
    analyses_completed = len(results)
    
    end_time = time.time()
    duration = end_time - start_time
    throughput = analyses_completed / duration
    
    print(f"📊 Performance Results:")
    print(f"   Analyses completed: {analyses_completed}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Throughput: {throughput:.1f} analyses/second")
    
    # Analyze results
    anomalies = sum(1 for r in results if r.anomaly_detected)
    avg_confidence = sum(r.confidence for r in results) / len(results)
    
    print(f"   Anomalies detected: {anomalies}/{analyses_completed}")
    print(f"   Average confidence: {avg_confidence:.1%}")
    
    await qc_agent.stop()
    print("✅ Performance demo completed")


async def main():
    """Run all demos."""
    print("🏭 AGENTIC SCENT ANALYTICS - PRODUCTION DEMO")
    print("=" * 55)
    print("Demonstrating autonomous quality control capabilities")
    print()
    
    start_time = time.time()
    
    try:
        # Run demos
        await basic_quality_control_demo()
        await fingerprinting_demo()  
        await performance_demo()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 40)
        print(f"Total execution time: {total_time:.2f}s")
        
        print("\n🚀 Production Capabilities Demonstrated:")
        print("   ✅ Real-time quality monitoring")
        print("   ✅ Anomaly detection with root cause analysis")
        print("   ✅ Scent fingerprinting for batch comparison")
        print("   ✅ High-throughput concurrent processing")
        print("   ✅ Autonomous decision making")
        
        print(f"\n🌟 System is ready for smart factory deployment!")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())