#!/usr/bin/env python3
"""
Scent fingerprinting demonstration.
Shows how to create and use scent fingerprints for quality control.
"""

import asyncio
import logging
import numpy as np
from agentic_scent import ScentAnalyticsFactory, ScentFingerprinter
from agentic_scent.sensors.base import SensorReading, SensorType
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def generate_mock_training_data(product_name: str, n_samples: int = 50) -> list:
    """Generate mock training data for fingerprinting."""
    training_data = []
    
    # Base signature for the product
    if product_name == "aspirin_500mg":
        base_signature = [500, 120, 80, 200, 150, 90, 300, 180] + [100] * 24
    elif product_name == "vitamin_c_tablets":
        base_signature = [300, 180, 120, 150, 200, 110, 250, 160] + [80] * 24
    else:
        base_signature = [400, 150, 100, 175, 125, 95, 275, 170] + [90] * 24
    
    for i in range(n_samples):
        # Add controlled variation to simulate natural batch differences
        variation = np.random.normal(1.0, 0.05, len(base_signature))  # 5% variation
        values = [base * var for base, var in zip(base_signature, variation)]
        
        # Ensure positive values
        values = [max(10, val) for val in values]
        
        reading = SensorReading(
            sensor_id=f"training_sensor_{i}",
            sensor_type=SensorType.E_NOSE,
            values=values,
            timestamp=datetime.now(),
            metadata={"batch_id": f"good_batch_{i}", "product": product_name}
        )
        
        training_data.append(reading)
    
    return training_data

def generate_test_sample(product_name: str, contamination_level: float = 0.0) -> SensorReading:
    """Generate a test sample with optional contamination."""
    # Base signature
    if product_name == "aspirin_500mg":
        base_signature = [500, 120, 80, 200, 150, 90, 300, 180] + [100] * 24
    else:
        base_signature = [400, 150, 100, 175, 125, 95, 275, 170] + [90] * 24
    
    # Add normal variation
    variation = np.random.normal(1.0, 0.03, len(base_signature))
    values = [base * var for base, var in zip(base_signature, variation)]
    
    # Add contamination if specified
    if contamination_level > 0:
        # Simulate contamination by increasing certain channels
        contamination_channels = [0, 2, 4, 6, 8, 10]  # Specific channels affected
        for channel in contamination_channels:
            if channel < len(values):
                values[channel] *= (1.0 + contamination_level)
    
    return SensorReading(
        sensor_id="test_sensor",
        sensor_type=SensorType.E_NOSE,
        values=values,
        timestamp=datetime.now(),
        metadata={"test_sample": True, "contamination_level": contamination_level}
    )

async def main():
    """Fingerprinting demonstration."""
    print("🔍 Agentic Scent Analytics - Fingerprinting Demo")
    print("=" * 55)
    
    # Initialize fingerprinting system
    fingerprinter = ScentFingerprinter(
        method='deep_embedding',
        embedding_dim=64
    )
    
    # Product to analyze
    product_name = "aspirin_500mg"
    print(f"📦 Product: {product_name}")
    print()
    
    # Generate training data from "good" batches
    print("🔬 Generating training data from good batches...")
    training_data = generate_mock_training_data(product_name, n_samples=100)
    print(f"  Generated {len(training_data)} training samples")
    
    # Create fingerprint model
    print("🧠 Creating scent fingerprint...")
    fingerprint_model = fingerprinter.create_fingerprint(
        training_data=training_data,
        product_id=product_name,
        augmentation=True,
        contamination_simulation=False
    )
    
    print(f"  ✅ Fingerprint created successfully")
    print(f"  📏 Embedding dimension: {fingerprint_model.embedding_dim}")
    print(f"  🎯 Similarity threshold: {fingerprint_model.similarity_threshold:.3f}")
    print(f"  📊 Training samples: {fingerprint_model.training_samples}")
    print()
    
    # Test fingerprinting with various samples
    test_scenarios = [
        {"name": "Good batch sample", "contamination": 0.0},
        {"name": "Minor deviation", "contamination": 0.1},
        {"name": "Moderate contamination", "contamination": 0.3},
        {"name": "Severe contamination", "contamination": 0.8},
    ]
    
    print("🧪 Testing fingerprint against various samples:")
    print("-" * 55)
    
    for scenario in test_scenarios:
        print(f"\n🔬 {scenario['name']}:")
        
        # Generate test sample
        test_sample = generate_test_sample(product_name, scenario['contamination'])
        
        # Compare to fingerprint
        similarity_result = fingerprinter.compare_to_fingerprint(test_sample, fingerprint_model)
        
        # Display results
        status = "✅ PASS" if similarity_result.is_match else "❌ FAIL"
        print(f"  Status: {status}")
        print(f"  Similarity: {similarity_result.similarity_score:.3f}")
        print(f"  Confidence: {similarity_result.confidence:.3f}")
        print(f"  Deviation channels: {similarity_result.deviation_channels}")
        
        # Detailed deviation analysis for failed samples
        if not similarity_result.is_match:
            deviation_analysis = fingerprinter.analyze_deviations(test_sample, fingerprint_model)
            
            print(f"  🔍 Detailed Analysis:")
            print(f"    Severity score: {deviation_analysis['severity_score']:.3f}")
            print(f"    Deviation type: {deviation_analysis['deviation_type']}")
            print(f"    Most affected channel: {deviation_analysis['max_deviation_channel']}")
            print(f"    Max deviation: {deviation_analysis['max_deviation_value']:.3f}")
    
    print("\n" + "="*55)
    print("📈 Fingerprinting Performance Summary:")
    print("-" * 35)
    
    # Test with multiple samples for statistics
    good_samples = [generate_test_sample(product_name, 0.0) for _ in range(20)]
    bad_samples = [generate_test_sample(product_name, 0.5) for _ in range(20)]
    
    # Evaluate good samples
    good_results = [fingerprinter.compare_to_fingerprint(sample, fingerprint_model) 
                    for sample in good_samples]
    good_pass_rate = sum(1 for r in good_results if r.is_match) / len(good_results)
    
    # Evaluate bad samples
    bad_results = [fingerprinter.compare_to_fingerprint(sample, fingerprint_model) 
                   for sample in bad_samples]
    bad_reject_rate = sum(1 for r in bad_results if not r.is_match) / len(bad_results)
    
    print(f"✅ Good samples correctly identified: {good_pass_rate:.1%}")
    print(f"❌ Bad samples correctly rejected: {bad_reject_rate:.1%}")
    print(f"🎯 Overall accuracy: {(good_pass_rate + bad_reject_rate) / 2:.1%}")
    
    # Calculate average similarities
    avg_good_similarity = np.mean([r.similarity_score for r in good_results])
    avg_bad_similarity = np.mean([r.similarity_score for r in bad_results])
    
    print(f"📊 Average similarity - Good: {avg_good_similarity:.3f}, Bad: {avg_bad_similarity:.3f}")
    print(f"📏 Separation margin: {avg_good_similarity - avg_bad_similarity:.3f}")
    
    print("\n🏁 Fingerprinting demonstration completed!")

if __name__ == "__main__":
    asyncio.run(main())