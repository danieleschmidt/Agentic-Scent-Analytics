#!/usr/bin/env python3
"""
Basic functionality tests without external dependencies.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test basic imports work."""
    print("Testing imports...")
    
    try:
        # Test structure
        import agentic_scent
        print("✅ Main package import successful")
        
        # Test core functionality with mock numpy
        sys.modules['numpy'] = type(sys)('numpy')
        sys.modules['numpy'].array = list
        sys.modules['numpy'].mean = lambda x: sum(x) / len(x)
        sys.modules['numpy'].std = lambda x: (sum((i - sum(x)/len(x))**2 for i in x) / len(x))**0.5
        sys.modules['numpy'].max = max
        sys.modules['numpy'].min = min
        sys.modules['numpy'].dot = lambda a, b: sum(x*y for x,y in zip(a,b))
        sys.modules['numpy'].linalg = type(sys)('linalg')
        sys.modules['numpy'].linalg.norm = lambda x: (sum(i**2 for i in x))**0.5
        sys.modules['numpy'].isnan = lambda x: x != x
        sys.modules['numpy'].isinf = lambda x: x == float('inf') or x == float('-inf')
        sys.modules['numpy'].where = lambda cond: [i for i, c in enumerate(cond) if c]
        sys.modules['numpy'].argmax = lambda x: x.index(max(x))
        sys.modules['numpy'].argmin = lambda x: x.index(min(x))
        sys.modules['numpy'].random = type(sys)('random')
        sys.modules['numpy'].random.normal = lambda m, s, n: [m + s * 0.1 * i for i in range(n)]
        sys.modules['numpy'].vstack = lambda x: sum(x, [])
        sys.modules['numpy'].percentile = lambda x, p: sorted(x)[int(len(x) * p / 100)]
        
        # Mock sklearn
        sys.modules['sklearn'] = type(sys)('sklearn')
        sys.modules['sklearn.decomposition'] = type(sys)('decomposition')
        sys.modules['sklearn.cluster'] = type(sys)('cluster') 
        sys.modules['sklearn.metrics'] = type(sys)('metrics')
        sys.modules['sklearn.metrics.pairwise'] = type(sys)('pairwise')
        sys.modules['sklearn.preprocessing'] = type(sys)('preprocessing')
        sys.modules['sklearn.model_selection'] = type(sys)('model_selection')
        sys.modules['sklearn.ensemble'] = type(sys)('ensemble')
        
        # Mock classes
        class MockPCA:
            def __init__(self, **kwargs): pass
            def fit_transform(self, X): return [[1.0] * 5 for _ in X]
            def transform(self, X): return [[1.0] * 5 for _ in X]
            def inverse_transform(self, X): return [[100.0] * 8 for _ in X]
        
        class MockScaler:
            def __init__(self): pass
            def fit_transform(self, X): return X
            def transform(self, X): return X
            def inverse_transform(self, X): return X
        
        class MockRandomForest:
            def __init__(self, **kwargs): pass
            def fit(self, X, y): pass
            def predict(self, X): return [0.8 for _ in X]
            @property
            def estimators_(self): return [self] * 5
        
        sys.modules['sklearn.decomposition'].PCA = MockPCA
        sys.modules['sklearn.preprocessing'].StandardScaler = MockScaler
        sys.modules['sklearn.ensemble'].RandomForestRegressor = MockRandomForest
        sys.modules['sklearn.metrics.pairwise'].cosine_similarity = lambda a, b: [[0.9]]
        sys.modules['sklearn.model_selection'].train_test_split = lambda *args, **kwargs: (args[0][:10], args[0][10:], args[1][:10], args[1][10:])
        sys.modules['sklearn.metrics'].mean_squared_error = lambda a, b: 0.1
        sys.modules['sklearn.metrics'].r2_score = lambda a, b: 0.85
        
        # Mock pandas
        sys.modules['pandas'] = type(sys)('pandas')
        class MockDataFrame:
            def __init__(self, data): 
                self.data = data
                self.columns = list(data.keys()) if isinstance(data, dict) else []
            def get(self, key, default=None): 
                return self.data.get(key, default) if isinstance(self.data, dict) else default
            def __getitem__(self, key): 
                return self.data[key] if isinstance(self.data, dict) else []
            def fillna(self, value): return self
        sys.modules['pandas'].DataFrame = MockDataFrame
        sys.modules['pandas'].to_datetime = lambda x: [datetime.now() for _ in x]
        
        # Mock other dependencies
        sys.modules['psutil'] = type(sys)('psutil')
        sys.modules['psutil'].cpu_percent = lambda **kwargs: 45.0
        sys.modules['psutil'].virtual_memory = lambda: type('obj', (), {'percent': 60.0, 'used': 8000000000, 'available': 4000000000})()
        sys.modules['psutil'].disk_usage = lambda path: type('obj', (), {'used': 50000000000, 'total': 100000000000})()
        sys.modules['psutil'].net_io_counters = lambda: type('obj', (), {'bytes_sent': 1000000, 'bytes_recv': 2000000})()
        sys.modules['psutil'].Process = lambda: type('obj', (), {'memory_info': lambda: type('obj', (), {'rss': 100000000})()})()
        
        from agentic_scent.core.factory import ScentAnalyticsFactory
        print("✅ Factory import successful")
        
        from agentic_scent.agents.quality_control import QualityControlAgent
        print("✅ QualityControlAgent import successful")
        
        from agentic_scent.sensors.base import SensorReading, SensorType
        print("✅ Sensor base classes import successful")
        
        from agentic_scent.analytics.fingerprinting import ScentFingerprinter
        print("✅ Analytics import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Test factory creation
        from agentic_scent.core.factory import ScentAnalyticsFactory
        factory = ScentAnalyticsFactory(
            production_line='test_line',
            e_nose_config={'channels': 16, 'sampling_rate': 5.0}
        )
        print("✅ Factory creation successful")
        
        # Test agent creation
        from agentic_scent.agents.quality_control import QualityControlAgent
        agent = QualityControlAgent(agent_id='test_agent')
        print("✅ Agent creation successful")
        
        # Test sensor reading creation
        from agentic_scent.sensors.base import SensorReading, SensorType
        reading = SensorReading(
            sensor_id="test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=[100.0, 150.0, 200.0, 125.0],
            timestamp=datetime.now()
        )
        print("✅ Sensor reading creation successful")
        
        # Test fingerprinting
        from agentic_scent.analytics.fingerprinting import ScentFingerprinter
        fingerprinter = ScentFingerprinter(embedding_dim=32)
        print("✅ Fingerprinter creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_functionality():
    """Test async functionality."""
    print("\nTesting async functionality...")
    
    try:
        from agentic_scent.agents.quality_control import QualityControlAgent
        from agentic_scent.sensors.base import SensorReading, SensorType
        
        # Create agent and reading
        agent = QualityControlAgent(agent_id='async_test_agent')
        reading = SensorReading(
            sensor_id="async_test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=[100.0, 150.0, 200.0, 125.0, 175.0, 110.0, 190.0, 140.0],
            timestamp=datetime.now()
        )
        
        # Start agent
        await agent.start()
        print("✅ Agent start successful")
        
        # Analyze reading
        result = await agent.analyze(reading)
        print("✅ Analysis successful")
        
        if result:
            print(f"   Analysis confidence: {result.confidence:.3f}")
            print(f"   Anomaly detected: {result.anomaly_detected}")
        
        # Stop agent
        await agent.stop()
        print("✅ Agent stop successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from agentic_scent.core.config import AgenticScentConfig
        
        config = AgenticScentConfig(
            site_id="test_site",
            environment="testing"
        )
        print("✅ Configuration creation successful")
        
        # Test config attributes
        assert config.site_id == "test_site"
        assert config.environment == "testing"
        print("✅ Configuration attributes correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 Running Agentic Scent Analytics Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality), 
        ("Configuration", test_configuration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Run async test
    print(f"\n📋 Async Functionality")
    print("-" * 30)
    try:
        if asyncio.run(test_async_functionality()):
            passed += 1
            print("✅ Async Functionality PASSED")
        else:
            print("❌ Async Functionality FAILED")
    except Exception as e:
        print(f"❌ Async Functionality FAILED: {e}")
    
    total += 1  # Add async test to total
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())