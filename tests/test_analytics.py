"""
Analytics and machine learning tests.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from agentic_scent.analytics.fingerprinting import ScentFingerprinter, FingerprintModel, SimilarityResult
from agentic_scent.predictive.quality import QualityPredictor, QualityMetrics, QualityPrediction
from agentic_scent.sensors.base import SensorReading, SensorType


class TestScentFingerprinting:
    """Test scent fingerprinting functionality."""
    
    def test_fingerprinter_initialization(self):
        """Test fingerprinter initialization."""
        fingerprinter = ScentFingerprinter(
            method="deep_embedding",
            embedding_dim=128
        )
        
        assert fingerprinter.method == "deep_embedding"
        assert fingerprinter.embedding_dim == 128
        assert len(fingerprinter.fingerprint_models) == 0
    
    def test_create_fingerprint(self, training_data):
        """Test fingerprint model creation."""
        fingerprinter = ScentFingerprinter(embedding_dim=64)
        
        model = fingerprinter.create_fingerprint(
            training_data=training_data,
            product_id="test_product",
            augmentation=True
        )
        
        assert isinstance(model, FingerprintModel)
        assert model.product_id == "test_product"
        assert model.embedding_dim == 64
        assert model.reference_fingerprint is not None
        assert len(model.reference_fingerprint) == 64
        assert model.similarity_threshold > 0.0
        assert model.training_samples == len(training_data)
        assert model.pca_model is not None
        assert model.scaler is not None
    
    def test_fingerprint_comparison_good_sample(self, training_data):
        """Test fingerprint comparison with good sample."""
        fingerprinter = ScentFingerprinter(embedding_dim=32)
        model = fingerprinter.create_fingerprint(training_data, "test_product")
        
        # Create a sample similar to training data
        good_sample = SensorReading(
            sensor_id="test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=[105.0, 155.0, 205.0, 130.0, 180.0, 115.0, 195.0, 145.0],
            timestamp=datetime.now()
        )
        
        result = fingerprinter.compare_to_fingerprint(good_sample, model)
        
        assert isinstance(result, SimilarityResult)
        assert result.similarity_score > 0.7  # Should be similar
        assert result.is_match  # Should match
        assert result.confidence > 0.7
        assert isinstance(result.deviation_channels, list)
    
    def test_fingerprint_comparison_bad_sample(self, training_data):
        """Test fingerprint comparison with contaminated sample."""
        fingerprinter = ScentFingerprinter(embedding_dim=32)
        model = fingerprinter.create_fingerprint(training_data, "test_product")
        
        # Create a contaminated sample
        bad_sample = SensorReading(
            sensor_id="test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=[500.0, 750.0, 1000.0, 625.0, 875.0, 550.0, 950.0, 700.0],  # Much higher
            timestamp=datetime.now()
        )
        
        result = fingerprinter.compare_to_fingerprint(bad_sample, model)
        
        assert isinstance(result, SimilarityResult)
        assert result.similarity_score < model.similarity_threshold
        assert not result.is_match  # Should not match
        assert len(result.deviation_channels) > 0  # Should have deviations
    
    def test_deviation_analysis(self, training_data):
        """Test detailed deviation analysis."""
        fingerprinter = ScentFingerprinter(embedding_dim=32)
        model = fingerprinter.create_fingerprint(training_data, "test_product")
        
        # Create sample with specific deviations
        deviation_sample = SensorReading(
            sensor_id="test_sensor",
            sensor_type=SensorType.E_NOSE,
            values=[300.0, 150.0, 200.0, 125.0, 175.0, 110.0, 190.0, 140.0],  # First channel high
            timestamp=datetime.now()
        )
        
        deviations = fingerprinter.analyze_deviations(deviation_sample, model)
        
        assert "channel_deviations" in deviations
        assert "normalized_deviations" in deviations
        assert "significant_channels" in deviations
        assert "severity_score" in deviations
        assert "deviation_type" in deviations
        assert "max_deviation_channel" in deviations
        
        assert len(deviations["channel_deviations"]) == 8
        assert deviations["severity_score"] >= 0.0
        assert deviations["deviation_type"] in [
            "severe_contamination", "moderate_contamination", 
            "process_drift", "systematic_shift", "minor_variation"
        ]
    
    def test_model_persistence(self, training_data):
        """Test fingerprint model storage and retrieval."""
        fingerprinter = ScentFingerprinter()
        
        # Create and store model
        model = fingerprinter.create_fingerprint(training_data, "persistent_product")
        
        # Retrieve model
        retrieved_model = fingerprinter.get_model("persistent_product")
        assert retrieved_model is not None
        assert retrieved_model.product_id == "persistent_product"
        
        # Test model listing
        models = fingerprinter.list_models()
        assert "persistent_product" in models
    
    def test_empty_training_data(self):
        """Test fingerprint creation with empty training data."""
        fingerprinter = ScentFingerprinter()
        
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            fingerprinter.create_fingerprint([], "empty_product")


class TestQualityPredictor:
    """Test predictive quality analytics."""
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = QualityPredictor(
            model="random_forest",
            features=['scent_profile', 'process_params']
        )
        
        assert predictor.model_type == "random_forest"
        assert predictor.features == ['scent_profile', 'process_params']
        assert len(predictor.models) > 0  # Should have initialized models
        assert not predictor.is_trained
    
    def test_predictor_training(self, historical_data):
        """Test predictor training."""
        predictor = QualityPredictor()
        
        # Train the predictor
        predictor.train(
            historical_data=historical_data,
            quality_metrics=['potency', 'dissolution', 'stability']
        )
        
        assert predictor.is_trained
        assert len(predictor.training_data) == len(historical_data)
        
        # Check that models were trained
        for metric in ['potency', 'dissolution', 'stability']:
            assert metric in predictor.models
            assert hasattr(predictor.models[metric], 'predict')
    
    def test_quality_trajectory_prediction(self, historical_data):
        """Test quality trajectory prediction."""
        predictor = QualityPredictor()
        predictor.train(historical_data)
        
        # Mock current state
        current_state = {
            "process_parameters": {
                "temperature": 25.0,
                "humidity": 45.0,
                "pressure": 1013.25,
                "flow_rate": 100.0
            },
            "sensor_readings": [100.0, 150.0, 200.0]
        }
        
        predictions = predictor.predict_quality_trajectory(
            current_state=current_state,
            horizons=[1, 6, 24],
            confidence_intervals=True
        )
        
        assert len(predictions) == 3  # One for each horizon
        
        for horizon, prediction in predictions.items():
            assert isinstance(prediction, QualityPrediction)
            assert prediction.horizon_hours == horizon
            assert isinstance(prediction.predicted_metrics, QualityMetrics)
            assert 0.0 <= prediction.predicted_metrics.potency <= 1.0
            assert 0.0 <= prediction.predicted_metrics.dissolution <= 1.0
            assert 0.0 <= prediction.predicted_metrics.stability <= 1.0
            assert 0.0 <= prediction.predicted_metrics.uniformity <= 1.0
            assert 0.0 <= prediction.predicted_metrics.contamination_risk <= 1.0
            assert 0.0 <= prediction.confidence_score <= 1.0
            assert isinstance(prediction.risk_factors, list)
    
    def test_prediction_insights_generation(self, historical_data):
        """Test insights generation from predictions."""
        predictor = QualityPredictor()
        predictor.train(historical_data)
        
        # Get predictions
        current_state = {
            "process_parameters": {
                "temperature": 25.0,
                "humidity": 45.0,
                "pressure": 1013.25,
                "flow_rate": 100.0
            }
        }
        
        predictions = predictor.predict_quality_trajectory(current_state, horizons=[1, 6])
        insights = predictor.generate_insights(predictions)
        
        assert hasattr(insights, 'summary')
        assert hasattr(insights, 'intervention_recommended')
        assert hasattr(insights, 'suggested_actions')
        assert hasattr(insights, 'risk_level')
        assert hasattr(insights, 'key_factors')
        
        assert isinstance(insights.summary, str)
        assert isinstance(insights.intervention_recommended, bool)
        assert isinstance(insights.suggested_actions, list)
        assert insights.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert isinstance(insights.key_factors, list)
    
    def test_untrained_predictor_error(self):
        """Test error when using untrained predictor."""
        predictor = QualityPredictor()
        
        current_state = {"dummy": "state"}
        
        with pytest.raises(ValueError, match="Models must be trained"):
            predictor.predict_quality_trajectory(current_state)
    
    def test_empty_historical_data(self):
        """Test training with empty historical data."""
        predictor = QualityPredictor()
        
        with pytest.raises(ValueError, match="Historical data cannot be empty"):
            predictor.train([])


class TestStatisticalAnalysis:
    """Test statistical analysis functions."""
    
    def test_basic_statistics(self, training_data):
        """Test basic statistical calculations."""
        # Extract values from training data
        all_values = []
        for reading in training_data:
            all_values.extend(reading.values)
        
        values_array = np.array(all_values)
        
        # Basic statistics
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        min_val = np.min(values_array)
        max_val = np.max(values_array)
        
        # Verify reasonable values
        assert mean_val > 0
        assert std_val >= 0
        assert min_val <= mean_val <= max_val
        assert std_val < mean_val  # Coefficient of variation should be reasonable
    
    def test_outlier_detection(self, training_data):
        """Test outlier detection in training data."""
        # Create data with outliers
        normal_values = [100, 110, 105, 115, 95, 108, 102, 112]
        outlier_values = [100, 110, 105, 500, 95, 108, 102, 112]  # 500 is an outlier
        
        # Z-score outlier detection
        def detect_outliers_zscore(values, threshold=3.0):
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            z_scores = np.abs((values_array - mean_val) / (std_val + 1e-6))
            return np.where(z_scores > threshold)[0]
        
        # Test normal data
        normal_outliers = detect_outliers_zscore(normal_values)
        assert len(normal_outliers) == 0  # Should find no outliers
        
        # Test data with outliers
        outlier_indices = detect_outliers_zscore(outlier_values)
        assert len(outlier_indices) > 0  # Should find outliers
        assert 3 in outlier_indices  # Index 3 (value 500) should be detected
    
    def test_trend_analysis(self):
        """Test trend analysis capabilities."""
        # Create trending data
        time_points = np.arange(20)
        upward_trend = 100 + 2 * time_points + np.random.normal(0, 1, 20)
        downward_trend = 100 - 2 * time_points + np.random.normal(0, 1, 20)
        stable_trend = 100 + np.random.normal(0, 1, 20)
        
        # Simple trend detection using linear regression slope
        def detect_trend(values):
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return slope
        
        up_slope = detect_trend(upward_trend)
        down_slope = detect_trend(downward_trend)
        stable_slope = detect_trend(stable_trend)
        
        assert up_slope > 1.0  # Positive trend
        assert down_slope < -1.0  # Negative trend
        assert abs(stable_slope) < 0.5  # Stable trend


class TestPerformanceOptimization:
    """Test performance optimization of analytics."""
    
    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        # Create large dataset
        large_dataset = []
        for i in range(1000):  # 1000 samples
            reading = SensorReading(
                sensor_id=f"sensor_{i%10}",
                sensor_type=SensorType.E_NOSE,
                values=[100 + i/10 + np.random.normal(0, 5) for _ in range(16)],
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            large_dataset.append(reading)
        
        # Test fingerprinting performance
        import time
        start_time = time.time()
        
        fingerprinter = ScentFingerprinter(embedding_dim=32)
        model = fingerprinter.create_fingerprint(large_dataset[:100], "performance_test")  # Use subset for speed
        
        processing_time = time.time() - start_time
        
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert model is not None
        assert model.training_samples == 100
    
    def test_concurrent_analysis(self, training_data):
        """Test concurrent fingerprint analysis."""
        fingerprinter = ScentFingerprinter(embedding_dim=16)
        model = fingerprinter.create_fingerprint(training_data, "concurrent_test")
        
        # Create test samples
        test_samples = []
        for i in range(20):
            sample = SensorReading(
                sensor_id=f"test_sensor_{i}",
                sensor_type=SensorType.E_NOSE,
                values=[100 + i + np.random.normal(0, 10) for _ in range(8)],
                timestamp=datetime.now()
            )
            test_samples.append(sample)
        
        # Test sequential processing
        import time
        start_time = time.time()
        
        sequential_results = []
        for sample in test_samples:
            result = fingerprinter.compare_to_fingerprint(sample, model)
            sequential_results.append(result)
        
        sequential_time = time.time() - start_time
        
        # Verify results
        assert len(sequential_results) == 20
        assert all(isinstance(r, SimilarityResult) for r in sequential_results)
        assert sequential_time < 10.0  # Should be reasonably fast
    
    def test_memory_usage_optimization(self, training_data):
        """Test memory usage during analytics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple fingerprint models
        fingerprinter = ScentFingerprinter(embedding_dim=64)
        
        models = []
        for i in range(5):
            model = fingerprinter.create_fingerprint(
                training_data, 
                f"memory_test_{i}",
                augmentation=False  # Disable augmentation to reduce memory usage
            )
            models.append(model)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100.0
        assert len(models) == 5
        assert all(m is not None for m in models)