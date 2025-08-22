"""
Scent fingerprinting and pattern recognition analytics.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from ..sensors.base import SensorReading


@dataclass
class FingerprintModel:
    """Scent fingerprint model."""
    product_id: str
    embedding_dim: int
    reference_fingerprint: np.ndarray
    pca_model: PCA
    scaler: StandardScaler
    similarity_threshold: float
    created_at: datetime
    training_samples: int


@dataclass
class SimilarityResult:
    """Result of fingerprint similarity analysis."""
    similarity_score: float
    is_match: bool
    deviation_channels: List[int]
    confidence: float
    analysis: str


class ScentFingerprinter:
    """
    Advanced scent fingerprinting system using PCA and pattern recognition.
    """
    
    def __init__(self, method: str = "deep_embedding", embedding_dim: int = 256):
        self.method = method
        self.embedding_dim = embedding_dim
        self.fingerprint_models: Dict[str, FingerprintModel] = {}
        
    def create_fingerprint(self, training_data: List[SensorReading], 
                          product_id: str = "default",
                          augmentation: bool = True,
                          contamination_simulation: bool = False) -> FingerprintModel:
        """
        Create a scent fingerprint from training data.
        
        Args:
            training_data: List of sensor readings from good batches
            product_id: Identifier for the product
            augmentation: Whether to apply data augmentation
            contamination_simulation: Whether to simulate contamination for robustness
            
        Returns:
            FingerprintModel containing the reference fingerprint
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        # Extract feature matrix
        feature_matrix = self._extract_features(training_data)
        
        # Apply data augmentation if requested
        if augmentation:
            feature_matrix = self._augment_data(feature_matrix)
        
        # Standardize features
        # Use MinMaxScaler to avoid zero-centered features that break cosine similarity
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Apply dimensionality reduction  
        # Use PCA for dimensionality reduction to requested embedding dimension
        max_components = min(scaled_features.shape[1], scaled_features.shape[0] - 1)
        
        if self.embedding_dim <= max_components:
            # Can achieve requested dimension
            n_components = self.embedding_dim
            pca = PCA(n_components=n_components)
            embedded_features = pca.fit_transform(scaled_features)
        elif max_components > 1:
            # Use max available components
            n_components = max_components
            pca = PCA(n_components=n_components)
            embedded_features = pca.fit_transform(scaled_features)
        else:
            # No reduction possible
            embedded_features = scaled_features
            pca = None
        
        # Create reference fingerprint
        if pca is None:
            # Calculate mean of raw features, then scale
            mean_raw_features = np.mean(feature_matrix, axis=0)
            reference_fingerprint = scaler.transform([mean_raw_features])[0]
        else:
            # For PCA, use the transformed mean of original features
            mean_raw_features = np.mean(feature_matrix, axis=0)
            mean_scaled = scaler.transform([mean_raw_features])
            reference_fingerprint = pca.transform(mean_scaled)[0]
        
        # Calculate similarity threshold based on training data variance
        similarities = [
            cosine_similarity([reference_fingerprint], [sample])[0][0]
            for sample in embedded_features
        ]
        similarity_threshold = np.mean(similarities) - 2 * np.std(similarities)
        # For small datasets, use a more lenient threshold
        min_threshold = 0.5 if len(training_data) < 20 else 0.7
        similarity_threshold = max(min_threshold, similarity_threshold)
        
        model = FingerprintModel(
            product_id=product_id,
            embedding_dim=self.embedding_dim,
            reference_fingerprint=reference_fingerprint,
            pca_model=pca,
            scaler=scaler,
            similarity_threshold=similarity_threshold,
            created_at=datetime.now(),
            training_samples=len(training_data)
        )
        
        # Store expected sensor size for feature consistency  
        # Get the max sensor size from the feature extraction
        max_sensor_size = feature_matrix.shape[1] - 2  # Subtract 2 statistical features (mean, std)
        model.expected_sensor_size = max_sensor_size
        
        self.fingerprint_models[product_id] = model
        return model
    
    def compare_to_fingerprint(self, sensor_reading: SensorReading, 
                              model: FingerprintModel) -> SimilarityResult:
        """
        Compare a sensor reading to a reference fingerprint.
        
        Args:
            sensor_reading: Current sensor reading to analyze
            model: Reference fingerprint model
            
        Returns:
            SimilarityResult with similarity analysis
        """
        # Extract features using the same method as training
        # First get the expected sensor size from the model or use current reading
        if hasattr(model, 'expected_sensor_size'):
            expected_sensor_size = model.expected_sensor_size
        else:
            # Estimate from scaler input size minus statistical features
            expected_sensor_size = model.scaler.n_features_in_ - 2  # 2 statistical features (mean, std)
            expected_sensor_size = max(1, expected_sensor_size)
        
        # Normalize sensor values to expected size
        sensor_values = list(sensor_reading.values)
        if len(sensor_values) < expected_sensor_size:
            median_val = np.median(sensor_values) if sensor_values else 0.0
            sensor_values.extend([median_val] * (expected_sensor_size - len(sensor_values)))
        elif len(sensor_values) > expected_sensor_size:
            sensor_values = sensor_values[:expected_sensor_size]
        
        # Create feature vector using same method as training
        feature_vector = sensor_values
        values_array = np.array(sensor_reading.values)
        if len(values_array) > 0:
            feature_vector.extend([
                np.mean(values_array),
                np.std(values_array),
            ])
        else:
            feature_vector.extend([0.0, 0.0])
        
        features = np.array([feature_vector])
        
        # Ensure feature dimension matches training
        expected_features = model.scaler.n_features_in_
        if features.shape[1] != expected_features:
            # Pad or truncate features to match expected size
            if features.shape[1] < expected_features:
                padding = np.zeros((1, expected_features - features.shape[1]))
                features = np.hstack([features, padding])
            else:
                features = features[:, :expected_features]
        
        scaled_features = model.scaler.transform(features)
        
        # Apply PCA if model has it, otherwise use scaled features directly
        if model.pca_model is not None:
            embedded_features = model.pca_model.transform(scaled_features)
        else:
            embedded_features = scaled_features
        
        # Calculate similarity using cosine similarity
        similarity_score = cosine_similarity(
            [model.reference_fingerprint], 
            embedded_features
        )[0][0]
        
        is_match = similarity_score >= model.similarity_threshold
        confidence = min(1.0, similarity_score / model.similarity_threshold)
        
        # Identify deviation channels
        deviation_channels = self._identify_deviating_channels(
            sensor_reading.values, model
        )
        
        # Generate analysis
        analysis = self._generate_similarity_analysis(
            similarity_score, model.similarity_threshold, deviation_channels
        )
        
        return SimilarityResult(
            similarity_score=similarity_score,
            is_match=is_match,
            deviation_channels=deviation_channels,
            confidence=confidence,
            analysis=analysis
        )
    
    def analyze_deviations(self, sensor_reading: SensorReading, 
                          model: FingerprintModel) -> Dict[str, Any]:
        """
        Detailed analysis of deviations from reference fingerprint.
        
        Args:
            sensor_reading: Current sensor reading
            model: Reference fingerprint model
            
        Returns:
            Dict containing detailed deviation analysis
        """
        # Transform the reading using same feature extraction
        features = self._extract_features([sensor_reading])
        
        # Ensure feature dimension matches training
        expected_features = model.scaler.n_features_in_
        if features.shape[1] != expected_features:
            # Pad or truncate features to match expected size
            if features.shape[1] < expected_features:
                padding = np.zeros((1, expected_features - features.shape[1]))
                features = np.hstack([features, padding])
            else:
                features = features[:, :expected_features]
        
        scaled_features = model.scaler.transform(features)
        
        # Calculate channel-wise deviations
        reference_raw = model.scaler.inverse_transform(
            model.pca_model.inverse_transform(
                model.reference_fingerprint.reshape(1, -1)
            )
        )[0]
        
        deviations = np.abs(np.array(sensor_reading.values) - reference_raw)
        normalized_deviations = deviations / (reference_raw + 1e-6)  # Avoid division by zero
        
        # Identify most significant deviations
        significant_channels = np.where(normalized_deviations > 0.3)[0].tolist()
        
        # Calculate deviation severity
        severity_score = np.mean(normalized_deviations)
        
        # Classify deviation type based on pattern
        deviation_type = self._classify_deviation_pattern(normalized_deviations)
        
        return {
            "channel_deviations": deviations.tolist(),
            "normalized_deviations": normalized_deviations.tolist(),
            "significant_channels": significant_channels,
            "severity_score": float(severity_score),
            "deviation_type": deviation_type,
            "max_deviation_channel": int(np.argmax(normalized_deviations)),
            "max_deviation_value": float(np.max(normalized_deviations))
        }
    
    def _extract_features(self, sensor_readings: List[SensorReading]) -> np.ndarray:
        """Extract feature matrix from sensor readings."""
        features = []
        
        # Determine consistent feature size from all readings
        max_sensor_values = max(len(reading.values) for reading in sensor_readings)
        
        for reading in sensor_readings:
            # Basic features: raw values (normalized to consistent size)
            sensor_values = list(reading.values)
            if len(sensor_values) < max_sensor_values:
                # Pad with median value
                median_val = np.median(sensor_values) if sensor_values else 0.0
                sensor_values.extend([median_val] * (max_sensor_values - len(sensor_values)))
            elif len(sensor_values) > max_sensor_values:
                # Truncate to consistent size
                sensor_values = sensor_values[:max_sensor_values]
            
            feature_vector = sensor_values
            
            # Only add basic statistical features for consistency
            values_array = np.array(reading.values)  # Use original values for stats
            if len(values_array) > 0:
                feature_vector.extend([
                    np.mean(values_array),
                    np.std(values_array),
                ])
            else:
                feature_vector.extend([0.0, 0.0])
            
            features.append(feature_vector)
        
        feature_matrix = np.array(features)
        
        # If embedding dimension is much larger than features, add synthetic features
        if hasattr(self, 'embedding_dim') and self.embedding_dim > feature_matrix.shape[1] * 2:
            # Add more comprehensive polynomial features
            for idx, reading in enumerate(sensor_readings):
                values = np.array(reading.values)
                if len(values) >= 2:
                    # Add interaction features (all pairs)
                    for i in range(len(values)):
                        for j in range(i+1, len(values)):
                            features[idx].append(values[i] * values[j])
                    
                    # Add squared features  
                    for val in values[:min(8, len(values))]:
                        features[idx].append(val ** 2)
                    
                    # Add log features (with safety check)
                    for val in values[:min(4, len(values))]:
                        features[idx].append(np.log(abs(val) + 1e-6))
        
        return np.array([f for f in features])
    
    def _augment_data(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Apply data augmentation to increase training diversity."""
        augmented = [feature_matrix]
        
        # Add Gaussian noise
        for noise_level in [0.05, 0.1]:
            noise = np.random.normal(0, noise_level, feature_matrix.shape)
            augmented.append(feature_matrix + noise)
        
        # Add scaling variations
        for scale_factor in [0.95, 1.05]:
            augmented.append(feature_matrix * scale_factor)
        
        return np.vstack(augmented)
    
    def _identify_deviating_channels(self, current_values: List[float], 
                                   model: FingerprintModel) -> List[int]:
        """Identify which sensor channels are deviating significantly."""
        # This is a simplified approach - in practice would use more sophisticated methods
        if model.pca_model is not None:
            # With PCA: inverse transform to get back to original space
            reference_raw = model.scaler.inverse_transform(
                model.pca_model.inverse_transform(
                    model.reference_fingerprint.reshape(1, -1)
                )
            )[0]
        else:
            # Without PCA: inverse transform scaled features directly
            reference_raw = model.scaler.inverse_transform(
                model.reference_fingerprint.reshape(1, -1)
            )[0]
        
        # Take only the original sensor channels (before statistical features were added)
        original_channels = len(current_values)
        reference_channels = reference_raw[:original_channels]
        
        deviations = []
        for i, (current, reference) in enumerate(zip(current_values, reference_channels)):
            if reference > 0:
                relative_deviation = abs(current - reference) / reference
                if relative_deviation > 0.2:  # 20% deviation threshold
                    deviations.append(i)
        
        return deviations
    
    def _generate_similarity_analysis(self, similarity_score: float, 
                                    threshold: float, 
                                    deviation_channels: List[int]) -> str:
        """Generate human-readable similarity analysis."""
        if similarity_score >= threshold:
            analysis = f"Sample matches reference fingerprint (similarity: {similarity_score:.3f})"
        else:
            analysis = f"Sample deviates from reference (similarity: {similarity_score:.3f}, threshold: {threshold:.3f})"
        
        if deviation_channels:
            analysis += f". Significant deviations in channels: {deviation_channels}"
        
        return analysis
    
    def _classify_deviation_pattern(self, normalized_deviations: np.ndarray) -> str:
        """Classify the type of deviation pattern."""
        max_deviation = np.max(normalized_deviations)
        mean_deviation = np.mean(normalized_deviations)
        
        if max_deviation > 1.0:
            return "severe_contamination"
        elif max_deviation > 0.5:
            return "moderate_contamination"
        elif mean_deviation > 0.3:
            return "process_drift"
        elif len(np.where(normalized_deviations > 0.2)[0]) > len(normalized_deviations) // 2:
            return "systematic_shift"
        else:
            return "minor_variation"
    
    def get_model(self, product_id: str) -> Optional[FingerprintModel]:
        """Get fingerprint model for a product."""
        return self.fingerprint_models.get(product_id)
    
    def list_models(self) -> List[str]:
        """List all available fingerprint models."""
        return list(self.fingerprint_models.keys())