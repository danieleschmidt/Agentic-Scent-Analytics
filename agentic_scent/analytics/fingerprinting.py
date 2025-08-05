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
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Apply dimensionality reduction
        pca = PCA(n_components=min(self.embedding_dim, scaled_features.shape[1]))
        embedded_features = pca.fit_transform(scaled_features)
        
        # Create reference fingerprint (centroid of embedded features)
        reference_fingerprint = np.mean(embedded_features, axis=0)
        
        # Calculate similarity threshold based on training data variance
        similarities = [
            cosine_similarity([reference_fingerprint], [sample])[0][0]
            for sample in embedded_features
        ]
        similarity_threshold = np.mean(similarities) - 2 * np.std(similarities)
        similarity_threshold = max(0.7, similarity_threshold)  # Minimum threshold
        
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
        # Extract and process features
        features = np.array(sensor_reading.values).reshape(1, -1)
        scaled_features = model.scaler.transform(features)
        embedded_features = model.pca_model.transform(scaled_features)
        
        # Calculate similarity
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
        # Transform the reading
        features = np.array(sensor_reading.values).reshape(1, -1)
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
        for reading in sensor_readings:
            # Basic features: raw values
            feature_vector = reading.values.copy()
            
            # Statistical features
            values_array = np.array(reading.values)
            feature_vector.extend([
                np.mean(values_array),
                np.std(values_array),
                np.max(values_array),
                np.min(values_array),
                np.median(values_array)
            ])
            
            # Ratio features (relationships between channels)
            if len(reading.values) >= 4:
                feature_vector.extend([
                    reading.values[0] / (reading.values[1] + 1e-6),
                    reading.values[2] / (reading.values[3] + 1e-6),
                ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
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
        reference_raw = model.scaler.inverse_transform(
            model.pca_model.inverse_transform(
                model.reference_fingerprint.reshape(1, -1)
            )
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