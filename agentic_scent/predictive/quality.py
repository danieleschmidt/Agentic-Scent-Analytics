"""
Predictive quality analytics using machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from ..sensors.base import SensorReading


@dataclass
class QualityMetrics:
    """Quality metrics for prediction."""
    potency: float = 1.0
    dissolution: float = 1.0
    stability: float = 1.0
    uniformity: float = 1.0
    contamination_risk: float = 0.0


@dataclass
class QualityPrediction:
    """Quality prediction result."""
    horizon_hours: int
    predicted_metrics: QualityMetrics
    confidence_intervals: Dict[str, Tuple[float, float]]
    risk_factors: List[str]
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionInsights:
    """Insights generated from quality predictions."""
    summary: str
    intervention_recommended: bool
    suggested_actions: List[str]
    risk_level: str
    key_factors: List[str]


class QualityPredictor:
    """
    Predictive quality analytics system using machine learning.
    """
    
    def __init__(self, model: str = "random_forest", 
                 features: List[str] = None):
        self.model_type = model
        self.features = features or ['scent_profile', 'process_params', 'ambient_conditions']
        
        # ML models for different metrics
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Historical data storage
        self.training_data = []
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for each quality metric."""
        quality_metrics = ['potency', 'dissolution', 'stability', 'uniformity', 'contamination_risk']
        
        for metric in quality_metrics:
            if self.model_type == "random_forest":
                self.models[metric] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            else:
                # Default to random forest
                self.models[metric] = RandomForestRegressor(n_estimators=100, random_state=42)
            
            self.scalers[metric] = StandardScaler()
    
    def train(self, historical_data: List[Dict[str, Any]], 
              quality_metrics: List[str] = None):
        """
        Train predictive models on historical data.
        
        Args:
            historical_data: Historical sensor readings with quality outcomes
            quality_metrics: List of quality metrics to predict
        """
        if not historical_data:
            raise ValueError("Historical data cannot be empty")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(historical_data)
        
        # Extract features and targets
        features_df = self._extract_features_from_history(df)
        
        # Train each quality metric model
        trained_metrics = quality_metrics or ['potency', 'dissolution', 'stability', 'uniformity', 'contamination_risk']
        
        for metric in trained_metrics:
            if metric in df.columns:
                # Prepare data
                X = features_df.values
                y = df[metric].values
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                X_train_scaled = self.scalers[metric].fit_transform(X_train)
                X_test_scaled = self.scalers[metric].transform(X_test)
                
                # Train model
                self.models[metric].fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = self.models[metric].predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                print(f"Model {metric}: R² = {r2:.3f}, MSE = {mse:.3f}")
        
        self.is_trained = True
        self.training_data = historical_data
    
    def predict_quality_trajectory(self, current_state: Dict[str, Any],
                                 horizons: List[int] = [1, 6, 24],
                                 confidence_intervals: bool = True) -> Dict[int, QualityPrediction]:
        """
        Predict quality trajectory over multiple time horizons.
        
        Args:
            current_state: Current factory state and sensor readings
            horizons: Prediction horizons in hours
            confidence_intervals: Whether to calculate confidence intervals
            
        Returns:
            Dict mapping horizon to prediction results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = {}
        
        for horizon in horizons:
            # Extract features for current state
            features = self._extract_features_from_state(current_state, horizon)
            
            # Make predictions for each quality metric
            predicted_metrics = {}
            confidence_intervals_dict = {}
            
            for metric, model in self.models.items():
                if metric in self.scalers:
                    # Scale features
                    features_scaled = self.scalers[metric].transform([features])
                    
                    # Make prediction
                    prediction = model.predict(features_scaled)[0]
                    predicted_metrics[metric] = max(0.0, min(1.0, prediction))
                    
                    # Calculate confidence intervals if requested
                    if confidence_intervals and hasattr(model, 'estimators_'):
                        # Use ensemble predictions for confidence intervals
                        ensemble_predictions = [
                            estimator.predict(features_scaled)[0] 
                            for estimator in model.estimators_
                        ]
                        ci_lower = np.percentile(ensemble_predictions, 25)
                        ci_upper = np.percentile(ensemble_predictions, 75)
                        confidence_intervals_dict[metric] = (ci_lower, ci_upper)
            
            # Create quality metrics object
            quality_metrics = QualityMetrics(**predicted_metrics) if predicted_metrics else QualityMetrics()
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(quality_metrics, horizon)
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence_score(quality_metrics, horizon)
            
            predictions[horizon] = QualityPrediction(
                horizon_hours=horizon,
                predicted_metrics=quality_metrics,
                confidence_intervals=confidence_intervals_dict,
                risk_factors=risk_factors,
                confidence_score=confidence_score
            )
        
        return predictions
    
    def generate_insights(self, predictions: Dict[int, QualityPrediction]) -> PredictionInsights:
        """
        Generate actionable insights from quality predictions.
        
        Args:
            predictions: Quality predictions for different horizons
            
        Returns:
            PredictionInsights with recommendations
        """
        if not predictions:
            return PredictionInsights(
                summary="No predictions available",
                intervention_recommended=False,
                suggested_actions=[],
                risk_level="UNKNOWN",
                key_factors=[]
            )
        
        # Analyze predictions
        shortest_horizon = min(predictions.keys())
        longest_horizon = max(predictions.keys())
        
        short_term = predictions[shortest_horizon]
        long_term = predictions[longest_horizon]
        
        # Generate summary
        summary = self._generate_prediction_summary(short_term, long_term)
        
        # Determine intervention need
        intervention_recommended = (
            short_term.confidence_score < 0.7 or
            any(getattr(short_term.predicted_metrics, metric, 1.0) < 0.8 
                for metric in ['potency', 'dissolution', 'stability', 'uniformity']) or
            getattr(short_term.predicted_metrics, 'contamination_risk', 0.0) > 0.3
        )
        
        # Generate suggested actions
        suggested_actions = self._generate_action_recommendations(predictions)
        
        # Determine risk level
        risk_level = self._assess_risk_level(predictions)
        
        # Identify key factors
        key_factors = self._identify_key_factors(predictions)
        
        return PredictionInsights(
            summary=summary,
            intervention_recommended=intervention_recommended,
            suggested_actions=suggested_actions,
            risk_level=risk_level,
            key_factors=key_factors
        )
    
    def _extract_features_from_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from historical data DataFrame."""
        # Mock feature extraction - would be more sophisticated in practice
        features = pd.DataFrame()
        
        # Basic statistical features
        if 'sensor_values' in df.columns:
            # Assume sensor_values is a list of values
            features['mean_sensor'] = df['sensor_values'].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
            features['std_sensor'] = df['sensor_values'].apply(lambda x: np.std(x) if isinstance(x, list) else 0)
            features['max_sensor'] = df['sensor_values'].apply(lambda x: np.max(x) if isinstance(x, list) else x)
        
        # Process parameters
        features['temperature'] = df.get('temperature', 25.0)
        features['humidity'] = df.get('humidity', 45.0)
        features['pressure'] = df.get('pressure', 1013.25)
        
        # Time-based features
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            features['hour_of_day'] = timestamps.dt.hour
            features['day_of_week'] = timestamps.dt.dayofweek
        
        return features.fillna(0)
    
    def _extract_features_from_state(self, state: Dict[str, Any], horizon: int) -> List[float]:
        """Extract features from current state for prediction."""
        features = []
        
        # Process parameters
        process_params = state.get('process_parameters', {})
        features.extend([
            process_params.get('temperature', 25.0),
            process_params.get('humidity', 45.0),
            process_params.get('pressure', 1013.25),
            process_params.get('flow_rate', 100.0)
        ])
        
        # Mock sensor features - would use actual sensor readings
        features.extend([500.0, 50.0, 1000.0])  # mean, std, max
        
        # Time-based features
        current_time = datetime.now()
        features.extend([
            current_time.hour,
            current_time.weekday(),
            horizon  # prediction horizon as a feature
        ])
        
        return features
    
    def _identify_risk_factors(self, quality_metrics: QualityMetrics, horizon: int) -> List[str]:
        """Identify risk factors based on predicted quality metrics."""
        risk_factors = []
        
        if quality_metrics.potency < 0.9:
            risk_factors.append("potency_decline")
        
        if quality_metrics.dissolution < 0.8:
            risk_factors.append("dissolution_issues")
        
        if quality_metrics.stability < 0.85:
            risk_factors.append("stability_concerns")
        
        if quality_metrics.uniformity < 0.9:
            risk_factors.append("uniformity_problems")
        
        if quality_metrics.contamination_risk > 0.2:
            risk_factors.append("contamination_risk")
        
        if horizon > 12:
            risk_factors.append("extended_horizon_uncertainty")
        
        return risk_factors
    
    def _calculate_confidence_score(self, quality_metrics: QualityMetrics, horizon: int) -> float:
        """Calculate overall confidence score for predictions."""
        base_confidence = 0.9
        
        # Reduce confidence for longer horizons
        horizon_penalty = min(0.3, horizon * 0.02)
        
        # Reduce confidence for poor quality predictions
        quality_score = np.mean([
            quality_metrics.potency,
            quality_metrics.dissolution,
            quality_metrics.stability,
            quality_metrics.uniformity
        ])
        
        quality_penalty = (1.0 - quality_score) * 0.2
        
        # Increase penalty for contamination risk
        contamination_penalty = quality_metrics.contamination_risk * 0.3
        
        confidence = base_confidence - horizon_penalty - quality_penalty - contamination_penalty
        return max(0.1, min(1.0, confidence))
    
    def _generate_prediction_summary(self, short_term: QualityPrediction, 
                                   long_term: QualityPrediction) -> str:
        """Generate summary of prediction results."""
        st_metrics = short_term.predicted_metrics
        lt_metrics = long_term.predicted_metrics
        
        summary = f"Short-term ({short_term.horizon_hours}h) outlook: "
        
        if short_term.confidence_score > 0.8:
            summary += "Quality expected to remain stable. "
        elif short_term.confidence_score > 0.6:
            summary += "Minor quality variations anticipated. "
        else:
            summary += "Significant quality concerns detected. "
        
        # Identify trends
        potency_trend = lt_metrics.potency - st_metrics.potency
        if abs(potency_trend) > 0.1:
            direction = "improving" if potency_trend > 0 else "declining"
            summary += f"Potency trend: {direction}. "
        
        if st_metrics.contamination_risk > 0.3:
            summary += "High contamination risk detected. "
        
        return summary
    
    def _generate_action_recommendations(self, predictions: Dict[int, QualityPrediction]) -> List[str]:
        """Generate action recommendations based on predictions."""
        actions = []
        
        shortest_horizon = min(predictions.keys())
        short_term = predictions[shortest_horizon]
        
        if short_term.predicted_metrics.contamination_risk > 0.3:
            actions.append("Increase sampling frequency and implement enhanced cleaning protocols")
        
        if short_term.predicted_metrics.potency < 0.85:
            actions.append("Review raw material quality and adjust process parameters")
        
        if short_term.predicted_metrics.uniformity < 0.8:
            actions.append("Check mixing equipment and calibrate dosing systems")
        
        if short_term.confidence_score < 0.6:
            actions.append("Implement additional monitoring and consider manual quality checks")
        
        if not actions:
            actions.append("Continue standard monitoring procedures")
        
        return actions
    
    def _assess_risk_level(self, predictions: Dict[int, QualityPrediction]) -> str:
        """Assess overall risk level based on predictions."""
        shortest_horizon = min(predictions.keys())
        short_term = predictions[shortest_horizon]
        
        risk_score = 0
        
        # Add risk based on quality metrics
        metrics = short_term.predicted_metrics
        quality_score = np.mean([metrics.potency, metrics.dissolution, metrics.stability, metrics.uniformity])
        
        if quality_score < 0.7:
            risk_score += 3
        elif quality_score < 0.8:
            risk_score += 2
        elif quality_score < 0.9:
            risk_score += 1
        
        # Add risk based on contamination
        if metrics.contamination_risk > 0.4:
            risk_score += 3
        elif metrics.contamination_risk > 0.2:
            risk_score += 2
        elif metrics.contamination_risk > 0.1:
            risk_score += 1
        
        # Add risk based on confidence
        if short_term.confidence_score < 0.5:
            risk_score += 2
        elif short_term.confidence_score < 0.7:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 5:
            return "CRITICAL"
        elif risk_score >= 3:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _identify_key_factors(self, predictions: Dict[int, QualityPrediction]) -> List[str]:
        """Identify key factors influencing quality predictions."""
        factors = []
        
        # This would use feature importance from trained models
        # For now, return common factors
        factors.extend([
            "process_temperature",
            "raw_material_quality",
            "equipment_performance",
            "environmental_conditions"
        ])
        
        return factors