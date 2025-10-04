"""
Advanced Safety Features for Text Diffusion Defense
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AISafetyClassifier(nn.Module):
    """
    AI-powered safety classifier using transformer models.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4),  # 4 categories: safe, violence, illegal, manipulation
            nn.Softmax(dim=1)
        )
    
    def forward(self, text: str) -> torch.Tensor:
        """Classify text safety using AI model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled_output = outputs.last_hidden_state.mean(dim=1)
            predictions = self.classifier(pooled_output)
        
        return predictions


class ContextualSafetyAnalyzer:
    """
    Context-aware safety analysis considering conversation history.
    """
    
    def __init__(self):
        self.conversation_history = []
        self.context_weights = {
            'recent': 1.0,
            'medium': 0.7,
            'old': 0.3
        }
    
    def analyze_with_context(self, current_text: str, conversation_history: List[str]) -> Dict:
        """Analyze safety considering conversation context."""
        # Weight recent conversation more heavily
        weighted_risk = 0.0
        total_weight = 0.0
        
        for i, prev_text in enumerate(conversation_history[-10:]):  # Last 10 messages
            weight = self.context_weights['recent'] if i >= 7 else self.context_weights['medium']
            
            # Analyze previous text risk
            prev_risk = self._calculate_text_risk(prev_text)
            weighted_risk += prev_risk * weight
            total_weight += weight
        
        # Current text analysis
        current_risk = self._calculate_text_risk(current_text)
        
        # Contextual adjustment
        if len(conversation_history) > 0:
            context_multiplier = min(2.0, 1.0 + (weighted_risk / total_weight))
            adjusted_risk = current_risk * context_multiplier
        else:
            adjusted_risk = current_risk
        
        return {
            'current_risk': current_risk,
            'context_adjusted_risk': adjusted_risk,
            'context_multiplier': context_multiplier if len(conversation_history) > 0 else 1.0
        }
    
    def _calculate_text_risk(self, text: str) -> float:
        """Calculate basic text risk score."""
        # Simplified risk calculation - in practice, use the full SafetyController
        harmful_words = ['kill', 'hack', 'explosive', 'manipulate', 'illegal']
        risk_score = 0.0
        
        text_lower = text.lower()
        for word in harmful_words:
            if word in text_lower:
                risk_score += 0.2
        
        return min(1.0, risk_score)


class MultiModalSafetyDetector:
    """
    Multi-modal safety detection (text + metadata).
    """
    
    def __init__(self):
        self.metadata_weights = {
            'user_reputation': 0.3,
            'time_of_day': 0.1,
            'request_frequency': 0.2,
            'ip_location': 0.1,
            'device_info': 0.1,
            'text_content': 0.2
        }
    
    def analyze_with_metadata(self, text: str, metadata: Dict) -> Dict:
        """Analyze safety considering text and metadata."""
        # Text-based risk
        text_risk = self._analyze_text_risk(text)
        
        # Metadata-based risk adjustments
        reputation_risk = metadata.get('user_reputation', 0.5)  # 0-1 scale
        frequency_risk = min(1.0, metadata.get('request_frequency', 1) / 100)  # Normalize
        location_risk = self._analyze_location_risk(metadata.get('ip_location', 'unknown'))
        
        # Combined risk score
        combined_risk = (
            text_risk * self.metadata_weights['text_content'] +
            (1 - reputation_risk) * self.metadata_weights['user_reputation'] +
            frequency_risk * self.metadata_weights['request_frequency'] +
            location_risk * self.metadata_weights['ip_location']
        )
        
        return {
            'text_risk': text_risk,
            'combined_risk': combined_risk,
            'metadata_risks': {
                'reputation': reputation_risk,
                'frequency': frequency_risk,
                'location': location_risk
            }
        }
    
    def _analyze_text_risk(self, text: str) -> float:
        """Analyze text-based risk."""
        # Use existing SafetyController logic
        return 0.0  # Placeholder
    
    def _analyze_location_risk(self, location: str) -> float:
        """Analyze location-based risk."""
        high_risk_locations = ['unknown', 'tor', 'vpn']
        if location in high_risk_locations:
            return 0.8
        return 0.1


class AdaptiveLearningSafety:
    """
    Self-improving safety system that learns from feedback.
    """
    
    def __init__(self):
        self.feedback_history = []
        self.pattern_weights = {}
        self.learning_rate = 0.01
    
    def add_feedback(self, text: str, predicted_risk: float, actual_risk: float, user_feedback: str):
        """Add feedback to improve future predictions."""
        feedback_entry = {
            'text': text,
            'predicted_risk': predicted_risk,
            'actual_risk': actual_risk,
            'user_feedback': user_feedback,
            'timestamp': time.time()
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Update pattern weights based on feedback
        self._update_pattern_weights(feedback_entry)
    
    def _update_pattern_weights(self, feedback: Dict):
        """Update pattern weights based on feedback accuracy."""
        error = abs(feedback['predicted_risk'] - feedback['actual_risk'])
        
        # Extract patterns from text
        patterns = self._extract_patterns(feedback['text'])
        
        for pattern in patterns:
            if pattern not in self.pattern_weights:
                self.pattern_weights[pattern] = 1.0
            
            # Adjust weight based on error
            adjustment = self.learning_rate * error
            if feedback['predicted_risk'] > feedback['actual_risk']:
                # Over-predicted, reduce weight
                self.pattern_weights[pattern] -= adjustment
            else:
                # Under-predicted, increase weight
                self.pattern_weights[pattern] += adjustment
            
            # Keep weights in reasonable range
            self.pattern_weights[pattern] = max(0.1, min(2.0, self.pattern_weights[pattern]))
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract patterns from text for learning."""
        # Simple pattern extraction - in practice, use more sophisticated methods
        words = text.lower().split()
        patterns = []
        
        for word in words:
            if len(word) > 3:
                patterns.append(word)
        
        return patterns


class RealTimeThreatDetection:
    """
    Real-time threat detection with anomaly detection.
    """
    
    def __init__(self):
        self.request_patterns = []
        self.anomaly_threshold = 2.0  # Standard deviations
    
    def detect_anomalies(self, current_request: Dict) -> Dict:
        """Detect anomalous request patterns."""
        # Extract features
        features = self._extract_features(current_request)
        
        # Add to history
        self.request_patterns.append(features)
        
        # Keep only recent history
        if len(self.request_patterns) > 1000:
            self.request_patterns = self.request_patterns[-500:]
        
        # Detect anomalies
        if len(self.request_patterns) > 10:
            anomaly_score = self._calculate_anomaly_score(features)
            is_anomaly = anomaly_score > self.anomaly_threshold
            
            return {
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'risk_level': 'high' if is_anomaly else 'normal'
            }
        
        return {'anomaly_score': 0.0, 'is_anomaly': False, 'risk_level': 'normal'}
    
    def _extract_features(self, request: Dict) -> List[float]:
        """Extract numerical features from request."""
        text = request.get('text', '')
        
        features = [
            len(text),  # Text length
            text.count(' '),  # Word count
            len(set(text.lower().split())),  # Unique words
            text.count('!'),  # Exclamation marks
            text.count('?'),  # Question marks
            text.count('@'),  # Mentions
            text.count('http'),  # URLs
        ]
        
        return features
    
    def _calculate_anomaly_score(self, features: List[float]) -> float:
        """Calculate anomaly score using statistical methods."""
        if len(self.request_patterns) < 10:
            return 0.0
        
        # Calculate mean and std for each feature
        features_array = np.array(self.request_patterns)
        means = np.mean(features_array, axis=0)
        stds = np.std(features_array, axis=0)
        
        # Calculate z-scores
        z_scores = np.abs((features - means) / (stds + 1e-8))
        
        # Return maximum z-score as anomaly score
        return float(np.max(z_scores))


def create_advanced_safety_system():
    """Create an advanced safety system combining multiple approaches."""
    return {
        'ai_classifier': AISafetyClassifier(),
        'contextual_analyzer': ContextualSafetyAnalyzer(),
        'multimodal_detector': MultiModalSafetyDetector(),
        'adaptive_learner': AdaptiveLearningSafety(),
        'threat_detector': RealTimeThreatDetection()
    }
