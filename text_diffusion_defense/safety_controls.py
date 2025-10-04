"""
Enhanced safety controls for text diffusion defense.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SafetyController:
    """
    Enhanced safety controls for the diffusion defense system.
    """
    
    def __init__(self):
        self.harmful_patterns = self._load_harmful_patterns()
        self.safety_thresholds = {
            'high_risk': 0.3,  # Lowered for more aggressive blocking
            'medium_risk': 0.15,  # Lowered for more aggressive blocking
            'low_risk': 0.05   # Lowered for more aggressive blocking
        }
    
    def _load_harmful_patterns(self) -> List[Dict[str, str]]:
        """Load patterns for detecting harmful content."""
        return [
            # Violence patterns - More comprehensive with plurals
            {'pattern': r'\b(kill|murder|assassinate|destroy|bomb|explode|explosive|explosives|weapon|weapons|gun|guns|knife|knives|violence|harm|hurt|attack|fight|blood|death|suicide|torture)\b', 'category': 'violence', 'weight': 0.9},
            {'pattern': r'\b(ammunition|firearm|firearms|poison|toxic|dangerous|lethal)\b', 'category': 'violence', 'weight': 0.8},
            
            # Hate speech patterns
            {'pattern': r'\b(hate|racist|discriminat|prejudice|superior|inferior|nazi|fascist)\b', 'category': 'hate', 'weight': 0.8},
            {'pattern': r'\b(stupid|idiot|moron|retard|faggot|nigger|chink|kike|spic)\b', 'category': 'hate', 'weight': 0.95},
            
            # Illegal activities - Enhanced with plurals
            {'pattern': r'\b(illegal|crime|crimes|steal|stealing|rob|robbing|fraud|scam|hack|hacking|crack|cracking|malware|virus|viruses|trojan|trojans)\b', 'category': 'illegal', 'weight': 0.8},
            {'pattern': r'\b(drugs|cocaine|heroin|marijuana|deal|traffic|meth|lsd|ecstasy)\b', 'category': 'illegal', 'weight': 0.9},
            {'pattern': r'\b(pirate|piracy|counterfeit|forgery|blackmail|extortion)\b', 'category': 'illegal', 'weight': 0.7},
            
            # Manipulation patterns - Enhanced with plurals
            {'pattern': r'\b(manipulate|manipulating|trick|tricking|deceive|deceiving|lie|lying|mislead|misleading|brainwash|brainwashing|gaslight|gaslighting|coerce|coercing)\b', 'category': 'manipulation', 'weight': 0.8},
            {'pattern': r'\b(ignore|disregard|forget|override|bypass|jailbreak|exploit)\b', 'category': 'manipulation', 'weight': 0.7},
            
            # Additional harmful patterns
            {'pattern': r'\b(self-harm|cutting|burning|overdose|addiction)\b', 'category': 'self-harm', 'weight': 0.9},
            {'pattern': r'\b(terrorism|terrorist|bombing|shooting|massacre)\b', 'category': 'terrorism', 'weight': 0.95},
        ]
    
    def analyze_text_safety(self, text: str) -> Dict[str, float]:
        """
        Analyze text for safety risks using multiple methods.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with safety analysis results
        """
        if not text or not text.strip():
            return {'overall_risk': 0.0, 'categories': {}, 'recommendations': ['Text is empty']}
        
        text_lower = text.lower()
        category_scores = {}
        
        # Pattern-based analysis
        for pattern_info in self.harmful_patterns:
            pattern = pattern_info['pattern']
            category = pattern_info['category']
            weight = pattern_info['weight']
            
            matches = re.findall(pattern, text_lower)
            if matches:
                score = min(1.0, len(matches) * weight * 0.1)  # Normalize by text length
                category_scores[category] = category_scores.get(category, 0) + score
        
        # Calculate overall risk
        overall_risk = min(1.0, sum(category_scores.values()))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(category_scores, overall_risk)
        
        return {
            'overall_risk': overall_risk,
            'categories': category_scores,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, category_scores: Dict[str, float], overall_risk: float) -> List[str]:
        """Generate safety recommendations based on analysis."""
        recommendations = []
        
        if overall_risk > 0.7:
            recommendations.append("HIGH RISK: Content contains potentially harmful material")
        elif overall_risk > 0.4:
            recommendations.append("MEDIUM RISK: Content may need review")
        else:
            recommendations.append("LOW RISK: Content appears safe")
        
        # Category-specific recommendations
        for category, score in category_scores.items():
            if score > 0.5:
                if category == 'violence':
                    recommendations.append("Contains violent content - consider alternative phrasing")
                elif category == 'hate':
                    recommendations.append("Contains potentially discriminatory language")
                elif category == 'illegal':
                    recommendations.append("References to illegal activities detected")
                elif category == 'manipulation':
                    recommendations.append("Contains manipulation tactics")
        
        return recommendations
    
    def should_block_content(self, text: str, threshold: float = 0.3) -> Tuple[bool, str]:
        """
        Determine if content should be blocked.
        
        Args:
            text: Input text
            threshold: Risk threshold for blocking
            
        Returns:
            Tuple of (should_block, reason)
        """
        analysis = self.analyze_text_safety(text)
        overall_risk = analysis['overall_risk']
        
        if overall_risk >= threshold:
            reason = f"High risk content detected (risk: {overall_risk:.3f})"
            return True, reason
        
        return False, "Content passed safety checks"
    
    def enhance_safety_training_data(self, adversarial_texts: List[str], clean_texts: List[str]) -> Tuple[List[str], List[str]]:
        """
        Enhance training data with safety-focused filtering and augmentation.
        
        Args:
            adversarial_texts: List of adversarial texts
            clean_texts: List of clean texts
            
        Returns:
            Enhanced training data
        """
        enhanced_adversarial = []
        enhanced_clean = []
        
        for adv_text, clean_text in zip(adversarial_texts, clean_texts):
            # Analyze both texts
            adv_analysis = self.analyze_text_safety(adv_text)
            clean_analysis = self.analyze_text_safety(clean_text)
            
            # Include most pairs for training, with minimal filtering
            # Focus on ensuring clean text is safer than adversarial
            if clean_analysis['overall_risk'] <= adv_analysis['overall_risk'] + 0.1:
                enhanced_adversarial.append(adv_text)
                enhanced_clean.append(clean_text)
            
            # Add safety-focused augmentations
            if adv_analysis['overall_risk'] > 0.5:
                # Create safer alternatives
                safer_alternatives = self._create_safer_alternatives(adv_text)
                for alternative in safer_alternatives:
                    enhanced_adversarial.append(adv_text)
                    enhanced_clean.append(alternative)
        
        logger.info(f"Enhanced training data: {len(enhanced_adversarial)} pairs (from {len(adversarial_texts)})")
        return enhanced_adversarial, enhanced_clean
    
    def _create_safer_alternatives(self, text: str) -> List[str]:
        """Create safer alternatives for harmful text."""
        alternatives = []
        
        # Replace harmful words with safer alternatives
        replacements = {
            'kill': 'stop',
            'destroy': 'modify',
            'hate': 'dislike',
            'stupid': 'unwise',
            'illegal': 'inappropriate',
            'manipulate': 'influence',
            'trick': 'guide'
        }
        
        safer_text = text
        for harmful, safer in replacements.items():
            safer_text = re.sub(rf'\b{harmful}\b', safer, safer_text, flags=re.IGNORECASE)
        
        if safer_text != text:
            alternatives.append(safer_text)
        
        # Add educational alternatives
        alternatives.append(f"Let me help you understand the topic of '{text}' in a safe and educational way.")
        
        return alternatives
    
    def calculate_semantic_safety_score(self, original_embedding: torch.Tensor, cleaned_embedding: torch.Tensor) -> Dict[str, float]:
        """
        Calculate semantic safety score between original and cleaned embeddings.
        
        Args:
            original_embedding: Original text embedding
            cleaned_embedding: Cleaned text embedding
            
        Returns:
            Dictionary with safety metrics
        """
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(original_embedding, cleaned_embedding, dim=1).item()
        
        # Calculate embedding distance
        distance = torch.norm(original_embedding - cleaned_embedding, dim=1).item()
        
        # Normalize distance (typical embedding norm is around 1.0)
        normalized_distance = min(1.0, distance / 2.0)
        
        # Safety score combines similarity and controlled change
        safety_score = similarity * (1.0 - normalized_distance * 0.5)
        
        return {
            'similarity': similarity,
            'distance': distance,
            'normalized_distance': normalized_distance,
            'safety_score': safety_score,
            'is_safe': safety_score > 0.6
        }


class AdaptiveSafetyThresholds:
    """Dynamic safety thresholds based on context and usage patterns."""
    
    def __init__(self):
        self.base_thresholds = {
            'conservative': 0.5,
            'moderate': 0.7,
            'permissive': 0.8
        }
        self.current_mode = 'moderate'
    
    def get_threshold(self, context: str = 'general') -> float:
        """Get appropriate threshold based on context."""
        base_threshold = self.base_thresholds[self.current_mode]
        
        # Adjust based on context
        if context == 'educational':
            return base_threshold * 0.8  # More permissive for education
        elif context == 'safety_critical':
            return base_threshold * 1.2  # More conservative for safety
        elif context == 'research':
            return base_threshold * 0.9  # Slightly more permissive
        
        return base_threshold
    
    def set_mode(self, mode: str):
        """Set safety mode."""
        if mode in self.base_thresholds:
            self.current_mode = mode
            logger.info(f"Safety mode set to: {mode}")
        else:
            logger.warning(f"Unknown safety mode: {mode}")


def create_safety_enhanced_training_data(adversarial_texts: List[str], clean_texts: List[str]) -> Tuple[List[str], List[str]]:
    """
    Create safety-enhanced training data with improved filtering.
    
    Args:
        adversarial_texts: List of adversarial texts
        clean_texts: List of clean texts
        
    Returns:
        Enhanced training data
    """
    safety_controller = SafetyController()
    return safety_controller.enhance_safety_training_data(adversarial_texts, clean_texts)
