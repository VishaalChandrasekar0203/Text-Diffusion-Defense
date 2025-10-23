"""
ControlDD: Main interface for the Text Diffusion Defense library.
Simple, clean, production-ready.
"""

import torch
import logging
import re
from typing import List, Dict, Any, Optional
from .model import DiffusionDefense
from .utils import DefenseConfig, EmbeddingProcessor, setup_logging, AdaptiveSafetyThresholds
from .enhanced_safety import EnhancedSafetyController

# Set up logging
logger = setup_logging(DefenseConfig())


class ControlDD:
    """
    Main interface class for Text Diffusion Defense.
    
    Pre-trained and ready to use - just initialize and call methods.
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        """
        Initialize ControlDD with pre-trained model.
        
        Args:
            config: Optional DefenseConfig. Uses defaults if None.
        """
        self.config = config or DefenseConfig()
        self.diffusion_defense = DiffusionDefense(self.config)
        self.embedding_processor = EmbeddingProcessor(self.config.model_name, self.config.cache_dir)
        self.safety_controller = EnhancedSafetyController(self.embedding_processor)
        self.adaptive_thresholds = AdaptiveSafetyThresholds()
        
        logger.info("ControlDD initialized successfully!")
        logger.info(f"Version: {self.version}")
        logger.info(f"Device: {self.config.device}")
    
    @property
    def version(self) -> str:
        """Get library version."""
        return "1.0.0"
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "version": self.version,
            "embedding_dim": self.config.embedding_dim,
            "device": self.config.device,
            "is_trained": self.diffusion_defense.is_trained
        }
    
    def analyze_risk(self, text: str) -> float:
        """
        Analyze risk level of text.
        
        Returns:
            Risk score between 0 and 1
        """
        return self.diffusion_defense.analyze_embedding_risk(
            self.embedding_processor.text_to_embedding(text)
        )
    
    def get_clean_text_for_llm(self, prompt: str) -> str:
        """
        SIMPLE USAGE: Clean prompt and return safe text for LLM.
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Clean text ready for LLM
        """
        logger.info(f"Processing: '{prompt[:30]}...'")
        clean_text = self.diffusion_defense.clean_prompt_to_text(prompt)
        logger.info(f"Cleaned: '{clean_text[:50]}...'")
        return clean_text
    
    def get_clean_embedding_for_llm(self, prompt: str) -> torch.Tensor:
        """
        Clean prompt and return embedding for LLM.
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Clean embedding tensor
        """
        logger.info(f"Processing: '{prompt[:30]}...'")
        risk_score = self.analyze_risk(prompt)
        logger.info(f"Risk: {risk_score:.3f}")
        
        clean_embedding = self.diffusion_defense.clean_prompt(prompt).detach()
        logger.info(f"Embedding shape: {clean_embedding.shape}")
        return clean_embedding
    
    def analyze_and_respond(self, prompt: str) -> Dict[str, Any]:
        """
        TRANSPARENT WORKFLOW: Analyze prompt and provide feedback to user.
        
        Risk Levels:
        - <0.05: Pass through to LLM
        - 0.05-0.3: Suggest cleaned version
        - >0.3: Reject
        
        Returns:
            {
                'status': 'approved' | 'needs_clarification' | 'rejected',
                'original_prompt': str,
                'cleaned_prompt': str | None,
                'risk_score': float,
                'risk_categories': List[str],
                'problematic_words': List[str],
                'message_to_user': str | None,
                'send_to_llm': bool,
                'llm_prompt': str | None
            }
        """
        logger.info(f"Analyzing: '{prompt[:50]}...'")
        
        # Analyze safety
        safety_analysis = self.safety_controller.analyze_text_safety(prompt)
        risk_score = safety_analysis['overall_risk']
        risk_categories = list(safety_analysis['categories'].keys())
        problematic_words = self._identify_problematic_words(prompt, safety_analysis)
        
        logger.info(f"Risk: {risk_score:.3f}, Categories: {risk_categories}")
        
        result = {
            'original_prompt': prompt,
            'risk_score': risk_score,
            'risk_categories': risk_categories,
            'problematic_words': problematic_words
        }
        
        # LOW RISK: Pass through
        if risk_score < 0.05 and not risk_categories:
            logger.info("✅ Low risk - pass through")
            result.update({
                'status': 'approved',
                'cleaned_prompt': None,
                'message_to_user': None,
                'send_to_llm': True,
                'llm_prompt': prompt
            })
            return result
        
        # HIGH RISK: Reject
        if risk_score > 0.3:
            logger.warning("🚫 High risk - reject")
            result.update({
                'status': 'rejected',
                'cleaned_prompt': None,
                'message_to_user': "I'm sorry, but I cannot assist with this request as it may involve harmful or inappropriate content. Please feel free to ask me something else!",
                'send_to_llm': False,
                'llm_prompt': None
            })
            return result
        
        # MEDIUM RISK: Suggest cleaned version
        logger.info("⚠️  Medium risk - suggesting alternative")
        
        clean_text = self.diffusion_defense.clean_prompt_to_text(prompt)
        explanation = self._create_problem_explanation(risk_categories, problematic_words)
        
        message_to_user = (
            f"I noticed your prompt might contain some concerning content ({explanation}). "
            f"Did you mean: \"{clean_text}\"?\n\n"
            f"If this is what you intended, I can help with that. Otherwise, please rephrase your question."
        )
        
        logger.info(f"Suggesting: '{clean_text[:50]}...'")
        
        result.update({
            'status': 'needs_clarification',
            'cleaned_prompt': clean_text,
            'message_to_user': message_to_user,
            'send_to_llm': False,
            'llm_prompt': None
        })
        
        return result
    
    def verify_and_proceed(self, user_choice: str, original_prompt: str, cleaned_prompt: str) -> Dict[str, Any]:
        """
        VERIFICATION: Handle user's choice with double safety check.
        
        - If user chooses 'original' → Block immediately
        - If user chooses 'cleaned' → Re-verify safety, then approve if clean
        
        Returns:
            {
                'status': 'approved' | 'rejected' | 'needs_further_clarification',
                'prompt_to_use': str | None,
                'message_to_user': str | None,
                'send_to_llm': bool,
                'risk_score': float | None
            }
        """
        logger.info(f"Verifying user choice: {user_choice}")
        
        result = {
            'user_choice': user_choice,
            'original_prompt': original_prompt,
            'cleaned_prompt': cleaned_prompt
        }
        
        # User insists on bad prompt → BLOCK
        if user_choice == 'original':
            logger.warning("🚫 User insisted on bad prompt - BLOCKING")
            result.update({
                'status': 'rejected',
                'prompt_to_use': None,
                'message_to_user': "I'm sorry, but I cannot assist with the original request as it contains concerning content. Please feel free to ask me something else!",
                'send_to_llm': False,
                'risk_score': None
            })
            return result
        
        # User accepts cleaned → RE-VERIFY
        if user_choice == 'cleaned':
            logger.info("Re-verifying cleaned prompt")
            
            re_verification = self.analyze_and_respond(cleaned_prompt)
            logger.info(f"Re-verification: {re_verification['status']}, risk: {re_verification['risk_score']:.3f}")
            
            if re_verification['status'] == 'approved':
                logger.info("✅ Re-verification passed - APPROVED")
                result.update({
                    'status': 'approved',
                    'prompt_to_use': cleaned_prompt,
                    'message_to_user': None,
                    'send_to_llm': True,
                    'risk_score': re_verification['risk_score']
                })
                return result
            
            if re_verification['status'] == 'needs_clarification':
                logger.warning("⚠️  Cleaned prompt still has issues")
                result.update({
                    'status': 'needs_further_clarification',
                    'prompt_to_use': None,
                    'message_to_user': f"I notice the revised prompt still contains some concerning content. {re_verification['message_to_user']}",
                    'send_to_llm': False,
                    'risk_score': re_verification['risk_score'],
                    'further_cleaned_suggestion': re_verification.get('cleaned_prompt')
                })
                return result
            
            # Re-verification rejected
            logger.warning("🚫 Re-verification REJECTED")
            result.update({
                'status': 'rejected',
                'prompt_to_use': None,
                'message_to_user': "I'm sorry, but even the revised prompt contains concerning content. Please try rephrasing in a different way.",
                'send_to_llm': False,
                'risk_score': re_verification['risk_score']
            })
            return result
        
        # Invalid choice
        logger.error(f"Invalid choice: {user_choice}")
        result.update({
            'status': 'error',
            'prompt_to_use': None,
            'message_to_user': "Invalid choice. Please select 'original' or 'cleaned'.",
            'send_to_llm': False,
            'risk_score': None
        })
        return result
    
    def _identify_problematic_words(self, text: str, safety_analysis: Dict) -> List[str]:
        """Identify problematic words that triggered detection."""
        problematic = []
        text_lower = text.lower()
        harmful_patterns = self.safety_controller.harmful_patterns
        
        for pattern_info in harmful_patterns:
            matches = re.findall(pattern_info['pattern'], text_lower, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(match)
                    if match and match not in problematic:
                        problematic.append(match)
        
        return problematic
    
    def _create_problem_explanation(self, categories: List[str], problematic_words: List[str]) -> str:
        """Create user-friendly explanation of issues."""
        if not categories:
            return "potentially concerning language"
        
        category_map = {
            'violence': 'violent language',
            'illegal': 'references to illegal activities',
            'manipulation': 'manipulative language',
            'hate': 'potentially offensive language',
            'self-harm': 'concerning content',
            'terrorism': 'concerning content'
        }
        
        descriptions = [category_map.get(cat, cat) for cat in categories[:2]]
        return descriptions[0] if len(descriptions) == 1 else ' and '.join(descriptions)
    
    def save_model(self, path: str = "diffusion_defense_model.pt"):
        """Save the model (admin use only)."""
        self.diffusion_defense.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = None):
        """Load a model."""
        self.diffusion_defense.load_model(path)
        logger.info(f"Model loaded")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "version": self.version,
            "device": self.config.device,
            "embedding_dim": self.config.embedding_dim,
            "is_trained": self.diffusion_defense.is_trained
        }


# Global instance for easy access
control_dd_instance = ControlDD()

# Module-level functions
def analyze_risk(text: str) -> float:
    """Analyze risk level of text."""
    return control_dd_instance.analyze_risk(text)

def get_clean_text_for_llm(prompt: str) -> str:
    """Clean prompt and return safe text for LLM."""
    return control_dd_instance.get_clean_text_for_llm(prompt)

def get_clean_embedding_for_llm(prompt: str) -> torch.Tensor:
    """Clean prompt and return safe embedding for LLM."""
    return control_dd_instance.get_clean_embedding_for_llm(prompt)

def analyze_and_respond(prompt: str) -> Dict[str, Any]:
    """Analyze prompt with transparent feedback."""
    return control_dd_instance.analyze_and_respond(prompt)

def verify_and_proceed(user_choice: str, original_prompt: str, cleaned_prompt: str) -> Dict[str, Any]:
    """Verify user's choice with double safety check."""
    return control_dd_instance.verify_and_proceed(user_choice, original_prompt, cleaned_prompt)

def save_model(path: str = "diffusion_defense_model.pt"):
    """Save model."""
    return control_dd_instance.save_model(path)

def load_model(path: str = None):
    """Load model."""
    return control_dd_instance.load_model(path)

def get_status() -> Dict[str, Any]:
    """Get system status."""
    return control_dd_instance.get_status()

def version() -> str:
    """Get library version."""
    return control_dd_instance.version

def model_info() -> Dict[str, Any]:
    """Get model info."""
    return control_dd_instance.model_info

def demo():
    """Run simple demo."""
    logger.info("Running demo...")
    
    test_prompts = [
        "How to bake a cake",
        "How to make explosives",
        "How to learn programming"
    ]
    
    for prompt in test_prompts:
        logger.info(f"Testing: '{prompt}'")
        risk = control_dd_instance.analyze_risk(prompt)
        logger.info(f"Risk: {risk:.3f}")
        logger.info("---")
