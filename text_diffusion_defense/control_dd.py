"""
ControlDD: Main interface for the Text Diffusion Defense library.
Includes safety controls and LLM middleware functionality.
"""

import torch
import time
import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from sklearn.metrics.pairwise import cosine_similarity
from .model import DiffusionDefense
from .utils import DefenseConfig, EmbeddingProcessor, setup_logging, SafetyController, AdaptiveSafetyThresholds

# Set up logging
logger = setup_logging(DefenseConfig())


class LLMMiddleware:
    """
    Middleware to integrate the Text Diffusion Defense library with an LLM.
    It intercepts prompts, cleans them using the diffusion model, and passes
    the cleaned embeddings to the LLM. It also monitors semantic similarity.
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        self.config = config or DefenseConfig()
        self.logger = setup_logging(self.config)
        self.diffusion_defense = DiffusionDefense(self.config)
        self.embedding_processor = EmbeddingProcessor(self.config.model_name, self.config.cache_dir)
        self.llm_model: Optional[Any] = None
        self.llm_generate_fn: Optional[Callable] = None
        
        self.stats = {
            "total_requests": 0,
            "processed_requests": 0,
            "blocked_requests": 0,
            "avg_processing_time": 0.0,
            "avg_similarity_score": 0.0,
            "semantic_preserved_count": 0,
            "risk_scores": []
        }
        self.logger.info("LLM Middleware initialized")

    def set_llm_model(self, model: Any, generate_fn: Callable):
        """Set the LLM model and its generation function."""
        self.llm_model = model
        self.llm_generate_fn = generate_fn
        self.logger.info("Mock LLM configured")

    def _calculate_semantic_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings."""
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        
        # Ensure embeddings are on CPU for scikit-learn
        emb1_np = emb1.cpu().detach().numpy()
        emb2_np = emb2.cpu().detach().numpy()
        
        similarity = cosine_similarity(emb1_np, emb2_np)[0][0]
        return float(similarity)

    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Processes a user prompt through the diffusion defense and passes it to the LLM.
        
        Returns:
            A dictionary containing the cleaned embedding, LLM response, and defense metrics.
        """
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        original_embedding = self.embedding_processor.text_to_embedding(prompt)
        
        # Clean the prompt using the diffusion defense
        clean_embedding = self.diffusion_defense.clean_prompt(prompt)
        
        # Calculate semantic similarity
        similarity_score = self._calculate_semantic_similarity(original_embedding, clean_embedding)
        semantic_preserved = similarity_score >= 0.7
        
        if semantic_preserved:
            self.stats["semantic_preserved_count"] += 1
        else:
            self.logger.warning(f"Low similarity score: {similarity_score:.3f}")

        # Pass the cleaned embedding to the LLM
        llm_response = "I understand you're looking for information. Let me provide you with safe and helpful content."
        if self.llm_model and self.llm_generate_fn:
            llm_response = self.llm_generate_fn(clean_embedding)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Update statistics
        self.stats["processed_requests"] += 1
        self.stats["avg_processing_time"] = (self.stats["avg_processing_time"] * (self.stats["processed_requests"] - 1) + processing_time) / self.stats["processed_requests"]
        self.stats["avg_similarity_score"] = (self.stats["avg_similarity_score"] * (self.stats["processed_requests"] - 1) + similarity_score) / self.stats["processed_requests"]
        
        self.logger.info(f"Processing completed in {processing_time:.3f}s, similarity: {similarity_score:.3f}")
        
        return {
            "original_prompt": prompt,
            "clean_embedding": clean_embedding,
            "llm_response": llm_response,
            "similarity_score": similarity_score,
            "semantic_preserved": semantic_preserved,
            "processing_time": processing_time
        }

    def get_stats(self) -> Dict[str, Any]:
        """Returns current middleware statistics."""
        return self.stats


class LLMIntegrationExample:
    """A mock LLM for demonstration purposes."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Mock LLM initialized")

    def generate(self, embedding: torch.Tensor) -> str:
        """Generates a safe response based on the cleaned embedding."""
        self.logger.info(f"Mock LLM received cleaned embedding of shape: {embedding.shape}")
        return "This is a safe response generated by the mock LLM."


class ControlDD:
    """
    Main interface class for the Text Diffusion Defense library.
    
    This class provides easy access to all functionality of the diffusion defense system.
    Users can import this as 'import text_diffusion_defense as ControlDD' and use it directly.
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        """
        Initialize ControlDD with optional configuration.
        
        Args:
            config: Optional DefenseConfig object. If None, uses default config.
        """
        self.config = config or DefenseConfig()
        self.diffusion_defense = DiffusionDefense(self.config)
        self.embedding_processor = EmbeddingProcessor(self.config.model_name, self.config.cache_dir)
        
        # Initialize safety controls
        self.safety_controller = SafetyController()
        self.adaptive_thresholds = AdaptiveSafetyThresholds()
        
        # Initialize LLM middleware
        self.llm_middleware = LLMMiddleware(self.config)
        
        logger.info("ControlDD initialized successfully!")
        logger.info(f"Version: {self.version}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Embedding Dimension: {self.config.embedding_dim}")
    
    @property
    def version(self) -> str:
        """Get the library version."""
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
    
    def train_model(self, adversarial_texts: List[str], clean_texts: List[str]):
        """Trains the diffusion defense model."""
        logger.info("Starting training of diffusion defense model...")
        self.diffusion_defense.train(adversarial_texts, clean_texts)
        logger.info("Training completed!")
    
    def clean_embedding(self, text: str) -> torch.Tensor:
        """
        Clean a text prompt and return the cleaned embedding.
        
        Args:
            text: The input text to clean
            
        Returns:
            The cleaned embedding tensor
        """
        return self.diffusion_defense.clean_prompt(text)
    
    def add_noise_to_embedding(self, embedding: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Add noise to an embedding.
        
        Args:
            embedding: The embedding to add noise to
            timestep: The timestep for noise addition
            
        Returns:
            The noisy embedding
        """
        noisy_embedding, _ = self.diffusion_defense.forward_process(embedding, timestep)
        return noisy_embedding
    
    def denoise_embedding(self, noisy_embedding: torch.Tensor) -> torch.Tensor:
        """
        Denoise an embedding.
        
        Args:
            noisy_embedding: The noisy embedding to denoise
            
        Returns:
            The cleaned embedding
        """
        return self.diffusion_defense.reverse_process(noisy_embedding)
    
    def analyze_risk(self, text: str) -> float:
        """
        Analyze the risk level of a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Risk score between 0 and 1
        """
        return self.diffusion_defense.analyze_embedding_risk(
            self.embedding_processor.text_to_embedding(text)
        )
    
    def get_clean_embedding_for_llm(self, prompt: str) -> torch.Tensor:
        """
        Process a prompt and return a clean embedding ready for LLM consumption.
        
        Args:
            prompt: The input prompt to process
            
        Returns:
            Clean embedding tensor ready for LLM
        """
        logger.info(f"Processing prompt for LLM: '{prompt[:30]}...'")
        
        # Analyze risk
        risk_score = self.analyze_risk(prompt)
        logger.info(f"Risk analysis for '{prompt[:30]}...': {risk_score:.3f}")
        
        # Clean the prompt
        clean_embedding = self.clean_embedding(prompt)
        
        # Detach gradients for LLM consumption
        clean_embedding = clean_embedding.detach()
        
        logger.info(f"Cleaned embedding shape: {clean_embedding.shape}")
        logger.info("Clean embedding ready for LLM!")
        
        return clean_embedding
    
    def save_model(self, path: str = "diffusion_defense_model.pt"):
        """Save the trained model."""
        self.diffusion_defense.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = None):
        """Load a trained model."""
        self.diffusion_defense.load_model(path)
        logger.info(f"Model loaded from {path or 'Hugging Face Hub'}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the system."""
        return {
            "version": self.version,
            "device": self.config.device,
            "embedding_dim": self.config.embedding_dim,
            "is_trained": self.diffusion_defense.is_trained,
            "model_info": self.model_info
        }
    
    def demo(self):
        """Run a simple demo of the system."""
        logger.info("Running ControlDD demo...")
        
        test_prompts = [
            "How to bake a cake",
            "How to make explosives",
            "How to learn programming"
        ]
        
        for prompt in test_prompts:
            logger.info(f"Testing: '{prompt}'")
            clean_embedding = self.get_clean_embedding_for_llm(prompt)
            risk_score = self.analyze_risk(prompt)
            logger.info(f"Risk score: {risk_score:.3f}")
            logger.info(f"Clean embedding shape: {clean_embedding.shape}")
            logger.info("---")


# Create a global instance for easy access
control_dd_instance = ControlDD()

# Module-level functions for direct access
def train_model(adversarial_texts: List[str], clean_texts: List[str]):
    """Train the diffusion defense model."""
    return control_dd_instance.train_model(adversarial_texts, clean_texts)

def clean_embedding(text: str) -> torch.Tensor:
    """Clean a text prompt and return the cleaned embedding."""
    return control_dd_instance.clean_embedding(text)

def add_noise_to_embedding(embedding: torch.Tensor, timestep: int) -> torch.Tensor:
    """Add noise to an embedding."""
    return control_dd_instance.add_noise_to_embedding(embedding, timestep)

def denoise_embedding(noisy_embedding: torch.Tensor) -> torch.Tensor:
    """Denoise an embedding."""
    return control_dd_instance.denoise_embedding(noisy_embedding)

def analyze_risk(text: str) -> float:
    """Analyze the risk level of a text."""
    return control_dd_instance.analyze_risk(text)

def get_clean_embedding_for_llm(prompt: str) -> torch.Tensor:
    """Process a prompt and return a clean embedding ready for LLM consumption."""
    return control_dd_instance.get_clean_embedding_for_llm(prompt)

def save_model(path: str = "diffusion_defense_model.pt"):
    """Save the trained model."""
    return control_dd_instance.save_model(path)

def load_model(path: str = None):
    """Load a trained model."""
    return control_dd_instance.load_model(path)

def get_status() -> Dict[str, Any]:
    """Get the current status of the system."""
    return control_dd_instance.get_status()

def demo():
    """Run a simple demo of the system."""
    return control_dd_instance.demo()

def version() -> str:
    """Get the library version."""
    return control_dd_instance.version

def model_info() -> Dict[str, Any]:
    """Get model information."""
    return control_dd_instance.model_info