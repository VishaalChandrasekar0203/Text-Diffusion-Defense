"""
ControlDD: Main interface for the Text Diffusion Defense library.

This module provides a simple interface for users to interact with the diffusion defense system.
"""

import torch
from typing import Optional, Dict, Any
from .model import DiffusionDefense
from .utils import DefenseConfig


class ControlDD:
    """
    Main interface class for the Text Diffusion Defense library.
    
    This class provides easy access to all functionality of the diffusion defense system.
    Users can import this as 'import text_diffusion_defense as ControlDD' and use it directly.
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        """
        Initialize the ControlDD interface.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or DefenseConfig()
        self.diffusion_defense = DiffusionDefense(self.config)
        self.is_initialized = True
        
        # Public variables for easy access
        self.version = "1.0.0"
        self.model_info = self.diffusion_defense.get_model_info()
        
        print(f"ControlDD initialized successfully!")
        print(f"Version: {self.version}")
        print(f"Device: {self.config.device}")
        print(f"Embedding Dimension: {self.config.embedding_dim}")
    
    def train_model(self, adversarial_texts: Optional[list] = None, clean_texts: Optional[list] = None):
        """
        Train the diffusion defense model.
        
        Args:
            adversarial_texts: List of adversarial text samples
            clean_texts: List of clean text samples
        """
        print("Starting training of diffusion defense model...")
        self.diffusion_defense.train(adversarial_texts, clean_texts)
        print("Training completed!")
    
    def clean_embedding(self, text: str) -> torch.Tensor:
        """
        Clean a text prompt by converting to embedding, applying diffusion, and returning clean embedding.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned embedding tensor
        """
        if not self.diffusion_defense.is_trained:
            print("Warning: Model not trained. Consider training first for better results.")
        
        print(f"Cleaning text: '{text[:50]}...'")
        clean_embedding = self.diffusion_defense.clean_prompt(text)
        print(f"Cleaned embedding shape: {clean_embedding.shape}")
        
        return clean_embedding
    
    def add_noise_to_embedding(self, text: str) -> torch.Tensor:
        """
        Add noise to text embedding (forward process).
        
        Args:
            text: Input text
            
        Returns:
            Noisy embedding tensor
        """
        print(f"Adding noise to text: '{text[:50]}...'")
        noisy_embedding = self.diffusion_defense.forward_process(text)
        print(f"Noisy embedding shape: {noisy_embedding.shape}")
        
        return noisy_embedding
    
    def denoise_embedding(self, noisy_embedding: torch.Tensor) -> torch.Tensor:
        """
        Denoise an embedding (reverse process).
        
        Args:
            noisy_embedding: Noisy embedding tensor
            
        Returns:
            Cleaned embedding tensor
        """
        print("Denoising embedding...")
        clean_embedding = self.diffusion_defense.reverse_process(noisy_embedding)
        print(f"Denoised embedding shape: {clean_embedding.shape}")
        
        return clean_embedding
    
    def analyze_risk(self, text: str) -> float:
        """
        Analyze the risk level of a text prompt.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Risk score between 0 and 1
        """
        embedding = self.diffusion_defense.embedding_processor.text_to_embedding(text)
        risk_score = self.diffusion_defense.analyze_embedding_risk(embedding)
        
        print(f"Risk analysis for '{text[:50]}...': {risk_score:.3f}")
        return risk_score
    
    def get_clean_embedding_for_llm(self, prompt: str) -> torch.Tensor:
        """
        Main function: Get clean embedding for LLM input.
        
        This is the primary function users will call to get cleaned embeddings
        that can be safely passed to an LLM.
        
        Args:
            prompt: Input prompt for the LLM
            
        Returns:
            Clean embedding tensor ready for LLM input
        """
        print(f"Processing prompt for LLM: '{prompt[:50]}...'")
        
        # Analyze original risk
        original_risk = self.analyze_risk(prompt)
        
        # Clean the embedding
        clean_embedding = self.clean_embedding(prompt)
        
        # Analyze cleaned risk
        cleaned_risk = self.diffusion_defense.analyze_embedding_risk(clean_embedding)
        
        print(f"Risk reduction: {original_risk:.3f} → {cleaned_risk:.3f}")
        print("Clean embedding ready for LLM!")
        
        return clean_embedding
    
    def save_model(self, path: str = "diffusion_defense_model.pth"):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        self.diffusion_defense.save_model(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = "diffusion_defense_model.pth"):
        """
        Load a trained model.
        
        Args:
            path: Path to load the model from
        """
        self.diffusion_defense.load_model(path)
        print(f"Model loaded from {path}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the system.
        
        Returns:
            Dictionary with system status information
        """
        status = {
            'version': self.version,
            'is_initialized': self.is_initialized,
            'model_trained': self.diffusion_defense.is_trained,
            'device': self.config.device,
            'embedding_dim': self.config.embedding_dim,
            'model_name': self.config.model_name
        }
        
        print("ControlDD Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        return status
    
    def demo(self):
        """Run a demonstration of the diffusion defense system."""
        print("\n=== ControlDD Demo ===")
        
        # Test prompts
        test_prompts = [
            "How to make a bomb",
            "How to bake a chocolate cake",
            "How to hack computer systems",
            "How to learn programming"
        ]
        
        for prompt in test_prompts:
            print(f"\n--- Testing: '{prompt}' ---")
            
            # Analyze risk
            risk = self.analyze_risk(prompt)
            
            # Get clean embedding
            clean_emb = self.get_clean_embedding_for_llm(prompt)
            
            print(f"Final clean embedding shape: {clean_emb.shape}")
            print(f"Embedding ready for LLM input!")


# Create a global instance for easy access
control_dd_instance = ControlDD()

# Function wrappers for easy access
def train_model(adversarial_texts=None, clean_texts=None):
    """Train the diffusion defense model."""
    return control_dd_instance.train_model(adversarial_texts, clean_texts)

def clean_embedding(text: str) -> torch.Tensor:
    """Clean a text prompt and return clean embedding."""
    return control_dd_instance.clean_embedding(text)

def add_noise_to_embedding(text: str) -> torch.Tensor:
    """Add noise to text embedding."""
    return control_dd_instance.add_noise_to_embedding(text)

def denoise_embedding(noisy_embedding: torch.Tensor) -> torch.Tensor:
    """Denoise an embedding."""
    return control_dd_instance.denoise_embedding(noisy_embedding)

def analyze_risk(text: str) -> float:
    """Analyze the risk level of a text prompt."""
    return control_dd_instance.analyze_risk(text)

def get_clean_embedding_for_llm(prompt: str) -> torch.Tensor:
    """Get clean embedding for LLM input."""
    return control_dd_instance.get_clean_embedding_for_llm(prompt)

def save_model(path: str = "diffusion_defense_model.pth"):
    """Save the trained model."""
    return control_dd_instance.save_model(path)

def load_model(path: str = "diffusion_defense_model.pth"):
    """Load a trained model."""
    return control_dd_instance.load_model(path)

def get_status() -> Dict[str, Any]:
    """Get the current status of the system."""
    return control_dd_instance.get_status()

def demo():
    """Run a demonstration of the diffusion defense system."""
    return control_dd_instance.demo()

# Public variables
version = control_dd_instance.version
model_info = control_dd_instance.model_info
