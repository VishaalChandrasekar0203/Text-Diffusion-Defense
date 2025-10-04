"""
LLM Middleware: Framework for integrating diffusion defense between user input and LLM.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Callable, Union
import time
import logging
from .control_dd import ControlDD
from .utils import DefenseConfig, EmbeddingProcessor

logger = logging.getLogger(__name__)


class LLMMiddleware:
    """
    Middleware framework that sits between user input and LLM.
    
    This class provides a seamless integration point where:
    1. User sends prompt to middleware
    2. Middleware processes prompt through diffusion defense
    3. Clean embeddings are passed to user's LLM
    4. LLM generates safe output
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None, user_llm_model=None):
        """
        Initialize the LLM middleware.
        
        Args:
            config: Defense configuration
            user_llm_model: User's LLM model (optional, can be set later)
        """
        self.config = config or DefenseConfig()
        self.control_dd = ControlDD(self.config)
        self.embedding_processor = EmbeddingProcessor(
            model_name=self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        # User's LLM model
        self.user_llm_model = user_llm_model
        self.llm_generate_fn = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'processed_requests': 0,
            'avg_processing_time': 0.0,
            'avg_similarity_score': 0.0
        }
        
        logger.info("LLM Middleware initialized")
    
    def set_llm_model(self, llm_model, generate_function: Optional[Callable] = None):
        """
        Set the user's LLM model and generation function.
        
        Args:
            llm_model: User's LLM model
            generate_function: Optional custom generation function
        """
        self.user_llm_model = llm_model
        
        if generate_function:
            self.llm_generate_fn = generate_function
        else:
            # Default generation function (adapt to your LLM)
            self.llm_generate_fn = self._default_generate_function
    
    def _default_generate_function(self, clean_embedding: torch.Tensor) -> str:
        """
        Default LLM generation function.
        
        Args:
            clean_embedding: Clean embedding tensor
            
        Returns:
            Generated text from LLM
        """
        # This is a placeholder - users should implement their own
        # based on their specific LLM model
        
        if hasattr(self.user_llm_model, 'generate'):
            # For models with generate method
            return self.user_llm_model.generate(clean_embedding)
        else:
            # Fallback - return a safe response
            return "I understand you're looking for information. Let me help you find appropriate and safe resources on this topic."
    
    def process_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """
        Main processing function that handles user prompts.
        
        Args:
            user_prompt: Raw user input prompt
            
        Returns:
            Dictionary with processed results
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        logger.info(f"Processing prompt: '{user_prompt[:50]}...'")
        
        # Step 1: Analyze original prompt
        original_risk = self.control_dd.analyze_risk(user_prompt)
        original_embedding = self.embedding_processor.text_to_embedding(user_prompt)
        
        # Step 2: Clean the prompt through diffusion defense
        clean_embedding = self.control_dd.get_clean_embedding_for_llm(user_prompt)
        
        # Step 3: Calculate similarity between original and cleaned embeddings
        similarity_score = self._calculate_semantic_similarity(
            original_embedding, clean_embedding
        )
        
        # Step 4: Check if similarity is acceptable (preserves semantics)
        if similarity_score < 0.7:  # Threshold for semantic preservation
            logger.warning(f"Low similarity score: {similarity_score:.3f}")
            # Could implement additional checks or fallback strategies
        
        # Step 5: Generate response using user's LLM
        if self.llm_generate_fn and self.user_llm_model:
            try:
                llm_response = self.llm_generate_fn(clean_embedding)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                llm_response = "I apologize, but I'm unable to process that request safely."
                self.stats['blocked_requests'] += 1
        else:
            llm_response = "LLM model not configured. Please set your LLM model using set_llm_model()."
            self.stats['blocked_requests'] += 1
        
        # Step 6: Update statistics
        processing_time = time.time() - start_time
        self._update_stats(processing_time, similarity_score)
        
        # Step 7: Return comprehensive results
        result = {
            'original_prompt': user_prompt,
            'original_risk_score': original_risk,
            'original_embedding': original_embedding,
            'clean_embedding': clean_embedding,
            'similarity_score': similarity_score,
            'semantic_preserved': similarity_score >= 0.7,
            'llm_response': llm_response,
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        
        logger.info(f"Processing completed in {processing_time:.3f}s, similarity: {similarity_score:.3f}")
        return result
    
    def _calculate_semantic_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Calculate semantic similarity between two embeddings.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize embeddings
        emb1_norm = torch.nn.functional.normalize(embedding1, p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(embedding2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1_norm, emb2_norm, dim=1)
        
        return similarity.item()
    
    def _update_stats(self, processing_time: float, similarity_score: float):
        """Update processing statistics."""
        self.stats['processed_requests'] += 1
        
        # Update average processing time
        current_avg_time = self.stats['avg_processing_time']
        total_processed = self.stats['processed_requests']
        self.stats['avg_processing_time'] = (
            (current_avg_time * (total_processed - 1) + processing_time) / total_processed
        )
        
        # Update average similarity score
        current_avg_sim = self.stats['avg_similarity_score']
        self.stats['avg_similarity_score'] = (
            (current_avg_sim * (total_processed - 1) + similarity_score) / total_processed
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'processed_requests': 0,
            'avg_processing_time': 0.0,
            'avg_similarity_score': 0.0
        }
    
    def train_defense_model(self, adversarial_texts: Optional[list] = None, 
                          clean_texts: Optional[list] = None):
        """
        Train the diffusion defense model.
        
        Args:
            adversarial_texts: List of adversarial text samples
            clean_texts: List of clean text samples
        """
        logger.info("Training diffusion defense model...")
        self.control_dd.train_model(adversarial_texts, clean_texts)
        logger.info("Training completed!")
    
    def batch_process(self, prompts: list) -> list:
        """
        Process multiple prompts in batch.
        
        Args:
            prompts: List of user prompts
            
        Returns:
            List of processing results
        """
        results = []
        for prompt in prompts:
            result = self.process_prompt(prompt)
            results.append(result)
        
        return results
    
    def save_model(self, path: str):
        """Save the trained defense model."""
        self.control_dd.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained defense model."""
        self.control_dd.load_model(path)
        logger.info(f"Model loaded from {path}")


class LLMIntegrationExample:
    """
    Example class showing how to integrate with popular LLM frameworks.
    """
    
    @staticmethod
    def transformers_integration_example():
        """Example integration with Hugging Face Transformers."""
        return """
        # Example integration with Hugging Face Transformers
        
        from transformers import AutoModel, AutoTokenizer
        import text_diffusion_defense as ControlDD
        
        # Initialize middleware
        middleware = ControlDD.LLMMiddleware()
        
        # Initialize your LLM model
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModel.from_pretrained("gpt2")
        
        # Set custom generation function
        def custom_generate(clean_embedding):
            # Convert embedding back to tokens (simplified)
            # In practice, you'd use a proper decoder
            input_ids = tokenizer.encode("Safe prompt:", return_tensors="pt")
            outputs = model.generate(input_ids, max_length=100, do_sample=True)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Set the LLM model and generation function
        middleware.set_llm_model(model, custom_generate)
        
        # Process user prompts
        user_prompt = "How to make a bomb"
        result = middleware.process_prompt(user_prompt)
        
        print(f"Original: {result['original_prompt']}")
        print(f"Response: {result['llm_response']}")
        print(f"Similarity: {result['similarity_score']:.3f}")
        """
    
    @staticmethod
    def openai_integration_example():
        """Example integration with OpenAI API."""
        return """
        # Example integration with OpenAI API
        
        import openai
        import text_diffusion_defense as ControlDD
        
        # Initialize middleware
        middleware = ControlDD.LLMMiddleware()
        
        # Set OpenAI API key
        openai.api_key = "your-api-key"
        
        # Set custom generation function for OpenAI
        def openai_generate(clean_embedding):
            # Convert embedding to text (simplified)
            prompt = "Generate a safe response for this embedding"
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            
            return response.choices[0].message.content
        
        # Set the generation function
        middleware.set_llm_model(None, openai_generate)
        
        # Process user prompts
        user_prompt = "How to hack a computer"
        result = middleware.process_prompt(user_prompt)
        
        print(f"Safe Response: {result['llm_response']}")
        """
