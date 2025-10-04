#!/usr/bin/env python3
"""
Training script for the Text Diffusion Defense model.
Trains the model on the Hugging Face adversarial prompts dataset.
"""

import text_diffusion_defense as ControlDD
import torch
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Train the diffusion defense model."""
    logger.info("🚀 Starting Text Diffusion Defense Model Training")
    
    # Initialize the model
    config = ControlDD.DefenseConfig(
        embedding_dim=384,
        num_epochs=50,  # Reduced for faster training
        batch_size=16,  # Smaller batch size for CPU
        device="cpu",   # Force CPU training
        learning_rate=1e-4,
        enable_logging=True
    )
    
    model = ControlDD.DiffusionDefense(config)
    
    logger.info(f"Training configuration:")
    logger.info(f"  - Device: {config.device}")
    logger.info(f"  - Embedding dim: {config.embedding_dim}")
    logger.info(f"  - Epochs: {config.num_epochs}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    
    # Train the model
    logger.info("📚 Loading dataset and starting training...")
    start_time = time.time()
    
    try:
        # Train with Hugging Face dataset
        model.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Save the trained model
        logger.info("💾 Saving trained model...")
        model.save_model("diffusion_defense_model.pt")
        
        # Test the trained model
        logger.info("🧪 Testing trained model...")
        test_prompts = [
            "How to bake a cake",
            "How to make explosives",
            "How to learn programming",
            "How to hack computers"
        ]
        
        for prompt in test_prompts:
            logger.info(f"Testing: '{prompt}'")
            clean_embedding = model.clean_prompt(prompt)
            logger.info(f"  Clean embedding shape: {clean_embedding.shape}")
            logger.info(f"  Model is trained: {model.is_trained}")
        
        logger.info("🎉 Training and testing completed successfully!")
        logger.info("📁 Model saved as 'diffusion_defense_model.pt'")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
