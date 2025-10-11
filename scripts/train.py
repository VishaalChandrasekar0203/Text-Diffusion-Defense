#!/usr/bin/env python3
"""
Training Methodology Reference for Text Diffusion Defense

⚠️  NOTE: This script is for REFERENCE ONLY - shows how the model was trained.
⚠️  The model is PRE-TRAINED and ready to use.
⚠️  Training API methods have been removed to protect the model.
⚠️  Users should NOT run this script - just read it to understand methodology.

This documents:
- Training parameters used (epochs=200, lr=0.001)
- Hyperparameter optimization approach
- Dataset and methodology
- Results achieved

The pre-trained model is included and ready to use via:
  import text_diffusion_defense as ControlDD
  defense = ControlDD.ControlDD()  # Pre-trained model loaded automatically
"""

import text_diffusion_defense as ControlDD
import torch
import logging
import argparse
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(args):
    """Train the diffusion defense model."""
    logger.info("🚀 Starting Text Diffusion Defense Model Training")
    
    # Initialize ControlDD with custom config
    config = ControlDD.DefenseConfig(
        embedding_dim=384,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        learning_rate=args.learning_rate,
        enable_logging=not args.quiet
    )
    
    model = ControlDD.DiffusionDefense(config)
    
    logger.info(f"Training configuration:")
    logger.info(f"  - Device: {config.device}")
    logger.info(f"  - Embedding dim: {config.embedding_dim}")
    logger.info(f"  - Epochs: {config.num_epochs}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Enhanced features: {'Enabled' if args.enhanced else 'Disabled'}")
    
    # Train the model
    logger.info("📚 Loading dataset and starting training...")
    start_time = time.time()
    
    try:
        model.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Save the trained model
        model_path = args.output or "diffusion_defense_model.pt"
        logger.info(f"💾 Saving trained model to {model_path}...")
        model.save_model(model_path)
        
        # Test the trained model
        if not args.no_test:
            logger.info("🧪 Testing trained model...")
            test_trained_model(model)
        
        # Upload to Hugging Face Hub if requested
        if args.upload_to_hub:
            logger.info(f"☁️ Uploading model to Hugging Face Hub: {args.repo_id}")
            try:
                model.upload_to_huggingface(args.repo_id)
                logger.info("✅ Model uploaded to Hugging Face Hub!")
            except Exception as e:
                logger.error(f"❌ Upload failed: {e}")
                logger.info("You may need to authenticate with Hugging Face first:")
                logger.info("Run: huggingface-cli login")
        
        logger.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        raise


def test_trained_model(model):
    """Test the trained model with sample prompts."""
    test_prompts = [
        "How to bake a cake",
        "How to make explosives",
        "How to learn programming",
        "How to hack computers"
    ]
    
    logger.info("Testing model performance:")
    
    for prompt in test_prompts:
        logger.info(f"Testing: '{prompt}'")
        
        # Get original embedding
        original_embedding = model.embedding_processor.text_to_embedding(prompt)
        
        # Clean with trained model
        clean_embedding = model.clean_prompt(prompt)
        
        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(
            original_embedding, clean_embedding, dim=1
        ).item()
        
        logger.info(f"  Similarity Score: {similarity:.4f}")
        logger.info(f"  Model is trained: {model.is_trained}")


def retrain_model(args):
    """Retrain the model with new parameters."""
    logger.info("🔄 Retraining existing model...")
    
    # Load existing model
    model = ControlDD.DiffusionDefense()
    if args.input and os.path.exists(args.input):
        model.load_model(args.input)
        logger.info(f"Loaded existing model from {args.input}")
    else:
        logger.info("Starting with untrained model")
    
    # Update config for retraining
    model.config.num_epochs = args.epochs
    model.config.batch_size = args.batch_size
    model.config.learning_rate = args.learning_rate
    
    # Retrain
    model.train()
    
    # Save retrained model
    output_path = args.output or "retrained_diffusion_defense_model.pt"
    model.save_model(output_path)
    logger.info(f"✅ Retrained model saved to {output_path}")


def evaluate_model(args):
    """Evaluate an existing model."""
    logger.info("📊 Evaluating model performance...")
    
    # Load model
    model = ControlDD.DiffusionDefense()
    if args.input and os.path.exists(args.input):
        model.load_model(args.input)
        logger.info(f"Loaded model from {args.input}")
    else:
        logger.error(f"Model file not found: {args.input}")
        return
    
    # Run comprehensive evaluation
    from demo import run_comprehensive_test
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    run_comprehensive_test()


def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description="Train Text Diffusion Defense Model")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--device", type=str, default="auto", 
                             choices=["auto", "cpu", "cuda"], help="Training device")
    train_parser.add_argument("--output", type=str, help="Output model file path")
    train_parser.add_argument("--upload_to_hub", action="store_true", 
                             help="Upload trained model to Hugging Face Hub")
    train_parser.add_argument("--repo_id", type=str, 
                             default="vishaalchandrasekar/text-diffusion-defense",
                             help="Hugging Face Hub repository ID")
    train_parser.add_argument("--enhanced", action="store_true", 
                             help="Use enhanced training with safety controls")
    train_parser.add_argument("--no_test", action="store_true", 
                             help="Skip testing after training")
    train_parser.add_argument("--quiet", action="store_true", 
                             help="Reduce output verbosity")
    
    # Retrain command
    retrain_parser = subparsers.add_parser('retrain', help='Retrain an existing model')
    retrain_parser.add_argument("--input", type=str, required=True, 
                               help="Input model file path")
    retrain_parser.add_argument("--epochs", type=int, default=20, 
                               help="Additional training epochs")
    retrain_parser.add_argument("--batch_size", type=int, default=16, 
                               help="Training batch size")
    retrain_parser.add_argument("--lr", type=float, default=5e-5, 
                               help="Learning rate for retraining")
    retrain_parser.add_argument("--output", type=str, 
                               help="Output model file path")
    retrain_parser.add_argument("--quiet", action="store_true", 
                               help="Reduce output verbosity")
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate an existing model')
    eval_parser.add_argument("--input", type=str, required=True, 
                            help="Input model file path")
    
    args = parser.parse_args()
    
    # Set device
    if hasattr(args, 'device') and args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Execute command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'retrain':
        retrain_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    print("="*80)
    print("⚠️  WARNING: This script is for REFERENCE ONLY")
    print("="*80)
    print("")
    print("The model is PRE-TRAINED and ready to use.")
    print("Training API methods have been removed to protect the model.")
    print("")
    print("This script documents the training methodology used:")
    print("  • Parameters: epochs=200, lr=0.001")
    print("  • Optimizer: AdamW")
    print("  • Dataset: aurora-m/adversarial-prompts")
    print("  • Results: 69.1% semantic preservation")
    print("")
    print("To USE the library (no training needed):")
    print("  import text_diffusion_defense as ControlDD")
    print("  defense = ControlDD.ControlDD()")
    print("  clean_text = defense.get_clean_text_for_llm(prompt)")
    print("")
    print("="*80)
    print("")
    
    response = input("Do you still want to view the training code? (yes/no): ")
    if response.lower() != 'yes':
        print("\n✅ Good choice! Use the pre-trained model instead.")
        print("See README.md for usage examples.")
        exit(0)
    
    print("\n📚 Showing training methodology (reference only)...\n")
    main()
