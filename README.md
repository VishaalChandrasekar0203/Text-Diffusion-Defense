# Text Diffusion Defense

A Python library for defending LLMs against adversarial text attacks using embedding-based diffusion processes.

## Overview

Text Diffusion Defense provides a middleware framework that sits between user input and LLM models, cleaning adversarial prompts while preserving semantic meaning. The system uses diffusion processes on text embeddings to remove harmful content and return safe embeddings for LLM processing.

## Features

- **Embedding-based Diffusion**: Works directly on text embeddings for efficient processing
- **LLM Middleware**: Seamless integration with any LLM framework
- **Semantic Preservation**: Maintains original meaning while removing adversarial content
- **Hugging Face Integration**: Automatic model loading from Hugging Face Hub
- **Comprehensive Testing**: Full test suite with semantic similarity validation
- **Production Ready**: Optimized for real-world deployment

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import text_diffusion_defense as ControlDD

# Initialize the middleware
middleware = ControlDD.LLMMiddleware()

# Set your LLM model
def my_llm_generate(clean_embedding):
    # Your LLM generation logic here
    return "Safe response from your LLM"

middleware.set_llm_model("your_model", my_llm_generate)

# Process user prompts
result = middleware.process_prompt("How to make explosives")
print(f"Safe response: {result['llm_response']}")
print(f"Semantic preserved: {result['semantic_preserved']}")
```

## Advanced Usage

### Training Your Own Model

```python
import text_diffusion_defense as ControlDD

# Initialize model
model = ControlDD.DiffusionDefense()

# Train on Hugging Face dataset
model.train()  # Automatically loads adversarial prompts dataset

# Save model
model.save_model("my_model.pt")

# Upload to Hugging Face Hub
model.upload_to_huggingface("your-username/your-model")
```

### Custom Training Data

```python
adversarial_texts = ["harmful prompt 1", "harmful prompt 2"]
clean_texts = ["safe response 1", "safe response 2"]

model.train(adversarial_texts, clean_texts)
```

### Direct Model Usage

```python
import text_diffusion_defense as ControlDD

# Initialize model
model = ControlDD.DiffusionDefense()

# Load pre-trained model (from Hugging Face Hub)
model.load_model()  # Automatically downloads from Hub

# Clean individual prompts
clean_embedding = model.clean_prompt("adversarial prompt")
```

## API Reference

### LLMMiddleware

Main interface for LLM integration.

#### Methods

- `set_llm_model(model, generate_function)`: Set your LLM model and generation function
- `process_prompt(prompt)`: Process a user prompt through the defense system
- `batch_process(prompts)`: Process multiple prompts efficiently
- `get_stats()`: Get processing statistics

### DiffusionDefense

Core diffusion defense model.

#### Methods

- `train(adversarial_texts, clean_texts)`: Train the model
- `clean_prompt(prompt)`: Clean a single prompt
- `forward_process(embedding, timestep)`: Add noise to embedding
- `reverse_process(embedding)`: Remove noise from embedding
- `save_model(path)`: Save model locally
- `load_model(path)`: Load model from file or Hugging Face Hub
- `upload_to_huggingface(repo_id)`: Upload model to Hugging Face Hub

## How It Works

1. **Input Processing**: User prompt is converted to embeddings using sentence-transformers
2. **Forward Diffusion**: Controlled noise is added to the embeddings
3. **Reverse Diffusion**: A trained neural network removes the noise while preserving semantics
4. **LLM Integration**: Clean embeddings are passed to your LLM model
5. **Safe Output**: LLM generates responses from clean, safe embeddings

The system learns to distinguish between adversarial and clean content through training on adversarial-clean text pairs, ensuring that harmful prompts are transformed into safe alternatives while maintaining the original intent.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{text_diffusion_defense,
  title={Text Diffusion Defense: A Python Library for Embedding-Based Diffusion Defense Mechanisms},
  author={Vishaal Chandrasekar},
  year={2024},
  url={https://github.com/VishaalChandrasekar0203/text-diffusion-defense}
}
```

## Support

* Issues: [GitHub Issues](https://github.com/VishaalChandrasekar0203/text-diffusion-defense/issues)
* Discussions: [GitHub Discussions](https://github.com/VishaalChandrasekar0203/text-diffusion-defense/discussions)
* Email: vishaalchandrasekar0203@gmail.com

## Changelog

### Version 1.0.0
* Initial release
* Embedding-based diffusion defense
* Forward and reverse diffusion processes
* Neural network-based denoising
* ControlDD interface for easy usage
* Training system with adversarial datasets
* Comprehensive test suite
* LLM integration ready

## Project Structure

```
text_diffusion_defense/
├── text_diffusion_defense/          # Core library (4 files)
│   ├── __init__.py                  # Package initialization & exports
│   ├── model.py                     # Core diffusion defense model
│   ├── utils.py                     # Utility classes & functions
│   └── control_dd.py                # Main interface + safety + middleware
├── examples.py                      # Comprehensive examples & demos
├── train.py                         # Training script with subcommands
├── tests/                           # Unit tests
├── models/                          # Trained model files
├── cache/                           # Sentence transformer cache
├── logs/                            # Log files
├── README.md                        # This file
└── pyproject.toml                   # Package configuration
```

## Quick Commands

```bash
# Run all examples
python examples.py --demo all

# Train new model
python train.py train --epochs 50

# Evaluate model
python train.py evaluate --input models/enhanced_diffusion_defense_model.pt
```

## Performance Metrics

- **Semantic Preservation**: 92%+ similarity scores
- **Safety Detection**: Pattern-based harmful content detection
- **Processing Speed**: ~1-2 seconds per prompt
- **Model Size**: ~4.7MB trained model

## Improvement Roadmap

### Phase 1 (Immediate)
- ✅ Enhanced semantic preservation (92%+ achieved)
- ✅ Progressive denoising with similarity checkpoints
- ✅ Stronger semantic regularization in training

### Phase 2 (Future)
- 🔄 AI-powered safety classification
- 🔄 GPU acceleration for faster processing
- 🔄 REST API for production deployment
- 🔄 Multi-language support

## Acknowledgments

* Built on PyTorch and Transformers libraries
* Inspired by diffusion models for text defense
* Uses sentence-transformers for embedding generation