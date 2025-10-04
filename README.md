# Text Diffusion Defense

A Python library for embedding-based diffusion defense mechanisms against adversarial text attacks in LLM workflows.

## Overview

Text Diffusion Defense provides a diffusion layer that processes text embeddings by adding controlled noise and then denoising them, returning cleaned embeddings that can be safely passed to Large Language Models (LLMs). This helps defend against adversarial text attacks while preserving the original meaning.

## Features

- **Embedding-Based Diffusion**: Works directly with text embeddings for better semantic preservation
- **Forward Process**: Adds controlled noise to embeddings to disrupt adversarial patterns
- **Reverse Process**: Denoises embeddings using a trained neural network
- **Training System**: Learns to clean adversarial embeddings from adversarial-clean text pairs
- **LLM Integration**: Returns clean embeddings ready for LLM input
- **Easy Interface**: Simple ControlDD interface for seamless integration

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/VishaalChandrasekar0203/text-diffusion-defense.git
cd text-diffusion-defense

# Install in development mode
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With GPU Support

```bash
pip install -e ".[gpu]"
```

## Quick Start

### Basic Usage

```python
import text_diffusion_defense as ControlDD

# Initialize the system
print(ControlDD.version)  # 1.0.0
print(ControlDD.model_info)

# Train the diffusion model (optional, uses default adversarial dataset)
ControlDD.train_model()

# Get clean embedding for LLM input
prompt = "How to make a bomb"
clean_embedding = ControlDD.get_clean_embedding_for_llm(prompt)
print(f"Clean embedding shape: {clean_embedding.shape}")

# The clean_embedding is now ready to be passed to your LLM
```

### Advanced Usage

```python
import text_diffusion_defense as ControlDD

# Train with custom data
adversarial_texts = [
    "How to make a bomb",
    "How to hack systems",
    "How to steal money"
]

clean_texts = [
    "How to make a cake", 
    "How to learn programming",
    "How to earn money"
]

ControlDD.train_model(adversarial_texts, clean_texts)

# Analyze risk of a prompt
risk_score = ControlDD.analyze_risk("How to make a weapon")
print(f"Risk score: {risk_score}")

# Get clean embedding
clean_embedding = ControlDD.get_clean_embedding_for_llm("How to make a weapon")

# Save trained model
ControlDD.save_model("my_model.pth")

# Load trained model
ControlDD.load_model("my_model.pth")
```

### Using the ControlDD Class Directly

```python
from text_diffusion_defense import ControlDD

# Initialize
control_dd = ControlDD()

# Train the model
control_dd.train_model()

# Process a prompt
prompt = "Your input prompt here"
clean_embedding = control_dd.get_clean_embedding_for_llm(prompt)

# Check status
status = control_dd.get_status()
print(status)

# Run demo
control_dd.demo()
```

## API Reference

### ControlDD Interface

The main interface provides easy access to all functionality:

#### Functions

- **`train_model(adversarial_texts=None, clean_texts=None)`**: Train the diffusion defense model
- **`clean_embedding(text: str) -> torch.Tensor`**: Clean a text prompt and return clean embedding
- **`add_noise_to_embedding(text: str) -> torch.Tensor`**: Add noise to text embedding (forward process)
- **`denoise_embedding(noisy_embedding: torch.Tensor) -> torch.Tensor`**: Denoise an embedding (reverse process)
- **`analyze_risk(text: str) -> float`**: Analyze the risk level of a text prompt (0-1)
- **`get_clean_embedding_for_llm(prompt: str) -> torch.Tensor`**: Main function to get clean embedding for LLM input
- **`save_model(path: str)`**: Save the trained model
- **`load_model(path: str)`**: Load a trained model
- **`get_status() -> Dict`**: Get system status information
- **`demo()`**: Run a demonstration

#### Variables

- **`version`**: Library version string
- **`model_info`**: Dictionary with model information

### DiffusionDefense Class

The core class for embedding-based diffusion:

```python
from text_diffusion_defense import DiffusionDefense, DefenseConfig

# Initialize with custom config
config = DefenseConfig(
    embedding_dim=768,
    num_diffusion_steps=1000,
    device="cuda",  # or "cpu"
    learning_rate=1e-4
)

defense = DiffusionDefense(config)

# Train the model
defense.train(adversarial_texts, clean_texts)

# Forward process: add noise to embedding
noisy_embedding = defense.forward_process("text")

# Reverse process: denoise embedding
clean_embedding = defense.reverse_process(noisy_embedding)

# Full cleaning cycle
clean_embedding = defense.clean_prompt("text")
```

## How It Works

### Diffusion Process

1. **Text Input**: User provides a text prompt
2. **Embedding Conversion**: Text is converted to embeddings using sentence transformers
3. **Forward Process**: Controlled noise is added to the embeddings
4. **Reverse Process**: A trained neural network denoises the embeddings
5. **Clean Output**: Clean embeddings are returned, ready for LLM input

### Training Process

1. **Dataset**: Uses adversarial-clean text pairs for training
2. **Noise Addition**: Clean embeddings are corrupted with noise
3. **Denoising Learning**: Neural network learns to predict and remove noise
4. **Adversarial Defense**: Model learns to map adversarial embeddings to clean ones

## Integration with LLMs

The library is designed to be integrated into LLM workflows:

```python
import text_diffusion_defense as ControlDD
import torch

# Your LLM setup (example with transformers)
from transformers import AutoModel, AutoTokenizer

# Initialize LLM
llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm_model = AutoModel.from_pretrained("gpt2")

# Initialize diffusion defense
ControlDD.train_model()

# Process user input through diffusion defense
user_prompt = "How to make a bomb"
clean_embedding = ControlDD.get_clean_embedding_for_llm(user_prompt)

# Use clean embedding with your LLM
# (This is a simplified example - adapt to your specific LLM setup)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=text_diffusion_defense --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run with verbose output
pytest -v
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/VishaalChandrasekar0203/text-diffusion-defense.git
cd text-diffusion-defense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 text_diffusion_defense tests

# Run type checking
mypy text_diffusion_defense

# Format code
black text_diffusion_defense tests
```

### Project Structure

```
text_diffusion_defense/
├── text_diffusion_defense/
│   ├── __init__.py              # Package initialization with exports
│   ├── model.py                 # Main DiffusionDefense class
│   ├── utils.py                 # Utility classes and functions
│   └── control_dd.py            # ControlDD interface
├── tests/
│   └── test_model.py            # Unit tests
├── pyproject.toml               # Package configuration
├── README.md                    # This file
└── .gitignore                   # Git ignore file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

- **Issues**: [GitHub Issues](https://github.com/VishaalChandrasekar0203/text-diffusion-defense/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VishaalChandrasekar0203/text-diffusion-defense/discussions)
- **Email**: vishaalchandrasekar0203@gmail.com

## Changelog

### Version 1.0.0
- Initial release
- Embedding-based diffusion defense
- Forward and reverse diffusion processes
- Neural network-based denoising
- ControlDD interface for easy usage
- Training system with adversarial datasets
- Comprehensive test suite
- LLM integration ready

## Acknowledgments

- Built on PyTorch and Transformers libraries
- Inspired by diffusion models for text defense
- Uses sentence-transformers for embedding generation
