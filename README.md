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

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.