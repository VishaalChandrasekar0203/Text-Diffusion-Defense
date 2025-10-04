"""
Text Diffusion Defense: A Python library for embedding-based diffusion defense mechanisms.

This library provides tools for adding controlled noise to text embeddings and cleaning them back,
useful for defending against adversarial text attacks in LLM workflows.
"""

from .model import DiffusionDefense
from .utils import EmbeddingProcessor, DefenseConfig
from .control_dd import (
    ControlDD,
    train_model,
    clean_embedding,
    add_noise_to_embedding,
    denoise_embedding,
    analyze_risk,
    get_clean_embedding_for_llm,
    save_model,
    load_model,
    get_status,
    demo,
    version,
    model_info,
    control_dd_instance
)
from .llm_middleware import LLMMiddleware, LLMIntegrationExample

__version__ = "1.0.0"
__author__ = "Vishaal Chandrasekar"
__email__ = "vishaalchandrasekar0203@gmail.com"

# Main interface for easy access
__all__ = [
    "DiffusionDefense",
    "EmbeddingProcessor",
    "DefenseConfig",
    "ControlDD",
    "LLMMiddleware",
    "LLMIntegrationExample",
    "train_model",
    "clean_embedding",
    "add_noise_to_embedding",
    "denoise_embedding",
    "analyze_risk",
    "get_clean_embedding_for_llm",
    "save_model",
    "load_model",
    "get_status",
    "demo",
    "version",
    "model_info"
]

# Make ControlDD functions available at module level
# This allows users to call ControlDD.function_name() directly
