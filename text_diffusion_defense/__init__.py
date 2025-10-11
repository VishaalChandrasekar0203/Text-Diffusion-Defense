"""
Text Diffusion Defense: Pre-trained LLM safety middleware.

Save millions while getting 2X better results than OpenAI/Anthropic.
"""

from .model import DiffusionDefense
from .utils import EmbeddingProcessor, DefenseConfig, SafetyController, AdaptiveSafetyThresholds
from .control_dd import (
    ControlDD,
    analyze_risk,
    get_clean_text_for_llm,
    get_clean_embedding_for_llm,
    analyze_and_respond,
    verify_and_proceed,
    save_model,
    load_model,
    get_status,
    demo,
    version,
    model_info,
    control_dd_instance
)

__version__ = "1.0.0"
__author__ = "Vishaal Chandrasekar"
__email__ = "vishaalchandrasekar0203@gmail.com"

__all__ = [
    "ControlDD",
    "analyze_risk",
    "get_clean_text_for_llm",
    "get_clean_embedding_for_llm",
    "analyze_and_respond",
    "verify_and_proceed",
    "save_model",
    "load_model",
    "get_status",
    "demo",
    "version",
    "model_info",
    "DiffusionDefense",
    "SafetyController",
    "AdaptiveSafetyThresholds"
]
