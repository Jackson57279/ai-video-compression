"""VideoQuant Diffusers Integration

Integrates VideoQuant with HuggingFace Diffusers pipeline.
Provides quantization-aware pipeline wrapper for video DiT models.

Validates:
- VAL-INT-002: Diffusers Integration
"""

from .config import VideoQuantDiffusersConfig, QuantizationConfig
from .diffusers_pipeline import VideoQuantDiffusersPipeline
from .quantization_hooks import ModelQuantizer, apply_quantization_to_model

__all__ = [
    "VideoQuantDiffusersConfig",
    "QuantizationConfig",
    "VideoQuantDiffusersPipeline",
    "ModelQuantizer",
    "apply_quantization_to_model",
]
