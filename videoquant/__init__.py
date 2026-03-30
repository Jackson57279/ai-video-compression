"""VideoQuant - Extreme Video Model Compression Library"""

from .core.tpq import TPQQuantizer, TPQConfig
from .core.sqjl import SQJLQuantizer, SQJLConfig, estimate_attention_with_sqjl
from .core.mamp import (
    MAMPAllocator,
    MAMPQuantizer,
    MAMPConfig,
    LayerType,
    create_default_mamp_config,
    get_precision_for_layer,
)
from .core.pipeline import (
    VideoQuantPipeline,
    VideoQuantConfig,
    PipelineStats,
    PipelineStage,
    create_default_pipeline,
    quantize_tensor,
    quantize_dequantize_tensor,
)

# Diffusers integration (optional dependency)
try:
    from .integration import (
        VideoQuantDiffusersPipeline,
        VideoQuantDiffusersConfig,
        QuantizationConfig,
        ModelQuantizer,
        apply_quantization_to_model,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    VideoQuantDiffusersPipeline = None  # type: ignore
    VideoQuantDiffusersConfig = None  # type: ignore
    QuantizationConfig = None  # type: ignore
    ModelQuantizer = None  # type: ignore
    apply_quantization_to_model = None  # type: ignore

__all__ = [
    # Core algorithms
    "TPQQuantizer",
    "TPQConfig",
    "SQJLQuantizer",
    "SQJLConfig",
    "estimate_attention_with_sqjl",
    "MAMPAllocator",
    "MAMPQuantizer",
    "MAMPConfig",
    "LayerType",
    "create_default_mamp_config",
    "get_precision_for_layer",
    "VideoQuantPipeline",
    "VideoQuantConfig",
    "PipelineStats",
    "PipelineStage",
    "create_default_pipeline",
    "quantize_tensor",
    "quantize_dequantize_tensor",
    # Diffusers integration (if available)
    "VideoQuantDiffusersPipeline",
    "VideoQuantDiffusersConfig",
    "QuantizationConfig",
    "ModelQuantizer",
    "apply_quantization_to_model",
    "DIFFUSERS_AVAILABLE",
]
