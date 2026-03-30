"""VideoQuant Core Quantization Algorithms"""

from .tpq import TPQQuantizer, TPQConfig
from .sqjl import SQJLQuantizer, SQJLConfig, estimate_attention_with_sqjl
from .mamp import (
    MAMPAllocator,
    MAMPQuantizer,
    MAMPConfig,
    LayerType,
    create_default_mamp_config,
    get_precision_for_layer,
)
from .pipeline import (
    VideoQuantPipeline,
    VideoQuantConfig,
    PipelineStats,
    PipelineStage,
    create_default_pipeline,
    quantize_tensor,
    quantize_dequantize_tensor,
)
from .kernels import (
    CPUOptimizedKernels,
    get_kernels,
    reset_kernels,
    cartesian_to_polar_optimized,
    polar_to_cartesian_optimized,
    quantize_symmetric_optimized,
    dequantize_symmetric_optimized,
    jl_projection_optimized,
    sign_quantize_optimized,
    sign_dequantize_optimized,
    NUMBA_AVAILABLE,
)

__all__ = [
    # TPQ
    "TPQQuantizer",
    "TPQConfig",
    # SQJL
    "SQJLQuantizer",
    "SQJLConfig",
    "estimate_attention_with_sqjl",
    # MAMP
    "MAMPAllocator",
    "MAMPQuantizer",
    "MAMPConfig",
    "LayerType",
    "create_default_mamp_config",
    "get_precision_for_layer",
    # Pipeline
    "VideoQuantPipeline",
    "VideoQuantConfig",
    "PipelineStats",
    "PipelineStage",
    "create_default_pipeline",
    "quantize_tensor",
    "quantize_dequantize_tensor",
    # Kernels
    "CPUOptimizedKernels",
    "get_kernels",
    "reset_kernels",
    "cartesian_to_polar_optimized",
    "polar_to_cartesian_optimized",
    "quantize_symmetric_optimized",
    "dequantize_symmetric_optimized",
    "jl_projection_optimized",
    "sign_quantize_optimized",
    "sign_dequantize_optimized",
    "NUMBA_AVAILABLE",
]
