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

__all__ = [
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
]
