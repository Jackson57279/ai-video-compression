"""Configuration classes for VideoQuant Diffusers integration.

Provides configuration dataclasses for quantization parameters
that can be passed to the Diffusers pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class QuantizationConfig:
    """Configuration for quantization settings.
    
    Args:
        weight_bits: Number of bits for weight quantization (default: 4 for W4)
        activation_bits: Number of bits for activation quantization (default: 4 for A4)
        enable_tpq: Enable Temporal-Polar Quantization for activations
        enable_sqjl: Enable Spatial-QJL residual correction
        enable_mamp: Enable Metric-Aware Mixed Precision
        mixed_precision: Use layer-specific precision (cross-attn: 6, temporal: 5, self: 4, FFN: 3)
    """
    weight_bits: int = 4
    activation_bits: int = 4
    enable_tpq: bool = True
    enable_sqjl: bool = True
    enable_mamp: bool = True
    mixed_precision: bool = True
    
    # TPQ specific
    tpq_target_bits: float = 3.5
    tpq_radii_allocation: float = 0.6
    
    # SQJL specific
    sqjl_projection_dim: int = 256
    
    # MAMP specific
    mamp_cross_attention_bits: int = 6
    mamp_temporal_attention_bits: int = 5
    mamp_self_attention_bits: int = 4
    mamp_ffn_bits: int = 3
    mamp_timestep_scale_early: float = 1.0
    mamp_timestep_scale_late: float = 1.3


@dataclass
class VideoQuantDiffusersConfig:
    """Configuration for VideoQuant Diffusers pipeline integration.
    
    This configuration can be passed to VideoQuantDiffusersPipeline
    to control quantization behavior.
    
    Args:
        quantization: Quantization configuration settings
        device: Target device for inference ("cpu", "cuda", etc.)
        dtype: Data type for model weights
        compile_model: Whether to use torch.compile() on the transformer
        memory_efficient: Enable memory-efficient attention
        cpu_offload: Enable CPU offloading for large models
    """
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    device: str = "cpu"
    dtype: str = "fp16"
    compile_model: bool = False
    memory_efficient: bool = True
    cpu_offload: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "quantization": {
                "weight_bits": self.quantization.weight_bits,
                "activation_bits": self.quantization.activation_bits,
                "enable_tpq": self.quantization.enable_tpq,
                "enable_sqjl": self.quantization.enable_sqjl,
                "enable_mamp": self.quantization.enable_mamp,
                "mixed_precision": self.quantization.mixed_precision,
            },
            "device": self.device,
            "dtype": self.dtype,
            "compile_model": self.compile_model,
            "memory_efficient": self.memory_efficient,
            "cpu_offload": self.cpu_offload,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VideoQuantDiffusersConfig":
        """Create config from dictionary."""
        quant_config = config_dict.get("quantization", {})
        quantization = QuantizationConfig(
            weight_bits=quant_config.get("weight_bits", 4),
            activation_bits=quant_config.get("activation_bits", 4),
            enable_tpq=quant_config.get("enable_tpq", True),
            enable_sqjl=quant_config.get("enable_sqjl", True),
            enable_mamp=quant_config.get("enable_mamp", True),
            mixed_precision=quant_config.get("mixed_precision", True),
        )
        
        return cls(
            quantization=quantization,
            device=config_dict.get("device", "cpu"),
            dtype=config_dict.get("dtype", "fp16"),
            compile_model=config_dict.get("compile_model", False),
            memory_efficient=config_dict.get("memory_efficient", True),
            cpu_offload=config_dict.get("cpu_offload", False),
        )
    
    @classmethod
    def default_w4a4(cls) -> "VideoQuantDiffusersConfig":
        """Create default W4A4 configuration."""
        return cls(
            quantization=QuantizationConfig(
                weight_bits=4,
                activation_bits=4,
                enable_tpq=True,
                enable_sqjl=True,
                enable_mamp=True,
                mixed_precision=True,
            ),
            device="cpu",
            dtype="fp16",
        )
    
    @classmethod
    def fp16_baseline(cls) -> "VideoQuantDiffusersConfig":
        """Create FP16 baseline (no quantization) configuration."""
        return cls(
            quantization=QuantizationConfig(
                weight_bits=16,
                activation_bits=16,
                enable_tpq=False,
                enable_sqjl=False,
                enable_mamp=False,
                mixed_precision=False,
            ),
            device="cpu",
            dtype="fp16",
        )
