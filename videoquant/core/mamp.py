#!/usr/bin/env python3
"""
VideoQuant - Metric-Aware Mixed Precision (MAMP) Implementation

Implements layer-type specific precision allocation with timestep-aware scaling:
- Cross-attention: 6 bits (text alignment critical for CLIPSIM)
- Temporal-attention: 5 bits (smoothness critical for temporal consistency)
- Self-attention: 4 bits (balanced quality/compression)
- FFN: 3 bits (less sensitive to quantization)

Timestep scaling:
- Early timesteps (high noise): 1.0x base precision
- Late timesteps (refining): 1.3x base precision
- Smooth transition throughout denoising process

Validates:
- VAL-MAMP-001: Layer-type precision assignment
- VAL-MAMP-002: Timestep-aware allocation
- VAL-MAMP-004: Cross-attention high precision protection
- VAL-MAMP-005: Temporal consistency optimization
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union, List, Any
from dataclasses import dataclass, field
from enum import Enum


class LayerType(Enum):
    """DiT layer types for precision allocation."""
    CROSS_ATTENTION = "cross_attention"
    TEMPORAL_ATTENTION = "temporal_attention"
    SELF_ATTENTION = "self_attention"
    FFN = "ffn"


@dataclass
class MAMPConfig:
    """Configuration for Metric-Aware Mixed Precision."""
    # Base bit allocations per layer type (VAL-MAMP-001)
    cross_attention_bits: int = 6  # Text alignment critical
    temporal_attention_bits: int = 5  # Smoothness critical
    self_attention_bits: int = 4  # Balanced
    ffn_bits: int = 3  # Less sensitive
    
    # Timestep scaling (VAL-MAMP-002)
    timestep_scale_early: float = 1.0  # Early denoising (high noise)
    timestep_scale_late: float = 1.3  # Late denoising (refining)
    
    # Smooth transition parameters
    timestep_schedule: str = "linear"  # "linear" or "sigmoid"
    
    # Metric preservation targets (VAL-MAMP-003)
    target_fid_preservation: float = 0.99  # 99% FID preservation
    target_clipsim_preservation: float = 0.99  # 99% CLIPSIM preservation
    target_temporal_preservation: float = 0.99  # 99% temporal consistency
    
    # Minimum/maximum bits enforcement
    min_bits: int = 2  # Never go below 2 bits
    max_bits: int = 8  # Never exceed 8 bits
    
    # Layer sensitivity weights for metric impact
    layer_sensitivity: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "cross_attention": {
            "spatial_quality": 0.3,
            "temporal_consistency": 0.1,
            "text_alignment": 0.9,  # High impact on CLIPSIM
        },
        "temporal_attention": {
            "spatial_quality": 0.2,
            "temporal_consistency": 0.9,  # High impact on smoothness
            "text_alignment": 0.1,
        },
        "self_attention": {
            "spatial_quality": 0.7,
            "temporal_consistency": 0.3,
            "text_alignment": 0.1,
        },
        "ffn": {
            "spatial_quality": 0.6,
            "temporal_consistency": 0.4,
            "text_alignment": 0.2,
        },
    })


class MAMPAllocator:
    """
    Metric-Aware Mixed Precision allocator for DiT layers.
    
    Allocates precision (bits) to different layer types based on:
    1. Layer sensitivity to different quality metrics
    2. Diffusion timestep (early vs late denoising)
    3. Target metric preservation thresholds
    
    Attributes:
        config: MAMPConfig with allocation parameters
    """
    
    def __init__(self, config: Optional[MAMPConfig] = None):
        self.config = config or MAMPConfig()
        self._validate_config()
        
        # Build lookup table for layer type to base bits
        self._base_bits = {
            LayerType.CROSS_ATTENTION: self.config.cross_attention_bits,
            LayerType.TEMPORAL_ATTENTION: self.config.temporal_attention_bits,
            LayerType.SELF_ATTENTION: self.config.self_attention_bits,
            LayerType.FFN: self.config.ffn_bits,
        }
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Ensure cross-attention has highest precision (VAL-MAMP-004)
        assert self.config.cross_attention_bits >= self.config.temporal_attention_bits, \
            "Cross-attention must have >= bits than temporal-attention"
        assert self.config.cross_attention_bits >= self.config.self_attention_bits, \
            "Cross-attention must have >= bits than self-attention"
        assert self.config.cross_attention_bits >= self.config.ffn_bits, \
            "Cross-attention must have >= bits than FFN"
        
        # Ensure temporal attention >= self-attention (VAL-MAMP-005)
        assert self.config.temporal_attention_bits >= self.config.self_attention_bits, \
            "Temporal-attention must have >= bits than self-attention"
        
        # Validate timestep scaling
        assert 0.5 <= self.config.timestep_scale_early <= 2.0, \
            "Early timestep scale should be in [0.5, 2.0]"
        assert 0.5 <= self.config.timestep_scale_late <= 2.0, \
            "Late timestep scale should be in [0.5, 2.0]"
        assert self.config.timestep_scale_late >= self.config.timestep_scale_early, \
            "Late timestep scale should be >= early scale"
        
        # Validate bit bounds
        assert 1 <= self.config.min_bits <= 16, "min_bits must be in [1, 16]"
        assert 1 <= self.config.max_bits <= 16, "max_bits must be in [1, 16]"
        assert self.config.min_bits <= self.config.max_bits, \
            "min_bits must be <= max_bits"
    
    def get_base_bits(self, layer_type: Union[LayerType, str]) -> int:
        """
        Get base precision (bits) for a layer type.
        
        Args:
            layer_type: Layer type enum or string name
            
        Returns:
            Base bit allocation for the layer type
        """
        if isinstance(layer_type, str):
            layer_type = LayerType(layer_type)
        return self._base_bits[layer_type]
    
    def compute_timestep_scale(self, timestep: float) -> float:
        """
        Compute timestep-dependent scaling factor.
        
        Args:
            timestep: Current diffusion timestep in [0, 1]
                - 0.0 = start of denoising (high noise, early)
                - 1.0 = end of denoising (low noise, late)
                
        Returns:
            Scaling factor in [timestep_scale_early, timestep_scale_late]
            
        Note:
            timestep=0 (early, high noise) -> scale = timestep_scale_early (1.0)
            timestep=1 (late, refining) -> scale = timestep_scale_late (1.3)
        """
        # Clamp timestep to valid range
        t = np.clip(timestep, 0.0, 1.0)
        
        early_scale = self.config.timestep_scale_early
        late_scale = self.config.timestep_scale_late
        
        if self.config.timestep_schedule == "linear":
            # Linear interpolation: scale = early + t * (late - early)
            scale = early_scale + t * (late_scale - early_scale)
        elif self.config.timestep_schedule == "sigmoid":
            # Smooth sigmoid transition centered at t=0.5
            # sigmoid((t - 0.5) * 10) maps t in [0,1] to approximately [0, 1]
            sigmoid_val = 1.0 / (1.0 + np.exp(-(t - 0.5) * 10))
            scale = early_scale + sigmoid_val * (late_scale - early_scale)
        else:
            raise ValueError(f"Unknown timestep schedule: {self.config.timestep_schedule}")
        
        return float(scale)
    
    def allocate_precision(
        self,
        layer_type: Union[LayerType, str],
        timestep: float,
        metric_budget: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Allocate precision (bits) for a layer at a given timestep.
        
        Args:
            layer_type: Type of layer (cross_attention, temporal_attention, etc.)
            timestep: Current diffusion timestep in [0, 1]
            metric_budget: Optional override for metric preservation targets
            
        Returns:
            Number of bits to use for quantization (int)
            
        Note:
            Implements VAL-MAMP-001 and VAL-MAMP-002.
        """
        # Get base bits for layer type
        base_bits = self.get_base_bits(layer_type)
        
        # Compute timestep scaling factor
        scale = self.compute_timestep_scale(timestep)
        
        # Calculate scaled bits
        scaled_bits = base_bits * scale
        
        # Round to nearest integer and clamp to valid range
        allocated_bits = int(round(scaled_bits))
        allocated_bits = max(self.config.min_bits, min(self.config.max_bits, allocated_bits))
        
        return allocated_bits
    
    def allocate_all_layers(self, timestep: float) -> Dict[str, int]:
        """
        Allocate precision for all layer types at a given timestep.
        
        Args:
            timestep: Current diffusion timestep in [0, 1]
            
        Returns:
            Dictionary mapping layer type names to bit allocations
        """
        allocations = {}
        for layer_type in LayerType:
            allocations[layer_type.value] = self.allocate_precision(layer_type, timestep)
        return allocations
    
    def get_layer_sensitivity(self, layer_type: Union[LayerType, str]) -> Dict[str, float]:
        """
        Get sensitivity scores for a layer type across quality metrics.
        
        Args:
            layer_type: Layer type enum or string
            
        Returns:
            Dictionary mapping metric names to sensitivity scores [0, 1]
        """
        if isinstance(layer_type, LayerType):
            layer_type = layer_type.value
        return self.config.layer_sensitivity.get(layer_type, {})
    
    def compute_metric_impact(
        self,
        layer_type: Union[LayerType, str],
        bits: int,
        baseline_bits: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate impact on quality metrics for a given bit allocation.
        
        Args:
            layer_type: Layer type being quantized
            bits: Number of bits allocated
            baseline_bits: Reference bit count (default: FP16=16)
            
        Returns:
            Dictionary mapping metric names to estimated preservation ratios
        """
        if baseline_bits is None:
            baseline_bits = 16  # FP16 baseline
        
        sensitivity = self.get_layer_sensitivity(layer_type)
        
        # Simple model: impact proportional to (bits / baseline) ^ sensitivity
        # Lower bits -> more impact on high-sensitivity metrics
        bit_ratio = bits / baseline_bits
        
        impacts = {}
        for metric, sens in sensitivity.items():
            # High sensitivity + low bits = significant degradation
            # Model: preservation = 1 - (1 - bit_ratio) * sensitivity
            preservation = 1.0 - (1.0 - bit_ratio) * sens
            impacts[metric] = max(0.0, min(1.0, preservation))
        
        return impacts
    
    def verify_metric_preservation(
        self,
        allocations: Dict[str, int],
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Verify that allocations meet metric preservation targets.
        
        Args:
            allocations: Layer type -> bits mapping
            target_metrics: Metric -> target preservation (default: from config)
            
        Returns:
            (passes, metric_scores): Whether targets are met and detailed scores
            
        Note:
            Implements VAL-MAMP-003 validation.
        """
        if target_metrics is None:
            target_metrics = {
                "spatial_quality": self.config.target_fid_preservation,
                "text_alignment": self.config.target_clipsim_preservation,
                "temporal_consistency": self.config.target_temporal_preservation,
            }
        
        # Aggregate impact across all layers
        aggregated_impact = {metric: 1.0 for metric in target_metrics.keys()}
        
        for layer_name, bits in allocations.items():
            layer_type = LayerType(layer_name)
            impacts = self.compute_metric_impact(layer_type, bits)
            
            for metric in aggregated_impact:
                if metric in impacts:
                    # Multiply impacts (assuming independence approximation)
                    aggregated_impact[metric] *= impacts[metric]
        
        # Check against targets
        passes = True
        for metric, target in target_metrics.items():
            if aggregated_impact[metric] < target:
                passes = False
        
        return passes, aggregated_impact
    
    def get_allocation_table(self, timesteps: Optional[List[float]] = None) -> Dict[str, Dict[float, int]]:
        """
        Generate allocation table for visualization/analysis.
        
        Args:
            timesteps: List of timesteps to evaluate (default: 0.0 to 1.0)
            
        Returns:
            Nested dict: layer_type -> timestep -> bits
        """
        if timesteps is None:
            timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        table = {}
        for layer_type in LayerType:
            table[layer_type.value] = {}
            for t in timesteps:
                table[layer_type.value][t] = self.allocate_precision(layer_type, t)
        
        return table


class MAMPQuantizer:
    """
    High-level MAMP quantizer that wraps other quantizers with layer-aware precision.
    
    This class provides a convenient interface for applying different quantization
    strategies based on layer type and timestep.
    """
    
    def __init__(self, config: Optional[MAMPConfig] = None):
        self.config = config or MAMPConfig()
        self.allocator = MAMPAllocator(self.config)
    
    def quantize_for_layer(
        self,
        tensor: torch.Tensor,
        layer_type: Union[LayerType, str],
        timestep: float,
        quantizer_fn: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Quantize a tensor with MAMP-aware precision for a specific layer.
        
        Args:
            tensor: Input tensor to quantize
            layer_type: Type of layer (determines precision)
            timestep: Current diffusion timestep
            quantizer_fn: Optional external quantizer function
            
        Returns:
            Dictionary with quantized data and metadata
        """
        # Get allocated bits for this layer and timestep
        bits = self.allocator.allocate_precision(layer_type, timestep)
        
        result = {
            "bits": bits,
            "layer_type": layer_type.value if isinstance(layer_type, LayerType) else layer_type,
            "timestep": timestep,
            "scale": self.allocator.compute_timestep_scale(timestep),
            "original_shape": tensor.shape,
            "quantized": None,  # Would be filled by actual quantization
        }
        
        return result
    
    def get_layer_profile(self, timestep: float = 0.5) -> Dict[str, Any]:
        """
        Get full layer precision profile at a given timestep.
        
        Args:
            timestep: Diffusion timestep to evaluate
            
        Returns:
            Dictionary with layer allocations and metric estimates
        """
        allocations = self.allocator.allocate_all_layers(timestep)
        
        # Compute metric preservation estimates
        passes, metric_scores = self.allocator.verify_metric_preservation(allocations)
        
        return {
            "timestep": timestep,
            "allocations": allocations,
            "metric_preservation": metric_scores,
            "meets_targets": passes,
        }


def create_default_mamp_config() -> MAMPConfig:
    """Create MAMP configuration with default settings matching validation contract."""
    return MAMPConfig(
        cross_attention_bits=6,
        temporal_attention_bits=5,
        self_attention_bits=4,
        ffn_bits=3,
        timestep_scale_early=1.0,
        timestep_scale_late=1.3,
        timestep_schedule="linear",
    )


def get_precision_for_layer(
    layer_type: str,
    timestep: float,
    config: Optional[MAMPConfig] = None
) -> int:
    """
    Convenience function to get precision for a layer type at a timestep.
    
    Args:
        layer_type: "cross_attention", "temporal_attention", "self_attention", or "ffn"
        timestep: Diffusion timestep in [0, 1]
        config: Optional MAMP config (uses default if None)
        
    Returns:
        Number of bits to use for quantization
    """
    allocator = MAMPAllocator(config)
    return allocator.allocate_precision(layer_type, timestep)
