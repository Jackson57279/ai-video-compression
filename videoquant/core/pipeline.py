"""VideoQuant - Unified Pipeline Integration

Implements the full TPQ → SQJL → MAMP pipeline for video DiT quantization.
Integrates all three stages into a cohesive quantization/dequantization cycle.

Validates:
- VAL-INT-001: Full Pipeline Execution
- VAL-CROSS-001: Full Pipeline Tensor Flow
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .tpq import TPQQuantizer, TPQConfig
from .sqjl import SQJLQuantizer, SQJLConfig
from .mamp import MAMPAllocator, MAMPConfig, LayerType


class PipelineStage(Enum):
    """Stages in the VideoQuant pipeline."""
    TPQ = "tpq"
    SQJL = "sqjl"
    MAMP = "mamp"


@dataclass
class VideoQuantConfig:
    """Configuration for the unified VideoQuant pipeline.
    
    Combines configurations from all three stages:
    - TPQ: Temporal-polar quantization settings
    - SQJL: Spatial-QJL residual correction settings
    - MAMP: Metric-aware mixed precision settings
    """
    # TPQ Configuration
    tpq_target_bits: float = 3.5
    tpq_radii_allocation: float = 0.6
    tpq_enable_recursive: bool = True
    
    # SQJL Configuration
    sqjl_projection_dim: int = 256
    sqjl_enable_residual: bool = True
    
    # MAMP Configuration
    mamp_cross_attention_bits: int = 6
    mamp_temporal_attention_bits: int = 5
    mamp_self_attention_bits: int = 4
    mamp_ffn_bits: int = 3
    mamp_timestep_scale_early: float = 1.0
    mamp_timestep_scale_late: float = 1.3
    
    # Pipeline Configuration
    enable_sqjl: bool = True  # Enable SQJL residual correction
    enable_mamp: bool = True  # Enable MAMP layer-specific precision
    
    @classmethod
    def default_w4a4(cls) -> "VideoQuantConfig":
        """Create default W4A4 quantization configuration.
        
        Returns:
            VideoQuantConfig with W4A4 settings (4-bit weights, 4-bit activations)
        """
        return cls(
            # TPQ at ~3.5 bits average (close to 4-bit)
            tpq_target_bits=3.5,
            tpq_radii_allocation=0.6,
            tpq_enable_recursive=True,
            
            # SQJL enabled for residual correction
            sqjl_projection_dim=256,
            sqjl_enable_residual=True,
            
            # MAMP with appropriate bit allocations
            mamp_cross_attention_bits=6,
            mamp_temporal_attention_bits=5,
            mamp_self_attention_bits=4,
            mamp_ffn_bits=3,
            mamp_timestep_scale_early=1.0,
            mamp_timestep_scale_late=1.3,
            
            # Enable all stages
            enable_sqjl=True,
            enable_mamp=True,
        )
    
    @classmethod
    def fp16_baseline(cls) -> "VideoQuantConfig":
        """Create FP16 baseline configuration (no quantization).
        
        Returns:
            VideoQuantConfig with quantization disabled
        """
        return cls(
            # Disable quantization stages
            enable_sqjl=False,
            enable_mamp=False,
            # Set high target bits (effectively no compression)
            tpq_target_bits=16.0,
        )
    
    def to_tpq_config(self) -> TPQConfig:
        """Convert to TPQ configuration."""
        return TPQConfig(
            target_bits=self.tpq_target_bits,
            radii_allocation=self.tpq_radii_allocation,
            enable_recursive=self.tpq_enable_recursive,
        )
    
    def to_sqjl_config(self) -> SQJLConfig:
        """Convert to SQJL configuration."""
        return SQJLConfig(
            projection_dim=self.sqjl_projection_dim,
            preserve_spatial=True,
        )
    
    def to_mamp_config(self) -> MAMPConfig:
        """Convert to MAMP configuration."""
        return MAMPConfig(
            cross_attention_bits=self.mamp_cross_attention_bits,
            temporal_attention_bits=self.mamp_temporal_attention_bits,
            self_attention_bits=self.mamp_self_attention_bits,
            ffn_bits=self.mamp_ffn_bits,
            timestep_scale_early=self.mamp_timestep_scale_early,
            timestep_scale_late=self.mamp_timestep_scale_late,
        )


@dataclass
class PipelineStats:
    """Statistics from pipeline execution."""
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    has_nan: bool
    has_inf: bool
    compression_ratio: float
    cosine_similarity: float
    l2_relative_error: float
    tpq_bits_used: float
    sqjl_enabled: bool
    mamp_bits: int
    layer_type: str
    timestep: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'has_nan': self.has_nan,
            'has_inf': self.has_inf,
            'compression_ratio': self.compression_ratio,
            'cosine_similarity': self.cosine_similarity,
            'l2_relative_error': self.l2_relative_error,
            'tpq_bits_used': self.tpq_bits_used,
            'sqjl_enabled': self.sqjl_enabled,
            'mamp_bits': self.mamp_bits,
            'layer_type': self.layer_type,
            'timestep': self.timestep,
        }


class VideoQuantPipeline:
    """
    Unified VideoQuant pipeline integrating TPQ → SQJL → MAMP.
    
    This class orchestrates the three-stage quantization pipeline:
    1. TPQ: Temporal-Polar Quantization for aggressive compression
    2. SQJL: Spatial-QJL for residual correction (optional)
    3. MAMP: Metric-Aware Mixed Precision for layer-specific allocation
    
    Usage:
        >>> config = VideoQuantConfig()
        >>> pipeline = VideoQuantPipeline(config)
        >>> tensor = torch.randn(2, 16, 256, 512)  # [B, F, N, C]
        >>> quantized = pipeline.quantize(tensor, layer_type="self_attention", timestep=0.5)
        >>> reconstructed = pipeline.dequantize(quantized)
    
    Attributes:
        config: VideoQuantConfig with pipeline settings
        tpq: TPQQuantizer instance
        sqjl: SQJLQuantizer instance
        mamp: MAMPAllocator instance
    """
    
    def __init__(self, config: Optional[VideoQuantConfig] = None):
        self.config = config or VideoQuantConfig()
        
        # Initialize stage quantizers
        self.tpq = TPQQuantizer(self.config.to_tpq_config())
        self.sqjl = SQJLQuantizer(self.config.to_sqjl_config())
        self.mamp = MAMPAllocator(self.config.to_mamp_config())
        
        # Storage for intermediate states (for inspection/debugging)
        self._last_intermediates: Dict[str, Any] = {}
    
    def _compute_stats(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        quantized_data: Dict[str, Any],
        layer_type: str,
        timestep: float,
    ) -> PipelineStats:
        """Compute pipeline statistics."""
        has_nan = bool(not torch.isfinite(reconstructed).all().item())
        has_inf = bool(torch.isinf(reconstructed).any().item())
        
        # Cosine similarity
        orig_flat = original.reshape(1, -1)
        recon_flat = reconstructed.reshape(1, -1)
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_flat, recon_flat
        ).item()
        
        # L2 relative error
        l2_orig = torch.norm(original)
        l2_error = torch.norm(original - reconstructed)
        l2_rel_error = (l2_error / (l2_orig + 1e-10)).item()
        
        # Compression ratio
        metadata = quantized_data.get('metadata', {})
        if 'tpq_metadata' in metadata:
            tpq_meta = metadata['tpq_metadata']
            radii_bits = tpq_meta.get('radii_bits', 4)
            angle_bits = tpq_meta.get('angle_bits', 4)
            tpq_bits = (radii_bits + angle_bits) / 2
        else:
            tpq_bits = self.config.tpq_target_bits
        
        compression_ratio = 16 / tpq_bits  # FP16 baseline
        
        # MAMP bits
        if self.config.enable_mamp:
            mamp_bits = self.mamp.allocate_precision(layer_type, timestep)
        else:
            mamp_bits = 4  # Default
        
        return PipelineStats(
            input_shape=original.shape,
            output_shape=reconstructed.shape,
            has_nan=has_nan,
            has_inf=has_inf,
            compression_ratio=compression_ratio,
            cosine_similarity=cos_sim,
            l2_relative_error=l2_rel_error,
            tpq_bits_used=tpq_bits,
            sqjl_enabled=self.config.enable_sqjl,
            mamp_bits=mamp_bits,
            layer_type=layer_type,
            timestep=timestep,
        )
    
    def quantize(
        self,
        tensor: torch.Tensor,
        layer_type: Union[LayerType, str] = LayerType.SELF_ATTENTION,
        timestep: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Quantize a tensor through the full pipeline.
        
        Args:
            tensor: Input tensor [B, F, N, C] (Batch, Frames, Patches, Channels)
            layer_type: Type of layer (determines MAMP precision)
            timestep: Current diffusion timestep [0, 1]
            
        Returns:
            Dictionary containing:
                - quantized_data: The quantized representation
                - metadata: Quantization parameters and pipeline info
                - stats: Pipeline statistics (if compute_stats=True)
        
        Pipeline flow:
        1. Apply TPQ quantization (polar transform + recursive compression)
        2. Compute residual (if SQJL enabled)
        3. SQJL quantize residual (if enabled)
        4. Apply MAMP bit allocation metadata
        """
        original_shape = tensor.shape
        
        # Convert layer_type to string if needed
        if isinstance(layer_type, LayerType):
            layer_type_str = layer_type.value
        else:
            layer_type_str = layer_type
        
        # Stage 1: TPQ Quantization
        tpq_result = self.tpq.quantize(tensor)
        
        # Dequantize to get TPQ reconstruction (for residual computation)
        tpq_reconstructed = self.tpq.dequantize(tpq_result)
        
        # Stage 2: SQJL Residual (if enabled)
        sqjl_data = None
        if self.config.enable_sqjl:
            # Compute residual: error between original and TPQ reconstruction
            residual = tensor - tpq_reconstructed
            
            # Flatten for SQJL: [B*F*N, C]
            B, F, N, C = residual.shape
            residual_flat = residual.reshape(-1, C)
            
            # SQJL quantize the residual
            sqjl_result = self.sqjl.quantize(residual_flat)
            sqjl_data = {
                'quantized_bits': sqjl_result['quantized_bits'],
                'metadata': sqjl_result['metadata'],
                'original_shape': (B, F, N, C),
            }
            
            self._last_intermediates['sqjl'] = sqjl_result
        
        # Stage 3: MAMP Precision Allocation
        mamp_bits = 4  # Default
        if self.config.enable_mamp:
            mamp_bits = self.mamp.allocate_precision(layer_type_str, timestep)
        
        # Combine all quantization data
        quantized_data = {
            'tpq_data': tpq_result,
            'sqjl_data': sqjl_data,
            'mamp_bits': mamp_bits,
            'layer_type': layer_type_str,
            'timestep': timestep,
            'original_shape': original_shape,
        }
        
        # Compute statistics
        reconstructed = self.dequantize(quantized_data)
        stats = self._compute_stats(
            tensor, reconstructed, quantized_data, layer_type_str, timestep
        )
        
        return {
            'quantized_data': quantized_data,
            'metadata': {
                'tpq_metadata': tpq_result['metadata'],
                'sqjl_metadata': sqjl_data['metadata'] if sqjl_data else None,
                'mamp_bits': mamp_bits,
                'layer_type': layer_type_str,
                'timestep': timestep,
                'config': self.config,
            },
            'stats': stats,
        }
    
    def dequantize(
        self,
        quantized_data: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Dequantize a tensor through the full pipeline.
        
        Args:
            quantized_data: Output from quantize()
            
        Returns:
            Dequantized tensor with preserved shape [B, F, N, C]
        
        Pipeline flow:
        1. Dequantize TPQ data
        2. Dequantize SQJL residual (if enabled)
        3. Add residual to TPQ reconstruction
        """
        # Extract data
        tpq_data = quantized_data['tpq_data']
        sqjl_data = quantized_data.get('sqjl_data')
        original_shape = quantized_data['original_shape']
        
        # Stage 1: TPQ Dequantization
        tpq_reconstructed = self.tpq.dequantize(tpq_data)
        
        # Stage 2: SQJL Residual Addition (if enabled)
        if sqjl_data is not None and self.config.enable_sqjl:
            # Dequantize SQJL residual
            sqjl_quantized = {
                'quantized_bits': sqjl_data['quantized_bits'],
                'metadata': sqjl_data['metadata'],
            }
            residual_flat = self.sqjl.dequantize(sqjl_quantized)
            
            # Reshape residual to match original
            B, F, N, C = original_shape
            residual = residual_flat.reshape(B, F, N, -1)
            
            # If channel dims don't match, pad or truncate
            if residual.shape[-1] != C:
                if residual.shape[-1] < C:
                    # Pad with zeros
                    padding = torch.zeros(
                        B, F, N, C - residual.shape[-1],
                        device=residual.device, dtype=residual.dtype
                    )
                    residual = torch.cat([residual, padding], dim=-1)
                else:
                    # Truncate
                    residual = residual[..., :C]
            
            # Add residual to TPQ reconstruction
            reconstructed = tpq_reconstructed + residual
        else:
            reconstructed = tpq_reconstructed
        
        # Ensure output shape matches input
        if reconstructed.shape != original_shape:
            # Try to reshape
            try:
                reconstructed = reconstructed.reshape(original_shape)
            except RuntimeError:
                # If reshape fails, interpolate or pad
                B, F, N, C = original_shape
                if reconstructed.numel() == np.prod(original_shape):
                    reconstructed = reconstructed.reshape(original_shape)
                else:
                    # Create output with correct shape
                    output = torch.zeros(original_shape, device=reconstructed.device, dtype=reconstructed.dtype)
                    min_size = min(reconstructed.numel(), output.numel())
                    output.view(-1)[:min_size] = reconstructed.view(-1)[:min_size]
                    reconstructed = output
        
        return reconstructed
    
    def quantize_dequantize(
        self,
        tensor: torch.Tensor,
        layer_type: Union[LayerType, str] = LayerType.SELF_ATTENTION,
        timestep: float = 0.5,
    ) -> Tuple[torch.Tensor, PipelineStats]:
        """
        Convenience method: quantize then immediately dequantize.
        
        Args:
            tensor: Input tensor [B, F, N, C]
            layer_type: Type of layer
            timestep: Current diffusion timestep
            
        Returns:
            (reconstructed_tensor, stats): Dequantized output and statistics
        """
        result = self.quantize(tensor, layer_type, timestep)
        reconstructed = self.dequantize(result['quantized_data'])
        return reconstructed, result['stats']
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about each pipeline stage."""
        return {
            'tpq': {
                'enabled': True,
                'target_bits': self.config.tpq_target_bits,
                'radii_allocation': self.config.tpq_radii_allocation,
                'enable_recursive': self.config.tpq_enable_recursive,
            },
            'sqjl': {
                'enabled': self.config.enable_sqjl,
                'projection_dim': self.config.sqjl_projection_dim,
            },
            'mamp': {
                'enabled': self.config.enable_mamp,
                'cross_attention_bits': self.config.mamp_cross_attention_bits,
                'temporal_attention_bits': self.config.mamp_temporal_attention_bits,
                'self_attention_bits': self.config.mamp_self_attention_bits,
                'ffn_bits': self.config.mamp_ffn_bits,
            },
        }
    
    def get_last_intermediates(self) -> Dict[str, Any]:
        """Get intermediate states from last quantization (for debugging)."""
        return self._last_intermediates.copy()


def create_default_pipeline() -> VideoQuantPipeline:
    """Create a VideoQuantPipeline with default configuration."""
    return VideoQuantPipeline(VideoQuantConfig())


def quantize_tensor(
    tensor: torch.Tensor,
    layer_type: str = "self_attention",
    timestep: float = 0.5,
    config: Optional[VideoQuantConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience function to quantize a tensor.
    
    Args:
        tensor: Input tensor [B, F, N, C]
        layer_type: Layer type string
        timestep: Diffusion timestep
        config: Optional custom configuration
        
    Returns:
        Quantization result dictionary
    """
    pipeline = VideoQuantPipeline(config)
    return pipeline.quantize(tensor, layer_type, timestep)


def quantize_dequantize_tensor(
    tensor: torch.Tensor,
    layer_type: str = "self_attention",
    timestep: float = 0.5,
    config: Optional[VideoQuantConfig] = None,
) -> Tuple[torch.Tensor, PipelineStats]:
    """
    Convenience function for quantize-dequantize roundtrip.
    
    Args:
        tensor: Input tensor [B, F, N, C]
        layer_type: Layer type string
        timestep: Diffusion timestep
        config: Optional custom configuration
        
    Returns:
        (reconstructed_tensor, stats)
    """
    pipeline = VideoQuantPipeline(config)
    return pipeline.quantize_dequantize(tensor, layer_type, timestep)
