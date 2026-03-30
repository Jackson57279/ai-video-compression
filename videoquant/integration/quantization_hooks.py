"""Quantization hooks for applying VideoQuant to DiT models.

This module provides model quantization functionality that hooks into
Diffusers model forward passes to apply TPQ+SQJL+MAMP quantization.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
import functools

from ..core.pipeline import VideoQuantPipeline, VideoQuantConfig
from ..core.mamp import LayerType


@dataclass
class QuantizationState:
    """State tracking for quantization during inference."""
    timestep: float = 0.5
    layer_counts: Dict[str, int] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.layer_counts is None:
            self.layer_counts = {
                "cross_attention": 0,
                "temporal_attention": 0,
                "self_attention": 0,
                "ffn": 0,
            }


class ModelQuantizer:
    """Applies VideoQuant quantization to a Diffusers transformer model.
    
    This class wraps a DiT model and applies quantization hooks to
    the forward pass, implementing TPQ+SQJL+MAMP quantization.
    
    Attributes:
        model: The transformer model to quantize
        config: VideoQuant configuration
        pipeline: VideoQuantPipeline instance for quantization
        state: Current quantization state
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[VideoQuantConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config or VideoQuantConfig()
        self.pipeline = VideoQuantPipeline(self.config)
        self.state = QuantizationState()
        self.device = device or torch.device("cpu")
        self._original_forwards: Dict[str, Callable] = {}
        self._hooks_installed = False
    
    def quantize_weights(self) -> None:
        """Quantize model weights to configured bit width.
        
        This should be called before inference to quantize static weights.
        """
        weight_bits = self.config.mamp_self_attention_bits  # Default to 4-bit
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "weight" in name and param.ndim > 1:
                    # Simple symmetric quantization for weights
                    scale = param.abs().max() / (2 ** (weight_bits - 1) - 1)
                    if scale > 0:
                        quantized = torch.round(param / scale).clamp(
                            -(2 ** (weight_bits - 1)), 2 ** (weight_bits - 1) - 1
                        )
                        param.data.copy_(quantized * scale)
    
    def _detect_layer_type(self, module_name: str) -> str:
        """Detect layer type from module name."""
        name_lower = module_name.lower()
        
        if "cross" in name_lower or "attn2" in name_lower:
            return "cross_attention"
        elif "temporal" in name_lower:
            return "temporal_attention"
        elif "ffn" in name_lower or "feedforward" in name_lower or "mlp" in name_lower:
            return "ffn"
        elif "attn" in name_lower or "attention" in name_lower:
            return "self_attention"
        else:
            return "ffn"  # Default to FFN (lowest precision)
    
    def _create_quantized_forward(
        self, 
        original_forward: Callable,
        layer_type: str,
    ) -> Callable:
        """Create a quantized forward pass wrapper."""
        
        @functools.wraps(original_forward)
        def quantized_forward(*args, **kwargs):
            if not self.state.enabled:
                return original_forward(*args, **kwargs)
            
            # Quantize inputs if they are tensors
            quantized_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dtype in [torch.float32, torch.float16]:
                    # Handle different tensor dimensions
                    original_shape = arg.shape
                    
                    # Convert to 4D video format if needed [B, F, N, C]
                    if arg.ndim == 2:
                        # [B, C] -> [B, 1, 1, C]
                        arg_4d = arg.unsqueeze(1).unsqueeze(2)
                    elif arg.ndim == 3:
                        # [B, N, C] -> [B, 1, N, C]
                        arg_4d = arg.unsqueeze(1)
                    elif arg.ndim == 4:
                        # Already 4D, assume [B, F, N, C]
                        arg_4d = arg
                    else:
                        # For other dimensions, flatten and reshape
                        B = arg.shape[0] if arg.ndim > 0 else 1
                        flat = arg.reshape(B, -1)
                        # Pad or truncate to make divisible
                        target_size = ((flat.shape[1] + 63) // 64) * 64
                        if flat.shape[1] < target_size:
                            padding = torch.zeros(B, target_size - flat.shape[1], device=flat.device, dtype=flat.dtype)
                            flat = torch.cat([flat, padding], dim=1)
                        else:
                            flat = flat[:, :target_size]
                        # Reshape to [B, 1, N, C]
                        N = flat.shape[1] // 64
                        arg_4d = flat.reshape(B, 1, N, 64)
                    
                    # Apply quantization-dequantization
                    q_result = self.pipeline.quantize(
                        arg_4d, 
                        layer_type=layer_type,
                        timestep=self.state.timestep,
                    )
                    arg_quantized = self.pipeline.dequantize(q_result["quantized_data"])
                    
                    # Restore original shape
                    if arg_quantized.shape != original_shape:
                        arg = arg_quantized.reshape(original_shape)
                    else:
                        arg = arg_quantized
                    
                quantized_args.append(arg)
            
            # Run original forward with quantized inputs
            output = original_forward(*quantized_args, **kwargs)
            
            # Quantize outputs if tensor
            if isinstance(output, torch.Tensor) and output.dtype in [torch.float32, torch.float16]:
                original_shape = output.shape
                
                # Convert to 4D video format if needed
                if output.ndim == 2:
                    output_4d = output.unsqueeze(1).unsqueeze(2)
                elif output.ndim == 3:
                    output_4d = output.unsqueeze(1)
                elif output.ndim == 4:
                    output_4d = output
                else:
                    # For other dimensions
                    B = output.shape[0] if output.ndim > 0 else 1
                    flat = output.reshape(B, -1)
                    target_size = ((flat.shape[1] + 63) // 64) * 64
                    if flat.shape[1] < target_size:
                        padding = torch.zeros(B, target_size - flat.shape[1], device=flat.device, dtype=flat.dtype)
                        flat = torch.cat([flat, padding], dim=1)
                    else:
                        flat = flat[:, :target_size]
                    N = flat.shape[1] // 64
                    output_4d = flat.reshape(B, 1, N, 64)
                
                q_result = self.pipeline.quantize(
                    output_4d,
                    layer_type=layer_type,
                    timestep=self.state.timestep,
                )
                output_quantized = self.pipeline.dequantize(q_result["quantized_data"])
                
                # Restore original shape
                if output_quantized.shape != original_shape:
                    output = output_quantized.reshape(original_shape)
                else:
                    output = output_quantized
            
            return output
        
        return quantized_forward
    
    def install_hooks(self) -> None:
        """Install quantization hooks on the model."""
        if self._hooks_installed:
            return
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_type = self._detect_layer_type(name)
                
                # Store original forward
                self._original_forwards[name] = module.forward
                
                # Install quantized forward
                module.forward = self._create_quantized_forward(
                    module.forward,
                    layer_type,
                )
        
        self._hooks_installed = True
    
    def remove_hooks(self) -> None:
        """Remove quantization hooks and restore original forwards."""
        if not self._hooks_installed:
            return
        
        for name, module in self.model.named_modules():
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
        
        self._original_forwards.clear()
        self._hooks_installed = False
    
    def set_timestep(self, timestep: float) -> None:
        """Update current timestep for timestep-aware quantization."""
        self.state.timestep = timestep
    
    def enable_quantization(self) -> None:
        """Enable quantization."""
        self.state.enabled = True
    
    def disable_quantization(self) -> None:
        """Disable quantization."""
        self.state.enabled = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quantization statistics."""
        return {
            "enabled": self.state.enabled,
            "timestep": self.state.timestep,
            "layer_counts": self.state.layer_counts.copy(),
            "config": {
                "tpq_target_bits": self.config.tpq_target_bits,
                "enable_sqjl": self.config.enable_sqjl,
                "enable_mamp": self.config.enable_mamp,
            },
        }


def apply_quantization_to_model(
    model: nn.Module,
    config: Optional[VideoQuantConfig] = None,
    quantize_weights: bool = True,
    install_hooks: bool = True,
) -> ModelQuantizer:
    """Convenience function to apply quantization to a model.
    
    Args:
        model: The model to quantize
        config: VideoQuant configuration
        quantize_weights: Whether to quantize model weights
        install_hooks: Whether to install forward pass hooks
        
    Returns:
        ModelQuantizer instance managing the quantization
        
    Example:
        >>> from diffusers import WanPipeline
        >>> pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B")
        >>> quantizer = apply_quantization_to_model(
        ...     pipe.transformer,
        ...     quantize_weights=True,
        ...     install_hooks=True,
        ... )
    """
    quantizer = ModelQuantizer(model, config)
    
    if quantize_weights:
        quantizer.quantize_weights()
    
    if install_hooks:
        quantizer.install_hooks()
    
    return quantizer
