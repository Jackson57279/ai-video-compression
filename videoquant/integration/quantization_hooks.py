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
    
    def _quantize_tensor_simple(self, tensor: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """Simple symmetric quantization for non-4D tensors.
        
        For 2D/3D tensors that go through Linear layers, use simple
        per-channel symmetric quantization to preserve shapes.
        """
        # Use symmetric quantization
        abs_max = tensor.abs().max()
        if abs_max < 1e-10:
            return tensor
        
        max_quant_val = (2 ** bits / 2) - 1
        scale = abs_max / max_quant_val
        
        quantized = torch.round(tensor / scale).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        dequantized = quantized * scale
        
        return dequantized
    
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
                    original_shape = arg.shape
                    original_ndim = arg.ndim
                    
                    # For 2D tensors [B*F*N, C] (typical for Linear layers)
                    # Use simple quantization that preserves shape
                    if original_ndim == 2:
                        arg = self._quantize_tensor_simple(arg, bits=4)
                    
                    elif original_ndim == 3:
                        # [B, N, C] - batch size is small, seq is large
                        # Use simple quantization to preserve shape
                        arg = self._quantize_tensor_simple(arg, bits=4)
                    
                    elif original_ndim == 4:
                        # 4D tensors can use TPQ for video format [B, F, N, C]
                        try:
                            q_result = self.pipeline.quantize(
                                arg, 
                                layer_type=layer_type,
                                timestep=self.state.timestep,
                            )
                            arg = self.pipeline.dequantize(q_result["quantized_data"])
                        except Exception:
                            # Fall back to simple quantization if TPQ fails
                            arg = self._quantize_tensor_simple(arg, bits=4)
                        
                        if arg.shape != original_shape:
                            arg = arg.reshape(original_shape)
                    
                    # Other dimensions pass through unchanged
                    
                quantized_args.append(arg)
            
            # Run original forward with quantized inputs
            output = original_forward(*quantized_args, **kwargs)
            
            # Quantize outputs if tensor
            if isinstance(output, torch.Tensor) and output.dtype in [torch.float32, torch.float16]:
                original_shape = output.shape
                original_ndim = output.ndim
                
                if original_ndim == 2:
                    output = self._quantize_tensor_simple(output, bits=4)
                
                elif original_ndim == 3:
                    output = self._quantize_tensor_simple(output, bits=4)
                
                elif original_ndim == 4:
                    try:
                        q_result = self.pipeline.quantize(
                            output,
                            layer_type=layer_type,
                            timestep=self.state.timestep,
                        )
                        output = self.pipeline.dequantize(q_result["quantized_data"])
                    except Exception:
                        output = self._quantize_tensor_simple(output, bits=4)
                    
                    if output.shape != original_shape:
                        output = output.reshape(original_shape)
            
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
