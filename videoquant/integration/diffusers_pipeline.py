"""VideoQuant Diffusers Pipeline Wrapper.

This module provides a Diffusers-compatible pipeline wrapper that integrates
VideoQuant quantization with the standard Diffusers API.

Usage:
    >>> from videoquant.integration import VideoQuantDiffusersPipeline, VideoQuantDiffusersConfig
    >>> 
    >>> config = VideoQuantDiffusersConfig.default_w4a4()
    >>> pipe = VideoQuantDiffusersPipeline.from_pretrained(
    ...     "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    ...     videoquant_config=config,
    ... )
    >>> 
    >>> video = pipe(
    ...     prompt="a cat playing piano",
    ...     num_frames=16,
    ...     height=512,
    ...     width=512,
    ... ).frames[0]

Validates:
- VAL-INT-002: Diffusers Integration
"""

import torch
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass

# Diffusers imports
try:
    from diffusers import DiffusionPipeline
    from diffusers.pipelines.wan import WanPipeline
    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    DiffusionPipeline = object
    WanPipeline = object

# Optional imports (may not be available in all diffusers versions)
try:
    from diffusers.pipelines.wan import WanPipelineOutput
except ImportError:
    WanPipelineOutput = object

from .config import VideoQuantDiffusersConfig, QuantizationConfig
from .quantization_hooks import ModelQuantizer, apply_quantization_to_model


@dataclass
class VideoQuantPipelineOutput:
    """Output from VideoQuantDiffusersPipeline.
    
    Attributes:
        frames: Generated video frames [num_frames, channels, height, width]
        quant_stats: Quantization statistics from generation
        config: Configuration used for generation
    """
    frames: torch.Tensor
    quant_stats: Optional[Dict[str, Any]] = None
    config: Optional[VideoQuantDiffusersConfig] = None
    
    def __getitem__(self, key: int) -> torch.Tensor:
        """Allow frame access via output[0]."""
        if key == 0:
            return self.frames
        raise IndexError(f"VideoQuantPipelineOutput only supports index 0, got {key}")


class VideoQuantDiffusersPipeline:
    """VideoQuant-enabled pipeline for Diffusers video generation models.
    
    This pipeline wraps standard Diffusers pipelines and adds VideoQuant
    quantization support, including:
    - W4A4 weight and activation quantization
    - TPQ+SQJL+MAMP three-stage quantization
    - Configurable precision via pipeline parameters
    - Compatibility with model.compile() and device placement
    
    The pipeline maintains full compatibility with the standard Diffusers API,
    adding only the `videoquant_config` parameter for quantization control.
    
    Attributes:
        base_pipeline: The underlying Diffusers pipeline
        quantizer: ModelQuantizer managing quantization state
        config: VideoQuantDiffusersConfig with quantization settings
        
    Example:
        >>> from videoquant.integration import VideoQuantDiffusersPipeline
        >>> 
        >>> # Load with default W4A4 quantization
        >>> pipe = VideoQuantDiffusersPipeline.from_pretrained(
        ...     "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ... )
        >>> 
        >>> # Generate with quantization automatically applied
        >>> video = pipe(
        ...     prompt="a cat playing piano",
        ...     num_frames=16,
        ...     height=512,
        ...     width=512,
        ...     num_inference_steps=50,
        ... ).frames[0]
    """
    
    def __init__(
        self,
        base_pipeline: DiffusionPipeline,
        config: Optional[VideoQuantDiffusersConfig] = None,
    ):
        """Initialize the VideoQuant Diffusers pipeline.
        
        Args:
            base_pipeline: The underlying Diffusers pipeline to wrap
            config: VideoQuant configuration (uses W4A4 default if None)
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers is required for VideoQuantDiffusersPipeline. "
                "Install with: pip install diffusers"
            )
        
        self.base_pipeline = base_pipeline
        self.config = config or VideoQuantDiffusersConfig.default_w4a4()
        self.quantizer: Optional[ModelQuantizer] = None
        
        # Apply quantization to the transformer if it exists
        self._apply_quantization()
    
    def _apply_quantization(self) -> None:
        """Apply VideoQuant quantization to the pipeline's transformer."""
        if not self.config.quantization.enable_tpq:
            # Quantization disabled
            return
        
        # Get the transformer from the base pipeline
        transformer = getattr(self.base_pipeline, "transformer", None)
        if transformer is None:
            # Some pipelines may have a different structure
            return
        
        # Create VideoQuantConfig from our config
        from ..core.pipeline import VideoQuantConfig
        
        vq_config = VideoQuantConfig(
            tpq_target_bits=self.config.quantization.tpq_target_bits,
            tpq_radii_allocation=self.config.quantization.tpq_radii_allocation,
            enable_sqjl=self.config.quantization.enable_sqjl,
            enable_mamp=self.config.quantization.enable_mamp,
            mamp_cross_attention_bits=self.config.quantization.mamp_cross_attention_bits,
            mamp_temporal_attention_bits=self.config.quantization.mamp_temporal_attention_bits,
            mamp_self_attention_bits=self.config.quantization.mamp_self_attention_bits,
            mamp_ffn_bits=self.config.quantization.mamp_ffn_bits,
            mamp_timestep_scale_early=self.config.quantization.mamp_timestep_scale_early,
            mamp_timestep_scale_late=self.config.quantization.mamp_timestep_scale_late,
        )
        
        # Apply quantization
        self.quantizer = apply_quantization_to_model(
            transformer,
            config=vq_config,
            quantize_weights=self.config.quantization.weight_bits < 16,
            install_hooks=self.config.quantization.enable_tpq,
        )
        
        # Apply torch.compile if requested
        if self.config.compile_model and hasattr(torch, "compile"):
            try:
                transformer = torch.compile(transformer, mode="reduce-overhead")
                self.base_pipeline.transformer = transformer
            except Exception as e:
                # torch.compile may not be available on all platforms
                print(f"Warning: torch.compile failed: {e}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        videoquant_config: Optional[VideoQuantDiffusersConfig] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "VideoQuantDiffusersPipeline":
        """Load a VideoQuant-enabled pipeline from pretrained weights.
        
        This is the main entry point for loading a quantized pipeline.
        It delegates to the underlying Diffusers pipeline loader and then
        wraps it with VideoQuant quantization.
        
        Args:
            pretrained_model_name_or_path: Model ID or path (e.g., "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
            videoquant_config: VideoQuant configuration (uses default W4A4 if None)
            torch_dtype: Data type for model weights
            device: Device to load the model on
            **kwargs: Additional arguments passed to DiffusionPipeline.from_pretrained()
            
        Returns:
            VideoQuantDiffusersPipeline instance with quantization enabled
            
        Example:
            >>> pipe = VideoQuantDiffusersPipeline.from_pretrained(
            ...     "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            ...     videoquant_config=VideoQuantDiffusersConfig.default_w4a4(),
            ...     torch_dtype=torch.float16,
            ...     device="cpu",
            ... )
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers is required for VideoQuantDiffusersPipeline. "
                "Install with: pip install diffusers"
            )
        
        config = videoquant_config or VideoQuantDiffusersConfig.default_w4a4()
        
        # Determine dtype
        if torch_dtype is None:
            if config.dtype == "fp16":
                torch_dtype = torch.float16
            elif config.dtype == "bf16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        
        # Load the base pipeline using Diffusers' loader
        # Try to auto-detect the pipeline type
        base_pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        
        # Move to device if specified
        if device is not None:
            base_pipeline = base_pipeline.to(device)
        elif config.device:
            base_pipeline = base_pipeline.to(config.device)
        
        # Create wrapped pipeline
        wrapped = cls(base_pipeline, config)
        
        return wrapped
    
    def to(self, device: Union[str, torch.device]) -> "VideoQuantDiffusersPipeline":
        """Move the pipeline to a device.
        
        Maintains compatibility with standard Diffusers device placement.
        
        Args:
            device: Target device ("cpu", "cuda:0", etc.)
            
        Returns:
            self (for method chaining)
        """
        self.base_pipeline = self.base_pipeline.to(device)
        if self.quantizer:
            self.quantizer.device = device if isinstance(device, torch.device) else torch.device(device)
        return self
    
    def cpu(self) -> "VideoQuantDiffusersPipeline":
        """Move pipeline to CPU."""
        return self.to("cpu")
    
    def cuda(self, device: Optional[int] = None) -> "VideoQuantDiffusersPipeline":
        """Move pipeline to CUDA.
        
        Args:
            device: CUDA device index (None for default)
        """
        if device is None:
            return self.to("cuda")
        return self.to(f"cuda:{device}")
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = "",
        negative_prompt: Union[str, List[str]] = "",
        num_inference_steps: int = 50,
        num_frames: int = 16,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> VideoQuantPipelineOutput:
        """Generate video with VideoQuant quantization.
        
        This method maintains the standard Diffusers API while automatically
        applying quantization during generation.
        
        Args:
            prompt: Text prompt(s) describing the desired video
            negative_prompt: Negative prompt(s) for what to avoid
            num_inference_steps: Number of denoising steps
            num_frames: Number of video frames to generate
            height: Video height in pixels
            width: Video width in pixels
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to the base pipeline
            
        Returns:
            VideoQuantPipelineOutput containing generated frames and stats
            
        Example:
            >>> output = pipe(
            ...     prompt="a cat playing piano",
            ...     num_frames=16,
            ...     height=512,
            ...     width=512,
            ...     num_inference_steps=50,
            ...     seed=42,
            ... )
            >>> frames = output.frames  # [16, 3, 512, 512]
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Get scheduler for timestep access
        scheduler = getattr(self.base_pipeline, "scheduler", None)
        
        # Store quant_stats during generation
        quant_stats_list: List[Dict[str, Any]] = []
        
        # Define callback for timestep-aware quantization
        def quant_callback(pipe, step_index, timestep, callback_kwargs):
            """Callback to update quantization timestep."""
            if self.quantizer:
                # Convert timestep to normalized [0, 1] range
                # Diffusion: timestep is steps from end (high = early, low = late)
                total_steps = num_inference_steps
                normalized_t = 1.0 - (step_index / total_steps)
                self.quantizer.set_timestep(normalized_t)
                
                # Collect stats
                stats = self.quantizer.get_stats()
                quant_stats_list.append({
                    "step": step_index,
                    "timestep": timestep.item() if hasattr(timestep, "item") else timestep,
                    "normalized_t": normalized_t,
                    **stats,
                })
            
            return callback_kwargs
        
        # Prepare callback kwargs
        callback_on_step_end = quant_callback
        callback_on_step_end_tensor_inputs = ["latents"]
        
        # Call the base pipeline
        # Note: We pass the callback for timestep-aware quantization
        output = self.base_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            **kwargs,
        )
        
        # Extract frames from output
        frames = output.frames[0] if hasattr(output, "frames") else output
        
        # Return wrapped output
        return VideoQuantPipelineOutput(
            frames=frames,
            quant_stats={
                "steps": len(quant_stats_list),
                "final_timestep": self.quantizer.state.timestep if self.quantizer else 0,
                "quantization_enabled": self.config.quantization.enable_tpq,
            } if self.quantizer else None,
            config=self.config,
        )
    
    def enable_model_cpu_offload(self, gpu_id: int = 0) -> None:
        """Enable CPU offloading for memory-efficient inference.
        
        Maintains compatibility with Diffusers' memory optimization features.
        
        Args:
            gpu_id: GPU device ID to use
        """
        if hasattr(self.base_pipeline, "enable_model_cpu_offload"):
            self.base_pipeline.enable_model_cpu_offload(gpu_id)
    
    def enable_sequential_cpu_offload(self, gpu_id: int = 0) -> None:
        """Enable sequential CPU offloading.
        
        Args:
            gpu_id: GPU device ID to use
        """
        if hasattr(self.base_pipeline, "enable_sequential_cpu_offload"):
            self.base_pipeline.enable_sequential_cpu_offload(gpu_id)
    
    def enable_vae_slicing(self) -> None:
        """Enable VAE slicing for lower memory usage."""
        if hasattr(self.base_pipeline, "enable_vae_slicing"):
            self.base_pipeline.enable_vae_slicing()
    
    def disable_vae_slicing(self) -> None:
        """Disable VAE slicing."""
        if hasattr(self.base_pipeline, "disable_vae_slicing"):
            self.base_pipeline.disable_vae_slicing()
    
    def get_quantization_stats(self) -> Optional[Dict[str, Any]]:
        """Get current quantization statistics.
        
        Returns:
            Dictionary with quantization statistics, or None if not quantized
        """
        if self.quantizer is None:
            return None
        return self.quantizer.get_stats()
    
    @property
    def device(self) -> torch.device:
        """Get the device the pipeline is on."""
        return self.base_pipeline.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the pipeline."""
        return self.base_pipeline.dtype
    
    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """Save the pipeline to a directory.
        
        Saves the base pipeline and VideoQuant configuration.
        
        Args:
            save_directory: Directory to save to
            safe_serialization: Use safetensors format
            **kwargs: Additional save arguments
        """
        # Save base pipeline
        self.base_pipeline.save_pretrained(
            save_directory,
            safe_serialization=safe_serialization,
            **kwargs,
        )
        
        # Save VideoQuant config
        import json
        config_path = f"{save_directory}/videoquant_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def load_quantized(
        cls,
        model_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "VideoQuantDiffusersPipeline":
        """Load a previously saved quantized pipeline.
        
        Args:
            model_path: Path to the saved pipeline
            device: Device to load on
            **kwargs: Additional load arguments
            
        Returns:
            VideoQuantDiffusersPipeline instance
        """
        import json
        import os
        
        # Load config if it exists
        config_path = os.path.join(model_path, "videoquant_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = VideoQuantDiffusersConfig.from_dict(config_dict)
        else:
            config = VideoQuantDiffusersConfig.default_w4a4()
        
        return cls.from_pretrained(
            model_path,
            videoquant_config=config,
            device=device,
            **kwargs,
        )
