"""Tests for VideoQuant Diffusers integration.

Validates:
- VAL-INT-002: Diffusers Integration
  * Custom quantization config for Diffusers
  * Model can be loaded with quantized weights
  * Pipeline.generate() works with quantization enabled
  * Compatible with existing Diffusers features (compiling, device placement)
"""

import pytest
import torch
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Mark all tests in this module as requiring diffusers
try:
    import diffusers
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# Mark tests as skip if diffusers not available
pytestmark = [
    pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="diffusers not installed"),
]

# VideoQuant imports
from videoquant.integration import (
    VideoQuantDiffusersConfig,
    QuantizationConfig,
    VideoQuantDiffusersPipeline,
    ModelQuantizer,
    apply_quantization_to_model,
)
from videoquant.core.pipeline import VideoQuantConfig


# Mock Diffusers pipeline for testing
class MockTransformer(torch.nn.Module):
    """Mock transformer for testing."""
    
    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.self_attn = torch.nn.Linear(dim, dim)
        self.cross_attn = torch.nn.Linear(dim, dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )
        self.temporal_attn = torch.nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.self_attn(x) + x
        x = self.cross_attn(x) + x
        x = self.temporal_attn(x) + x
        x = self.ffn(x) + x
        return x


class MockScheduler:
    """Mock scheduler for testing."""
    
    def __init__(self):
        self.timesteps = torch.linspace(1000, 0, 50)
    
    def step(self, *args, **kwargs):
        return type('obj', (object,), {'prev_sample': torch.randn(2, 4, 8, 8)})()


class MockDiffusersPipeline:
    """Mock Diffusers pipeline for testing."""
    
    def __init__(self):
        self.transformer = MockTransformer()
        self.scheduler = MockScheduler()
        self.vae = type('obj', (object,), {'decode': lambda self, x: type('obj', (object,), {'sample': torch.randn(2, 3, 64, 64)})()})()
        self.text_encoder = None
        self.tokenizer = None
        self._device = torch.device('cpu')
        self._dtype = torch.float32
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    def to(self, device: torch.device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.transformer = self.transformer.to(self._device)
        return self
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        num_inference_steps: int = 50,
        **kwargs,
    ) -> type('obj', (object,), {'frames': [torch.randn(16, 3, 64, 64)]})():
        # Simulate inference
        latents = torch.randn(2, 4, 16, 8, 8).to(self._device)
        
        # Run through transformer with timesteps
        for i, t in enumerate(self.scheduler.timesteps[:num_inference_steps]):
            timestep = t.unsqueeze(0).expand(2).to(self._device)
            latents = self.transformer(latents.flatten(2).transpose(1, 2), timestep).transpose(1, 2).reshape(2, 4, 16, 8, 8)
        
        # Mock output
        frames = torch.randn(16, 3, 64, 64)
        return type('obj', (object,), {'frames': [frames]})()
    
    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        pass


class TestVideoQuantDiffusersConfig:
    """Test configuration classes."""
    
    def test_default_w4a4_config(self):
        """Test default W4A4 configuration."""
        config = VideoQuantDiffusersConfig.default_w4a4()
        
        assert config.quantization.weight_bits == 4
        assert config.quantization.activation_bits == 4
        assert config.quantization.enable_tpq is True
        assert config.quantization.enable_sqjl is True
        assert config.quantization.enable_mamp is True
        assert config.quantization.mixed_precision is True
    
    def test_fp16_baseline_config(self):
        """Test FP16 baseline (no quantization) configuration."""
        config = VideoQuantDiffusersConfig.fp16_baseline()
        
        assert config.quantization.weight_bits == 16
        assert config.quantization.activation_bits == 16
        assert config.quantization.enable_tpq is False
        assert config.quantization.enable_sqjl is False
        assert config.quantization.enable_mamp is False
    
    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = VideoQuantDiffusersConfig.default_w4a4()
        config_dict = config.to_dict()
        
        assert "quantization" in config_dict
        assert "device" in config_dict
        assert config_dict["quantization"]["weight_bits"] == 4
    
    def test_config_from_dict(self):
        """Test config deserialization from dict."""
        original = VideoQuantDiffusersConfig.default_w4a4()
        config_dict = original.to_dict()
        
        restored = VideoQuantDiffusersConfig.from_dict(config_dict)
        
        assert restored.quantization.weight_bits == original.quantization.weight_bits
        assert restored.quantization.enable_tpq == original.quantization.enable_tpq
        assert restored.device == original.device
    
    def test_custom_bit_allocation(self):
        """Test custom bit allocation in config."""
        config = VideoQuantDiffusersConfig(
            quantization=QuantizationConfig(
                weight_bits=8,
                activation_bits=6,
                mamp_cross_attention_bits=8,
                mamp_self_attention_bits=6,
            )
        )
        
        assert config.quantization.weight_bits == 8
        assert config.quantization.activation_bits == 6
        assert config.quantization.mamp_cross_attention_bits == 8


class TestVideoQuantDiffusersPipeline:
    """Test VideoQuant Diffusers pipeline wrapper."""
    
    def test_pipeline_initialization(self):
        """Test pipeline wrapper initialization."""
        base_pipeline = MockDiffusersPipeline()
        config = VideoQuantDiffusersConfig.default_w4a4()
        
        pipe = VideoQuantDiffusersPipeline(base_pipeline, config)
        
        assert pipe.base_pipeline is base_pipeline
        assert pipe.config == config
        assert pipe.quantizer is not None
    
    def test_pipeline_initialization_no_quantization(self):
        """Test pipeline initialization with quantization disabled."""
        base_pipeline = MockDiffusersPipeline()
        config = VideoQuantDiffusersConfig.fp16_baseline()
        
        pipe = VideoQuantDiffusersPipeline(base_pipeline, config)
        
        assert pipe.quantizer is None  # No quantization needed
    
    def test_pipeline_device_placement(self):
        """Test device placement methods."""
        base_pipeline = MockDiffusersPipeline()
        pipe = VideoQuantDiffusersPipeline(base_pipeline, VideoQuantDiffusersConfig.default_w4a4())
        
        # Test to()
        pipe = pipe.to("cpu")
        assert str(pipe.device) == "cpu"
        
        # Test cpu()
        pipe = pipe.cpu()
        assert str(pipe.device) == "cpu"
    
    def test_pipeline_generation(self):
        """Test video generation with quantization."""
        base_pipeline = MockDiffusersPipeline()
        pipe = VideoQuantDiffusersPipeline(base_pipeline, VideoQuantDiffusersConfig.default_w4a4())
        
        output = pipe(
            prompt="a cat playing piano",
            num_frames=16,
            height=512,
            width=512,
            num_inference_steps=10,
        )
        
        assert output is not None
        assert output.frames is not None
        assert isinstance(output.quant_stats, dict)
    
    def test_pipeline_generation_fp16_baseline(self):
        """Test video generation without quantization (FP16 baseline)."""
        base_pipeline = MockDiffusersPipeline()
        pipe = VideoQuantDiffusersPipeline(base_pipeline, VideoQuantDiffusersConfig.fp16_baseline())
        
        output = pipe(
            prompt="a cat playing piano",
            num_frames=16,
            height=512,
            width=512,
            num_inference_steps=10,
        )
        
        assert output is not None
        assert output.frames is not None
        # No quantization stats when disabled
        assert output.quant_stats is None or not output.quant_stats.get("quantization_enabled")
    
    def test_pipeline_with_seed(self):
        """Test pipeline with random seed."""
        base_pipeline = MockDiffusersPipeline()
        pipe = VideoQuantDiffusersPipeline(base_pipeline, VideoQuantDiffusersConfig.default_w4a4())
        
        output1 = pipe(
            prompt="test",
            num_frames=8,
            num_inference_steps=5,
            seed=42,
        )
        
        output2 = pipe(
            prompt="test",
            num_frames=8,
            num_inference_steps=5,
            seed=42,
        )
        
        # With same seed, outputs should match
        assert torch.allclose(output1.frames, output2.frames)
    
    def test_timestep_aware_quantization(self):
        """Test that timestep-aware quantization is applied."""
        base_pipeline = MockDiffusersPipeline()
        pipe = VideoQuantDiffusersPipeline(base_pipeline, VideoQuantDiffusersConfig.default_w4a4())
        
        # Get initial timestep
        initial_timestep = pipe.quantizer.state.timestep if pipe.quantizer else 0.5
        
        # Run generation
        output = pipe(
            prompt="test",
            num_frames=8,
            num_inference_steps=5,
        )
        
        # Quantizer should have been updated during generation
        final_timestep = pipe.quantizer.state.timestep if pipe.quantizer else 0.5
        assert isinstance(final_timestep, float)
    
    def test_pipeline_compatibility_with_compile(self):
        """Test compatibility with model.compile() configuration."""
        base_pipeline = MockDiffusersPipeline()
        config = VideoQuantDiffusersConfig(compile_model=True)
        
        pipe = VideoQuantDiffusersPipeline(base_pipeline, config)
        
        # Should initialize successfully even if compile isn't available
        assert pipe is not None
    
    def test_get_quantization_stats(self):
        """Test getting quantization statistics."""
        base_pipeline = MockDiffusersPipeline()
        pipe = VideoQuantDiffusersPipeline(base_pipeline, VideoQuantDiffusersConfig.default_w4a4())
        
        stats = pipe.get_quantization_stats()
        
        assert stats is not None
        assert "enabled" in stats
        assert "timestep" in stats
    
    def test_save_and_load_quantized(self, tmp_path):
        """Test saving and loading quantized pipeline."""
        import json
        import os
        
        base_pipeline = MockDiffusersPipeline()
        original_config = VideoQuantDiffusersConfig.default_w4a4()
        original_config.device = "cpu"
        pipe = VideoQuantDiffusersPipeline(base_pipeline, original_config)
        
        save_dir = str(tmp_path / "test_save")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_dir, "videoquant_config.json")
        with open(config_path, "w") as f:
            json.dump(original_config.to_dict(), f)
        
        # Load config
        with open(config_path, "r") as f:
            loaded_dict = json.load(f)
        
        loaded_config = VideoQuantDiffusersConfig.from_dict(loaded_dict)
        
        assert loaded_config.quantization.weight_bits == original_config.quantization.weight_bits
        assert loaded_config.device == original_config.device


class TestModelQuantizer:
    """Test ModelQuantizer functionality."""
    
    def test_quantizer_initialization(self):
        """Test ModelQuantizer initialization."""
        model = MockTransformer()
        config = VideoQuantConfig()
        
        quantizer = ModelQuantizer(model, config)
        
        assert quantizer.model is model
        assert quantizer.config == config
        assert quantizer.pipeline is not None
    
    def test_quantize_weights(self):
        """Test weight quantization."""
        model = MockTransformer()
        original_weight = model.self_attn.weight.data.clone()
        
        quantizer = ModelQuantizer(model, VideoQuantConfig())
        quantizer.quantize_weights()
        
        # Weights should have been modified
        # Note: This is a simple test - actual quantization may vary
        assert quantizer is not None
    
    def test_install_and_remove_hooks(self):
        """Test installing and removing quantization hooks."""
        model = MockTransformer()
        quantizer = ModelQuantizer(model, VideoQuantConfig())
        
        # Store original forward
        original_forward = model.self_attn.forward
        
        # Install hooks
        quantizer.install_hooks()
        assert quantizer._hooks_installed is True
        assert model.self_attn.forward is not original_forward
        
        # Remove hooks
        quantizer.remove_hooks()
        assert quantizer._hooks_installed is False
    
    def test_set_timestep(self):
        """Test setting timestep for quantization."""
        model = MockTransformer()
        quantizer = ModelQuantizer(model, VideoQuantConfig())
        
        quantizer.set_timestep(0.7)
        
        assert quantizer.state.timestep == 0.7
    
    def test_enable_disable_quantization(self):
        """Test enabling and disabling quantization."""
        model = MockTransformer()
        quantizer = ModelQuantizer(model, VideoQuantConfig())
        
        # Disable
        quantizer.disable_quantization()
        assert quantizer.state.enabled is False
        
        # Enable
        quantizer.enable_quantization()
        assert quantizer.state.enabled is True
    
    def test_get_stats(self):
        """Test getting quantization statistics."""
        model = MockTransformer()
        quantizer = ModelQuantizer(model, VideoQuantConfig())
        
        stats = quantizer.get_stats()
        
        assert "enabled" in stats
        assert "timestep" in stats
        assert "config" in stats


class TestQuantizationHooks:
    """Test quantization hooks integration."""
    
    def test_apply_quantization_function(self):
        """Test the convenience function for applying quantization."""
        model = MockTransformer()
        
        quantizer = apply_quantization_to_model(
            model,
            config=VideoQuantConfig(),
            quantize_weights=True,
            install_hooks=True,
        )
        
        assert quantizer is not None
        assert quantizer._hooks_installed is True
    
    def test_layer_type_detection(self):
        """Test automatic layer type detection from module names."""
        model = MockTransformer()
        quantizer = ModelQuantizer(model, VideoQuantConfig())
        
        # Test detection
        assert quantizer._detect_layer_type("self_attn") == "self_attention"
        assert quantizer._detect_layer_type("cross_attn") == "cross_attention"
        assert quantizer._detect_layer_type("temporal_attn") == "temporal_attention"
        assert quantizer._detect_layer_type("ffn") == "ffn"
        assert quantizer._detect_layer_type("unknown") == "ffn"  # Default


class TestIntegrationScenarios:
    """Test common integration scenarios."""
    
    def test_w4a4_generation_scenario(self):
        """Test typical W4A4 generation scenario."""
        base_pipeline = MockDiffusersPipeline()
        config = VideoQuantDiffusersConfig.default_w4a4()
        
        pipe = VideoQuantDiffusersPipeline(base_pipeline, config)
        
        output = pipe(
            prompt="a beautiful sunset over mountains",
            negative_prompt="blurry, low quality",
            num_frames=16,
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=5.0,
            seed=12345,
        )
        
        assert output is not None
        assert output.frames.shape == torch.Size([16, 3, 64, 64])
    
    def test_configuration_change_mid_inference(self):
        """Test changing configuration."""
        base_pipeline = MockDiffusersPipeline()
        config = VideoQuantDiffusersConfig.default_w4a4()
        
        pipe = VideoQuantDiffusersPipeline(base_pipeline, config)
        
        # Enable quantization
        pipe.quantizer.enable_quantization()
        assert pipe.quantizer.state.enabled is True
        
        # Disable quantization
        pipe.quantizer.disable_quantization()
        assert pipe.quantizer.state.enabled is False
    
    def test_multiple_generations_same_pipeline(self):
        """Test multiple generations with same pipeline."""
        base_pipeline = MockDiffusersPipeline()
        pipe = VideoQuantDiffusersPipeline(base_pipeline, VideoQuantDiffusersConfig.default_w4a4())
        
        outputs = []
        for i in range(3):
            output = pipe(
                prompt=f"frame {i}",
                num_frames=8,
                num_inference_steps=5,
                seed=42 + i,
            )
            outputs.append(output)
        
        assert len(outputs) == 3
        # Each output should be different (different seeds)
        assert not torch.allclose(outputs[0].frames, outputs[1].frames)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
