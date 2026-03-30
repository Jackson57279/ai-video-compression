#!/usr/bin/env python3
"""Test script for VideoQuant Diffusers integration.

This script provides manual testing of the VideoQuantDiffusersPipeline
with various configurations and scenarios.

Usage:
    python scripts/test_pipeline.py --help
    python scripts/test_pipeline.py --test-mock
    python scripts/test_pipeline.py --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "a cat playing piano"
"""

import argparse
import sys
import os
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_config(config, prefix: str = "") -> None:
    """Print configuration in a formatted way."""
    from dataclasses import asdict
    
    if hasattr(config, "__dataclass_fields__"):
        config_dict = asdict(config)
    elif hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    else:
        config_dict = dict(config)
    
    for key, value in config_dict.items():
        if prefix:
            full_key = f"{prefix}.{key}"
        else:
            full_key = key
        
        if isinstance(value, dict):
            print(f"  {full_key}:")
            for sub_key, sub_value in value.items():
                print(f"    - {sub_key}: {sub_value}")
        elif hasattr(value, "__dataclass_fields__"):
            print(f"  {full_key}:")
            print_config(value, prefix=full_key)
        else:
            print(f"  {full_key}: {value}")


def test_mock_pipeline() -> bool:
    """Test with mock pipeline (no external dependencies beyond diffusers)."""
    print_section("Testing VideoQuant Diffusers Integration with Mock Pipeline")
    
    try:
        from videoquant.integration import (
            VideoQuantDiffusersConfig,
            QuantizationConfig,
            VideoQuantDiffusersPipeline,
        )
        from videoquant.core.pipeline import VideoQuantConfig
        print("✓ VideoQuant integration imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Create mock components
    print("\nCreating mock pipeline components...")
    
    class MockTransformer(nn.Module):
        def __init__(self, dim: int = 512):
            super().__init__()
            self.dim = dim
            self.self_attn = nn.Linear(dim, dim)
            self.cross_attn = nn.Linear(dim, dim)
            self.temporal_attn = nn.Linear(dim, dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
            self._device = torch.device('cpu')
            self._dtype = torch.float32
        
        def forward(self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
            x = x + self.self_attn(x)
            x = x + self.cross_attn(x)
            x = x + self.temporal_attn(x)
            x = x + self.ffn(x)
            return x
        
        def to(self, device):
            self._device = device
            return super().to(device)
    
    class MockPipeline:
        def __init__(self):
            self.transformer = MockTransformer()
            self._device = torch.device('cpu')
            self._dtype = torch.float32
        
        @property
        def device(self):
            return self._device
        
        @property
        def dtype(self):
            return self._dtype
        
        def to(self, device):
            self._device = device if isinstance(device, torch.device) else torch.device(device)
            self.transformer = self.transformer.to(self._device)
            return self
        
        def __call__(self, prompt: str = "", **kwargs):
            frames = torch.randn(16, 3, 64, 64)
            return type('obj', (object,), {'frames': [frames]})()
    
    print("✓ Mock components created")
    
    # Test W4A4 configuration
    print("\nTesting W4A4 configuration...")
    config = VideoQuantDiffusersConfig.default_w4a4()
    print_config(config)
    
    print("\nCreating pipeline wrapper...")
    base_pipeline = MockPipeline()
    pipe = VideoQuantDiffusersPipeline(base_pipeline, config)
    print("✓ Pipeline wrapper created")
    
    print(f"  Device: {pipe.device}")
    print(f"  Dtype: {pipe.dtype}")
    print(f"  Quantizer enabled: {pipe.quantizer is not None}")
    
    # Test device placement
    print("\nTesting device placement...")
    pipe = pipe.to("cpu")
    print(f"✓ Pipeline moved to CPU: {pipe.device}")
    
    # Test generation
    print("\nTesting generation...")
    start_time = time.time()
    output = pipe(
        prompt="a cat playing piano",
        num_frames=16,
        height=512,
        width=512,
        num_inference_steps=10,
        seed=42,
    )
    generation_time = time.time() - start_time
    
    print(f"✓ Generation completed in {generation_time:.2f}s")
    print(f"  Output shape: {output.frames.shape}")
    print(f"  Quantization stats available: {output.quant_stats is not None}")
    
    if output.quant_stats:
        print(f"    Steps: {output.quant_stats.get('steps', 'N/A')}")
        print(f"    Quantization enabled: {output.quant_stats.get('quantization_enabled', 'N/A')}")
    
    # Test FP16 baseline (no quantization)
    print("\nTesting FP16 baseline (no quantization)...")
    config_baseline = VideoQuantDiffusersConfig.fp16_baseline()
    pipe_baseline = VideoQuantDiffusersPipeline(MockPipeline(), config_baseline)
    
    output_baseline = pipe_baseline(
        prompt="a cat playing piano",
        num_frames=16,
        num_inference_steps=10,
        seed=42,
    )
    print("✓ FP16 baseline generation completed")
    print(f"  Quantization stats: {output_baseline.quant_stats}")
    
    # Test get_quantization_stats
    print("\nTesting get_quantization_stats...")
    stats = pipe.get_quantization_stats()
    if stats:
        print(f"  Enabled: {stats.get('enabled')}")
        print(f"  Timestep: {stats.get('timestep')}")
        print(f"  Layer counts: {stats.get('layer_counts', {})}")
    else:
        print("  No quantization stats available")
    
    print_section("Mock Pipeline Tests Complete")
    return True


def test_real_model(model_id: str, prompt: str, num_frames: int = 16, device: str = "cpu") -> bool:
    """Test with real Diffusers model (requires model download)."""
    print_section(f"Testing with Real Model: {model_id}")
    
    try:
        from diffusers import DiffusionPipeline
        print("✓ Diffusers imported successfully")
    except ImportError:
        print("✗ diffusers not installed. Install with: pip install diffusers")
        return False
    
    try:
        from videoquant.integration import (
            VideoQuantDiffusersConfig,
            VideoQuantDiffusersPipeline,
        )
        print("✓ VideoQuant integration imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VideoQuant integration: {e}")
        return False
    
    print(f"\nLoading model from HuggingFace: {model_id}")
    print(f"  Device: {device}")
    print(f"  This may take a few minutes for first-time download...")
    
    try:
        # Create configuration
        config = VideoQuantDiffusersConfig.default_w4a4()
        config.device = device
        
        # Load pipeline
        start_time = time.time()
        pipe = VideoQuantDiffusersPipeline.from_pretrained(
            model_id,
            videoquant_config=config,
            torch_dtype=torch.float16 if device == "cpu" else torch.float16,
        )
        load_time = time.time() - start_time
        
        print(f"✓ Pipeline loaded in {load_time:.2f}s")
        print(f"  Device: {pipe.device}")
        print(f"  Quantizer: {'enabled' if pipe.quantizer else 'disabled'}")
        
        # Run generation
        print(f"\nGenerating video with prompt: '{prompt}'")
        print(f"  Num frames: {num_frames}")
        print(f"  This may take several minutes on CPU...")
        
        start_time = time.time()
        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=512,
            width=512,
            num_inference_steps=50,
            seed=42,
        )
        generation_time = time.time() - start_time
        
        print(f"✓ Generation completed in {generation_time:.2f}s")
        print(f"  Output shape: {output.frames.shape}")
        
        if output.quant_stats:
            print("\nQuantization statistics:")
            for key, value in output.quant_stats.items():
                print(f"  {key}: {value}")
        
        print_section("Real Model Test Complete")
        return True
        
    except Exception as e:
        print(f"✗ Error during model loading or generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility_features() -> bool:
    """Test compatibility with Diffusers features like compile, offload, etc."""
    print_section("Testing Diffusers Compatibility Features")
    
    try:
        from videoquant.integration import (
            VideoQuantDiffusersConfig,
            QuantizationConfig,
            VideoQuantDiffusersPipeline,
        )
        from videoquant.integration.quantization_hooks import ModelQuantizer, apply_quantization_to_model
        from videoquant.core.pipeline import VideoQuantConfig
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test configuration options
    print("\nTesting configuration options...")
    
    # Test compile_model flag
    config_compile = VideoQuantDiffusersConfig(compile_model=True)
    print(f"✓ compile_model config: {config_compile.compile_model}")
    
    # Test memory_efficient flag
    config_memory = VideoQuantDiffusersConfig(memory_efficient=True)
    print(f"✓ memory_efficient config: {config_memory.memory_efficient}")
    
    # Test cpu_offload flag
    config_offload = VideoQuantDiffusersConfig(cpu_offload=True)
    print(f"✓ cpu_offload config: {config_offload.cpu_offload}")
    
    # Test custom quantization bits
    config_custom = VideoQuantDiffusersConfig(
        quantization=QuantizationConfig(
            weight_bits=8,
            activation_bits=6,
            mamp_cross_attention_bits=8,
            mamp_temporal_attention_bits=6,
            mamp_self_attention_bits=6,
            mamp_ffn_bits=4,
        )
    )
    print("\nCustom 8-bit configuration:")
    print(f"  Weight bits: {config_custom.quantization.weight_bits}")
    print(f"  Activation bits: {config_custom.quantization.activation_bits}")
    print(f"  Cross-attention bits: {config_custom.quantization.mamp_cross_attention_bits}")
    
    # Test serialization
    print("\nTesting configuration serialization...")
    config_dict = config_custom.to_dict()
    config_restored = VideoQuantDiffusersConfig.from_dict(config_dict)
    
    assert config_restored.quantization.weight_bits == config_custom.quantization.weight_bits
    assert config_restored.quantization.enable_tpq == config_custom.quantization.enable_tpq
    print("✓ Config serialization/deserialization works")
    
    print_section("Compatibility Features Tests Complete")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test VideoQuant Diffusers Integration"
    )
    parser.add_argument(
        "--test-mock",
        action="store_true",
        help="Run tests with mock pipeline (no model download required)",
    )
    parser.add_argument(
        "--test-compatibility",
        action="store_true",
        help="Test compatibility features",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Model ID to test with (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cat playing piano",
        help="Prompt for video generation",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests",
    )
    
    args = parser.parse_args()
    
    if not any([args.test_mock, args.test_compatibility, args.all]) and not args.model:
        parser.print_help()
        return 0
    
    results = []
    
    # Run mock tests
    if args.test_mock or args.all:
        results.append(("Mock Pipeline", test_mock_pipeline()))
    
    # Run compatibility tests
    if args.test_compatibility or args.all:
        results.append(("Compatibility Features", test_compatibility_features()))
    
    # Run real model test
    if args.model and not (args.test_mock or args.test_compatibility):
        results.append(("Real Model", test_real_model(
            args.model,
            args.prompt,
            args.num_frames,
            args.device,
        )))
    elif args.all:
        # Skip real model test in --all mode to avoid long downloads
        print("\nNote: Skipping real model test in --all mode. Use --model to test with real model.")
    
    # Summary
    print_section("Test Summary")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
