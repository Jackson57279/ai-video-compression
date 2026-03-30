#!/usr/bin/env python3
"""
End-to-end benchmark for the unified VideoQuant pipeline.

Validates VAL-INT-001 and VAL-CROSS-001 by testing:
1. Full TPQ → SQJL → MAMP pipeline execution
2. Tensor shape preservation [B, F, N, C]
3. No NaN/Inf propagation
4. Dequantized tensors usable for DiT computation
"""

import torch
import numpy as np
import sys
import time

sys.path.insert(0, '/home/dih/ai-video')

from videoquant.core.pipeline import (
    VideoQuantPipeline, 
    VideoQuantConfig,
    create_default_pipeline
)
from videoquant.core.mamp import LayerType


def test_pipeline_execution():
    """Test VAL-INT-001: Full Pipeline Execution."""
    print("=" * 70)
    print("Test VAL-INT-001: Full Pipeline Execution")
    print("=" * 70)
    
    pipeline = create_default_pipeline()
    
    # Test with typical video DiT tensor shape
    torch.manual_seed(42)
    tensor = torch.randn(2, 8, 16, 64)  # [B, F, N, C]
    
    print(f"\nInput tensor shape: {tensor.shape}")
    print(f"Input tensor elements: {tensor.numel():,}")
    
    # Test each layer type
    layer_types = [
        ("cross_attention", 6),
        ("temporal_attention", 5),
        ("self_attention", 4),
        ("ffn", 3),
    ]
    
    all_passed = True
    
    for layer_type, expected_bits in layer_types:
        print(f"\n--- Testing {layer_type} (expected: {expected_bits} bits) ---")
        
        try:
            # Quantize
            start = time.time()
            result = pipeline.quantize(tensor, layer_type=layer_type, timestep=0.5)
            quant_time = (time.time() - start) * 1000
            
            # Dequantize
            start = time.time()
            reconstructed = pipeline.dequantize(result['quantized_data'])
            dequant_time = (time.time() - start) * 1000
            
            stats = result['stats']
            
            # Check shape preservation
            shape_ok = reconstructed.shape == tensor.shape
            
            # Check NaN/Inf
            has_nan = torch.isnan(reconstructed).any().item()
            has_inf = torch.isinf(reconstructed).any().item()
            finite_ok = not has_nan and not has_inf
            
            # Check MAMP bits
            bits_ok = stats.mamp_bits >= expected_bits
            
            # Check cosine similarity
            cos_sim_ok = stats.cosine_similarity > 0.5
            
            print(f"  Shape preserved: {shape_ok} ({reconstructed.shape})")
            print(f"  No NaN/Inf: {finite_ok} (NaN: {has_nan}, Inf: {has_inf})")
            print(f"  MAMP bits: {stats.mamp_bits} (expected >= {expected_bits}): {bits_ok}")
            print(f"  Cosine similarity: {stats.cosine_similarity:.4f}")
            print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
            print(f"  Quantize time: {quant_time:.2f}ms")
            print(f"  Dequantize time: {dequant_time:.2f}ms")
            
            if shape_ok and finite_ok and bits_ok and cos_sim_ok:
                print(f"  ✓ {layer_type} PASSED")
            else:
                print(f"  ✗ {layer_type} FAILED")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ {layer_type} FAILED with exception: {e}")
            all_passed = False
    
    return all_passed


def test_tensor_shape_preservation():
    """Test tensor shape preservation through all stages."""
    print("\n" + "=" * 70)
    print("Test: Tensor Shape Preservation [B, F, N, C]")
    print("=" * 70)
    
    pipeline = create_default_pipeline()
    
    test_shapes = [
        (1, 4, 8, 16),
        (2, 8, 16, 32),
        (2, 16, 32, 64),
        (4, 8, 64, 128),
    ]
    
    all_passed = True
    
    for shape in test_shapes:
        torch.manual_seed(42)
        tensor = torch.randn(*shape)
        
        result = pipeline.quantize(tensor, layer_type="self_attention", timestep=0.5)
        reconstructed = pipeline.dequantize(result['quantized_data'])
        
        shape_match = reconstructed.shape == tensor.shape
        
        status = "✓" if shape_match else "✗"
        print(f"  {status} Shape {shape}: {'OK' if shape_match else 'FAILED'}")
        
        if not shape_match:
            print(f"      Expected: {shape}, Got: {reconstructed.shape}")
            all_passed = False
    
    return all_passed


def test_no_nan_inf_propagation():
    """Test that no NaN or Inf values are produced."""
    print("\n" + "=" * 70)
    print("Test: No NaN or Inf Propagation")
    print("=" * 70)
    
    pipeline = create_default_pipeline()
    
    test_tensors = [
        ("Normal", torch.randn(2, 8, 16, 32)),
        ("Zeros", torch.zeros(2, 8, 16, 32)),
        ("Ones", torch.ones(2, 8, 16, 32)),
        ("Small", torch.randn(2, 8, 16, 32) * 1e-6),
        ("Large", torch.randn(2, 8, 16, 32) * 1e3),
    ]
    
    all_passed = True
    
    for name, tensor in test_tensors:
        result = pipeline.quantize(tensor, layer_type="self_attention", timestep=0.5)
        reconstructed = pipeline.dequantize(result['quantized_data'])
        
        has_nan = torch.isnan(reconstructed).any().item()
        has_inf = torch.isinf(reconstructed).any().item()
        is_finite = torch.isfinite(reconstructed).all().item()
        
        ok = not has_nan and not has_inf and is_finite
        status = "✓" if ok else "✗"
        
        print(f"  {status} {name}: NaN={has_nan}, Inf={has_inf}, Finite={is_finite}")
        
        if not ok:
            all_passed = False
    
    return all_passed


def test_dit_computation_usability():
    """Test that dequantized tensors are usable for DiT computation."""
    print("\n" + "=" * 70)
    print("Test: Dequantized Tensors Usable for DiT Computation")
    print("=" * 70)
    
    pipeline = create_default_pipeline()
    
    torch.manual_seed(42)
    B, F, N, C = 2, 8, 16, 64
    
    # Test attention computation
    print("\n  Testing attention computation:")
    Q = torch.randn(B, F, N, C)
    K = torch.randn(B, F, N, C)
    V = torch.randn(B, F, N, C)
    
    # Quantize and dequantize
    Q_recon, _ = pipeline.quantize_dequantize(Q, "self_attention", 0.5)
    K_recon, _ = pipeline.quantize_dequantize(K, "self_attention", 0.5)
    V_recon, _ = pipeline.quantize_dequantize(V, "self_attention", 0.5)
    
    # Reshape for attention
    Q_att = Q_recon.reshape(B * F, N, C)
    K_att = K_recon.reshape(B * F, N, C)
    V_att = V_recon.reshape(B * F, N, C)
    
    # Compute attention
    try:
        scores = torch.matmul(Q_att, K_att.transpose(-2, -1)) / np.sqrt(C)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V_att)
        
        attn_finite = torch.isfinite(output).all().item()
        attn_shape_ok = output.shape == (B * F, N, C)
        
        status = "✓" if (attn_finite and attn_shape_ok) else "✗"
        print(f"    {status} Attention output: shape={output.shape}, finite={attn_finite}")
        
        attn_ok = attn_finite and attn_shape_ok
    except Exception as e:
        print(f"    ✗ Attention computation failed: {e}")
        attn_ok = False
    
    # Test FFN computation
    print("\n  Testing FFN computation:")
    x = torch.randn(B, F, N, C)
    x_recon, _ = pipeline.quantize_dequantize(x, "ffn", 0.5)
    
    try:
        # Reshape for linear layer
        x_flat = x_recon.reshape(-1, C)
        
        # FFN: Linear -> GELU -> Linear
        hidden_dim = C * 4
        W1 = torch.randn(C, hidden_dim)
        W2 = torch.randn(hidden_dim, C)
        
        hidden = torch.matmul(x_flat, W1)
        hidden = torch.nn.functional.gelu(hidden)
        output = torch.matmul(hidden, W2)
        
        # Reshape back
        output = output.reshape(B, F, N, C)
        
        ffn_finite = torch.isfinite(output).all().item()
        ffn_shape_ok = output.shape == (B, F, N, C)
        
        status = "✓" if (ffn_finite and ffn_shape_ok) else "✗"
        print(f"    {status} FFN output: shape={output.shape}, finite={ffn_finite}")
        
        ffn_ok = ffn_finite and ffn_shape_ok
    except Exception as e:
        print(f"    ✗ FFN computation failed: {e}")
        ffn_ok = False
    
    return attn_ok and ffn_ok


def test_pipeline_stages():
    """Test individual pipeline stages and their combination."""
    print("\n" + "=" * 70)
    print("Test: Pipeline Stages (TPQ → SQJL → MAMP)")
    print("=" * 70)
    
    torch.manual_seed(42)
    tensor = torch.randn(2, 8, 16, 32)
    
    configs = [
        ("Full pipeline", VideoQuantConfig(enable_sqjl=True, enable_mamp=True)),
        ("TPQ only", VideoQuantConfig(enable_sqjl=False, enable_mamp=False)),
        ("TPQ + SQJL", VideoQuantConfig(enable_sqjl=True, enable_mamp=False)),
        ("TPQ + MAMP", VideoQuantConfig(enable_sqjl=False, enable_mamp=True)),
    ]
    
    all_passed = True
    
    for name, config in configs:
        pipeline = VideoQuantPipeline(config)
        
        try:
            result = pipeline.quantize(tensor, layer_type="self_attention", timestep=0.5)
            reconstructed = pipeline.dequantize(result['quantized_data'])
            
            shape_ok = reconstructed.shape == tensor.shape
            finite_ok = torch.isfinite(reconstructed).all().item()
            
            stats = result['stats']
            
            status = "✓" if (shape_ok and finite_ok) else "✗"
            print(f"  {status} {name}:")
            print(f"      Shape: {reconstructed.shape}, Finite: {finite_ok}")
            print(f"      CosSim: {stats.cosine_similarity:.4f}, Compression: {stats.compression_ratio:.2f}x")
            
            if not (shape_ok and finite_ok):
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ {name}: FAILED with {e}")
            all_passed = False
    
    return all_passed


def test_layer_precision_allocation():
    """Test layer-specific precision allocation."""
    print("\n" + "=" * 70)
    print("Test: Layer-Specific Precision Allocation (MAMP)")
    print("=" * 70)
    
    pipeline = create_default_pipeline()
    
    torch.manual_seed(42)
    tensor = torch.randn(2, 8, 16, 32)
    
    layer_types = [
        ("cross_attention", 6),
        ("temporal_attention", 5),
        ("self_attention", 4),
        ("ffn", 3),
    ]
    
    all_passed = True
    
    for layer_type, expected_base in layer_types:
        result = pipeline.quantize(tensor, layer_type=layer_type, timestep=0.0)
        bits = result['stats'].mamp_bits
        
        ok = bits == expected_base
        status = "✓" if ok else "✗"
        
        print(f"  {status} {layer_type}: {bits} bits (expected: {expected_base})")
        
        if not ok:
            all_passed = False
    
    # Test timestep scaling
    print("\n  Testing timestep scaling:")
    
    for t in [0.0, 0.5, 1.0]:
        result = pipeline.quantize(tensor, layer_type="self_attention", timestep=t)
        bits = result['stats'].mamp_bits
        scale = result['stats'].timestep
        
        print(f"    t={t:.1f}: {bits} bits (scale: {scale:.2f})")
    
    return all_passed


def main():
    """Run all end-to-end tests."""
    print("\n" + "=" * 70)
    print("VideoQuant End-to-End Pipeline Benchmark")
    print("=" * 70)
    print()
    
    results = {}
    
    # Run all tests
    results["pipeline_execution"] = test_pipeline_execution()
    results["shape_preservation"] = test_tensor_shape_preservation()
    results["no_nan_inf"] = test_no_nan_inf_propagation()
    results["dit_usability"] = test_dit_computation_usability()
    results["pipeline_stages"] = test_pipeline_stages()
    results["layer_precision"] = test_layer_precision_allocation()
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print()
    
    if all_passed:
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nValidation Results:")
        print("  ✓ VAL-INT-001: Full Pipeline Execution - VERIFIED")
        print("  ✓ VAL-CROSS-001: Full Pipeline Tensor Flow - VERIFIED")
        print("\nThe VideoQuant pipeline successfully integrates:")
        print("  - TPQ: Temporal-Polar Quantization")
        print("  - SQJL: Spatial-QJL Residual Correction")
        print("  - MAMP: Metric-Aware Mixed Precision")
        print("\nTensor shape [B, F, N, C] is preserved through all stages.")
        print("No NaN or Inf values are produced.")
        print("Dequantized tensors are usable for DiT computation.")
        return 0
    else:
        print("=" * 70)
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        failed = [name for name, passed in results.items() if not passed]
        print(f"\nFailed tests: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
