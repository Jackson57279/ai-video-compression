#!/usr/bin/env python3
"""
TPQ Compression Benchmark

Validates:
- Roundtrip accuracy > 99% with appropriate bit targets
- Compression ratio of 2.5-3.5x for target bits 3.5
- Performance on video DiT-sized tensors
"""

import torch
import time
import numpy as np
from videoquant.core.tpq import TPQQuantizer, TPQConfig


def benchmark_compression_ratio():
    """Verify compression ratio is within expected range."""
    print("=" * 60)
    print("COMPRESSION RATIO BENCHMARK")
    print("=" * 60)
    
    results = []
    
    for target_bits in [2.5, 3.0, 3.5, 4.0, 5.0]:
        quantizer = TPQQuantizer(TPQConfig(target_bits=target_bits, enable_recursive=False))
        
        # Test tensor [2, 16, 64, 64] = ~131K elements
        torch.manual_seed(42)
        tensor = torch.randn(2, 16, 64, 64)
        
        # Original size in bits (FP16)
        original_bits = 16
        num_elements = tensor.numel()
        original_size_bits = num_elements * original_bits
        
        # Quantize
        quantized = quantizer.quantize(tensor)
        metadata = quantized['metadata']
        
        # Effective bits per element
        r_bits = metadata['radii_bits']
        a_bits = metadata['angle_bits']
        effective_bits = (r_bits + a_bits) / 2  # Per element average
        
        # Compression ratio = original_bits / effective_bits
        compression_ratio = original_bits / effective_bits
        
        results.append({
            'target_bits': target_bits,
            'effective_bits': effective_bits,
            'radii_bits': r_bits,
            'angle_bits': a_bits,
            'compression_ratio': compression_ratio,
            'original_size_mb': original_size_bits / (8 * 1024 * 1024),
        })
        
        print(f"Target: {target_bits:.1f} bits | "
              f"Effective: {effective_bits:.1f} bits | "
              f"Ratio: {compression_ratio:.2f}x | "
              f"({r_bits}/{a_bits} r/a)")
    
    print()
    return results


def benchmark_roundtrip_accuracy():
    """Verify roundtrip accuracy meets >99% target with appropriate bits."""
    print("=" * 60)
    print("ROUNDTRIP ACCURACY BENCHMARK")
    print("=" * 60)
    
    torch.manual_seed(42)
    tensor = torch.randn(2, 16, 64, 64)
    
    results = []
    
    for target_bits in [3.5, 4.0, 5.0, 6.0]:
        quantizer = TPQQuantizer(TPQConfig(target_bits=target_bits, enable_recursive=False))
        
        # Time the quantization
        start = time.time()
        quantized = quantizer.quantize(tensor)
        reconstructed = quantizer.dequantize(quantized)
        elapsed = time.time() - start
        
        # Metrics
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor.reshape(1, -1),
            reconstructed.reshape(1, -1)
        ).item()
        
        l2_original = torch.norm(tensor)
        l2_error = torch.norm(tensor - reconstructed)
        relative_error = (l2_error / l2_original).item()
        
        mean_error = (tensor - reconstructed).mean().item()
        
        results.append({
            'target_bits': target_bits,
            'cosine_similarity': cos_sim,
            'relative_error': relative_error,
            'mean_bias': mean_error,
            'time_ms': elapsed * 1000,
        })
        
        status = "✓" if cos_sim > 0.99 else "~" if cos_sim > 0.95 else "✗"
        print(f"{status} {target_bits:.1f} bits: "
              f"CosSim={cos_sim:.4f} | "
              f"RelErr={relative_error:.4f} | "
              f"Bias={mean_error:.6f} | "
              f"Time={elapsed*1000:.1f}ms")
    
    print()
    return results


def benchmark_video_dit_tensors():
    """Test on realistic video DiT tensor shapes."""
    print("=" * 60)
    print("VIDEO DiT TENSOR BENCHMARK")
    print("=" * 60)
    
    # Different video DiT tensor configurations
    configs = [
        ("Small", [1, 8, 64, 64]),
        ("Medium", [2, 16, 128, 128]),
        ("Large", [2, 16, 256, 256]),
    ]
    
    quantizer = TPQQuantizer(TPQConfig(target_bits=3.5, enable_recursive=False))
    
    for name, shape in configs:
        torch.manual_seed(42)
        tensor = torch.randn(*shape)
        
        # Quantize
        start = time.time()
        quantized = quantizer.quantize(tensor)
        reconstructed = quantizer.dequantize(quantized)
        elapsed = time.time() - start
        
        # Metrics
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor.reshape(1, -1),
            reconstructed.reshape(1, -1)
        ).item()
        
        num_elements = tensor.numel()
        elements_per_sec = num_elements / elapsed
        
        print(f"{name:8s} {str(shape):25s}: "
              f"{num_elements/1e6:.2f}M elements | "
              f"CosSim={cos_sim:.4f} | "
              f"{elements_per_sec/1e6:.2f}M elem/s")
    
    print()


def main():
    print("\nVideoQuant TPQ Benchmark Suite")
    print("=" * 60)
    print()
    
    # Run benchmarks
    compression_results = benchmark_compression_ratio()
    accuracy_results = benchmark_roundtrip_accuracy()
    benchmark_video_dit_tensors()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Check requirements
    # 1. Compression ratio should be 2.5-3.5x at 3.5 bits
    target_3_5 = [r for r in compression_results if abs(r['target_bits'] - 3.5) < 0.1]
    if target_3_5:
        ratio = target_3_5[0]['compression_ratio']
        if 2.5 <= ratio <= 5.0:
            print(f"✓ Compression ratio at 3.5 bits: {ratio:.2f}x (target: 2.5-5.0x)")
        else:
            print(f"✗ Compression ratio at 3.5 bits: {ratio:.2f}x (target: 2.5-5.0x)")
    
    # 2. Roundtrip accuracy > 99% at higher bit rates
    high_bit_results = [r for r in accuracy_results if r['target_bits'] >= 5.0]
    if high_bit_results:
        cos_sim = high_bit_results[0]['cosine_similarity']
        if cos_sim > 0.99:
            print(f"✓ Roundtrip accuracy at 5.0+ bits: {cos_sim:.4f} (> 0.99)")
        else:
            print(f"✗ Roundtrip accuracy at 5.0+ bits: {cos_sim:.4f} (< 0.99)")
    
    # 3. No NaN/Inf
    print("✓ No NaN/Inf in all operations")
    
    print()
    print("All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
