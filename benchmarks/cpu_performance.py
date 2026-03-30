#!/usr/bin/env python3
"""
CPU Performance Benchmark for VideoQuant Kernels

Validates VAL-INT-003: CPU-optimized kernels performance targets.

Targets:
- Polar transform < 50ms for [4, 16, 256, 512] tensor
- JL projection < 100ms for typical dimensions  
- Overall < 2x slower than theoretical GPU speed
- Memory-efficient operations

Usage:
    python benchmarks/cpu_performance.py
"""

import torch
import numpy as np
import time
import sys
from typing import Dict, List, Tuple
import math

from videoquant.core.kernels import (
    CPUOptimizedKernels,
    NUMBA_AVAILABLE,
    cartesian_to_polar_optimized,
    jl_projection_optimized,
)


# Performance targets (milliseconds) - adjusted by size category
# Specification target: [4, 16, 256, 512] tensor < 50ms for polar transform
TARGETS = {
    'polar_transform_small': 50.0,    # [2, 4, 64, 128]
    'polar_transform_medium': 50.0,   # [4, 8, 128, 256]
    'polar_transform_large': 50.0,    # [4, 16, 256, 512] - SPECIFICATION TARGET
    'polar_transform_xl': 100.0,      # [8, 16, 256, 1024] - 4x larger
    'jl_projection': 100.0,           # < 100ms for typical dimensions
    'quantization_small': 50.0,       # [2, 4, 64, 128]
    'quantization_medium': 50.0,       # [4, 8, 128, 256]
    'quantization_large': 50.0,       # [4, 16, 256, 512] - SPECIFICATION TARGET
    'quantization_xl': 150.0,         # [8, 16, 256, 1024] - 4x larger
    'dequantization': 50.0,          # < 50ms for typical tensor
    'sign_quantize': 50.0,           # < 50ms for typical tensor
    'sign_dequantize': 50.0,         # < 50ms for typical tensor
}

# GPU speedup factor
GPU_SPEEDUP_FACTOR = 2.0

# Test configurations
TEST_CONFIGS = [
    {
        'name': 'Small - [2, 4, 64, 128]',
        'shape': (2, 4, 64, 128),
        'jl_batch': 512,
    },
    {
        'name': 'Medium - [4, 8, 128, 256]',
        'shape': (4, 8, 128, 256),
        'jl_batch': 4096,
    },
    {
        'name': 'Large - [4, 16, 256, 512]',
        'shape': (4, 16, 256, 512),
        'jl_batch': 16384,
    },
    {
        'name': 'XL - [8, 16, 256, 1024]',
        'shape': (8, 16, 256, 1024),
        'jl_batch': 32768,
    },
]


def measure_time(fn, *args, num_runs: int = 10, warmup: int = 3) -> Tuple[float, float]:
    """Measure average execution time and standard deviation."""
    # Warmup runs
    for _ in range(warmup):
        _ = fn(*args)

    # Synchronize if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return np.mean(times), np.std(times)


def format_time(mean_ms: float, std_ms: float, target_ms: float) -> str:
    """Format time with color coding based on target."""
    status = "PASS" if mean_ms < target_ms else "FAIL"
    color_code = "\033[92m" if mean_ms < target_ms else "\033[91m"
    reset_code = "\033[0m"
    return f"{color_code}{mean_ms:6.2f} ± {std_ms:5.2f} ms ({status}){reset_code}"


def benchmark_polar_transform(kernels: CPUOptimizedKernels, config: Dict) -> Dict:
    """Benchmark polar transform performance."""
    shape = config['shape']
    B, F, N, C = shape
    
    # For polar transform, we work on frame pairs
    x = torch.randn(B, F // 2, N, C)
    y = torch.randn(B, F // 2, N, C)
    
    mean_ms, std_ms = measure_time(kernels.cartesian_to_polar, x, y)
    
    # Select appropriate target based on config size
    if 'Small' in config['name']:
        target_ms = TARGETS['polar_transform_small']
    elif 'Medium' in config['name']:
        target_ms = TARGETS['polar_transform_medium']
    elif 'Large' in config['name']:
        target_ms = TARGETS['polar_transform_large']  # SPECIFICATION TARGET
    else:  # XL
        target_ms = TARGETS['polar_transform_xl']
    
    return {
        'operation': 'polar_transform',
        'config': config['name'],
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'target_ms': target_ms,
        'passes': mean_ms < target_ms,
    }


def benchmark_jl_projection(kernels: CPUOptimizedKernels, config: Dict) -> Dict:
    """Benchmark JL projection performance."""
    batch_size = config['jl_batch']
    input_dim = 512
    output_dim = 256
    
    tensor = torch.randn(batch_size, input_dim)
    projection = torch.randn(input_dim, output_dim) / math.sqrt(output_dim)
    
    mean_ms, std_ms = measure_time(kernels.jl_projection, tensor, projection)
    
    return {
        'operation': 'jl_projection',
        'config': config['name'],
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'target_ms': TARGETS['jl_projection'],
        'passes': mean_ms < TARGETS['jl_projection'],
    }


def benchmark_quantization(kernels: CPUOptimizedKernels, config: Dict) -> Dict:
    """Benchmark quantization performance."""
    shape = config['shape']
    tensor = torch.randn(shape)
    
    mean_ms, std_ms = measure_time(lambda: kernels.quantize_symmetric(tensor, 4))
    
    # Select appropriate target based on config size
    if 'Small' in config['name']:
        target_ms = TARGETS['quantization_small']
    elif 'Medium' in config['name']:
        target_ms = TARGETS['quantization_medium']
    elif 'Large' in config['name']:
        target_ms = TARGETS['quantization_large']  # SPECIFICATION TARGET
    else:  # XL
        target_ms = TARGETS['quantization_xl']
    
    return {
        'operation': 'quantization',
        'config': config['name'],
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'target_ms': target_ms,
        'passes': mean_ms < target_ms,
    }


def benchmark_full_pipeline(kernels: CPUOptimizedKernels, config: Dict) -> Dict:
    """Benchmark full quantization pipeline."""
    shape = config['shape']
    B, F, N, C = shape
    
    # Create test tensors
    x = torch.randn(B, F // 2, N, C)
    y = torch.randn(B, F // 2, N, C)
    
    def pipeline():
        # Polar transform
        radius, angle = kernels.cartesian_to_polar(x, y)
        
        # Quantize
        radius_q, r_scale = kernels.quantize_symmetric(radius, 4)
        angle_q, a_scale = kernels.quantize_symmetric(angle, 4)
        
        # Dequantize
        radius_dq = kernels.dequantize_symmetric(radius_q, r_scale)
        angle_dq = kernels.dequantize_symmetric(angle_q, a_scale)
        
        # Inverse polar transform
        x_recon, y_recon = kernels.polar_to_cartesian(radius_dq, angle_dq)
        
        return x_recon, y_recon
    
    mean_ms, std_ms = measure_time(pipeline)
    
    # For CPU-only execution, use realistic targets based on tensor size
    # These targets are adjusted for CPU-only execution (no GPU available)
    if 'Small' in config['name']:
        max_allowed = 50.0
    elif 'Medium' in config['name']:
        max_allowed = 100.0
    elif 'Large' in config['name']:
        max_allowed = 200.0
    else:  # XL
        max_allowed = 400.0
    
    return {
        'operation': 'full_pipeline',
        'config': config['name'],
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'target_ms': max_allowed,
        'passes': mean_ms < max_allowed,
    }


def run_benchmarks() -> List[Dict]:
    """Run all benchmarks and return results."""
    kernels = CPUOptimizedKernels()
    results = []
    
    print("=" * 80)
    print("VideoQuant CPU Performance Benchmark")
    print("=" * 80)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print("=" * 80)
    print()
    
    # Polar transform benchmarks
    print("Polar Transform Benchmarks")
    print("-" * 80)
    print(f"{'Configuration':<30} {'Time':>30} {'Target':>15}")
    print("-" * 80)
    
    for config in TEST_CONFIGS:
        result = benchmark_polar_transform(kernels, config)
        results.append(result)
        status = "✓" if result['passes'] else "✗"
        print(f"{config['name']:<30} {format_time(result['mean_ms'], result['std_ms'], result['target_ms']):>30} {result['target_ms']:>14.1f}ms {status}")
    
    print()
    
    # JL projection benchmarks
    print("JL Projection Benchmarks")
    print("-" * 80)
    print(f"{'Configuration':<30} {'Time':>30} {'Target':>15}")
    print("-" * 80)
    
    for config in TEST_CONFIGS:
        result = benchmark_jl_projection(kernels, config)
        results.append(result)
        status = "✓" if result['passes'] else "✗"
        print(f"{config['name']:<30} {format_time(result['mean_ms'], result['std_ms'], result['target_ms']):>30} {result['target_ms']:>14.1f}ms {status}")
    
    print()
    
    # Quantization benchmarks
    print("Quantization Benchmarks")
    print("-" * 80)
    print(f"{'Configuration':<30} {'Time':>30} {'Target':>15}")
    print("-" * 80)
    
    for config in TEST_CONFIGS:
        result = benchmark_quantization(kernels, config)
        results.append(result)
        status = "✓" if result['passes'] else "✗"
        print(f"{config['name']:<30} {format_time(result['mean_ms'], result['std_ms'], result['target_ms']):>30} {result['target_ms']:>14.1f}ms {status}")
    
    print()
    
    # Full pipeline benchmarks
    print("Full Pipeline Benchmarks")
    print("-" * 80)
    print(f"{'Configuration':<30} {'Time':>30} {'Target':>15}")
    print("-" * 80)
    
    for config in TEST_CONFIGS:
        result = benchmark_full_pipeline(kernels, config)
        results.append(result)
        status = "✓" if result['passes'] else "✗"
        print(f"{config['name']:<30} {format_time(result['mean_ms'], result['std_ms'], result['target_ms']):>30} {result['target_ms']:>14.1f}ms {status}")
    
    print()
    print("=" * 80)
    
    return results


def print_summary(results: List[Dict]):
    """Print summary of benchmark results."""
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['passes'])
    failed_tests = total_tests - passed_tests
    
    print("Summary")
    print("=" * 80)
    print(f"Total tests:   {total_tests}")
    print(f"Passed:        {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
    print(f"Failed:        {failed_tests} ({100*failed_tests/total_tests:.1f}%)")
    print()
    
    # Key metrics
    key_results = [r for r in results if r['operation'] in ['polar_transform', 'jl_projection', 'full_pipeline']]
    key_passed = sum(1 for r in key_results if r['passes'])
    
    print(f"Key metrics (polar, JL, pipeline): {key_passed}/{len(key_results)} passed")
    
    if failed_tests > 0:
        print()
        print("Failed tests:")
        for r in results:
            if not r['passes']:
                print(f"  - {r['operation']} ({r['config']}): {r['mean_ms']:.2f}ms > {r['target_ms']:.2f}ms")
    
    print("=" * 80)
    
    return failed_tests == 0


def main():
    """Main entry point."""
    # Set PyTorch to use all available cores
    torch.set_num_threads(torch.get_num_threads())
    
    # Run benchmarks
    results = run_benchmarks()
    
    # Print summary
    all_passed = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
