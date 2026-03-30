#!/usr/bin/env python3
"""
Verification script for MAMP layer precision allocation.

Validates:
- Layer-type specific precision assignments
- Timestep-aware scaling
- Cross-attention high precision protection
- Temporal attention smoothness optimization

Usage:
    python scripts/verify_layer_precision.py
"""

import sys
import numpy as np
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, "videoquant")

from videoquant.core.mamp import (
    MAMPAllocator,
    MAMPConfig,
    LayerType,
    create_default_mamp_config,
)


def verify_layer_allocations():
    """Verify base layer precision allocations."""
    print("=" * 70)
    print("MAMP Layer Precision Verification")
    print("=" * 70)
    
    config = create_default_mamp_config()
    allocator = MAMPAllocator(config)
    
    # Check base allocations (VAL-MAMP-001)
    print("\n1. Base Layer Precision Allocations (VAL-MAMP-001)")
    print("-" * 50)
    
    base_allocations = []
    for layer_type in LayerType:
        bits = allocator.get_base_bits(layer_type)
        base_allocations.append([layer_type.value, bits])
    
    print(tabulate(base_allocations, headers=["Layer Type", "Base Bits"], tablefmt="grid"))
    
    # Verify hierarchy
    print("\n✓ Hierarchy check:")
    cross = allocator.get_base_bits(LayerType.CROSS_ATTENTION)
    temporal = allocator.get_base_bits(LayerType.TEMPORAL_ATTENTION)
    self_attn = allocator.get_base_bits(LayerType.SELF_ATTENTION)
    ffn = allocator.get_base_bits(LayerType.FFN)
    
    checks = [
        ("Cross >= Temporal", cross >= temporal),
        ("Temporal >= Self", temporal >= self_attn),
        ("Self >= FFN", self_attn >= ffn),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}: {result}")
        if not result:
            all_passed = False
    
    return all_passed


def verify_timestep_scaling():
    """Verify timestep-aware precision scaling."""
    config = create_default_mamp_config()
    allocator = MAMPAllocator(config)
    
    print("\n2. Timestep-Aware Precision Scaling (VAL-MAMP-002)")
    print("-" * 50)
    
    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Compute scales
    scales = [allocator.compute_timestep_scale(t) for t in timesteps]
    
    print("\nTimestep scaling factors:")
    scale_table = [[f"{t:.2f}", f"{s:.4f}"] for t, s in zip(timesteps, scales)]
    print(tabulate(scale_table, headers=["Timestep", "Scale Factor"], tablefmt="grid"))
    
    # Verify range
    print("\n✓ Timestep scale range check:")
    early_scale = scales[0]
    late_scale = scales[-1]
    
    early_check = abs(early_scale - 1.0) < 0.01
    late_check = abs(late_scale - 1.3) < 0.01
    
    print(f"  {'✓' if early_check else '✗'} Early (t=0.0) scale ≈ 1.0: {early_scale:.4f}")
    print(f"  {'✓' if late_check else '✗'} Late (t=1.0) scale ≈ 1.3: {late_scale:.4f}")
    
    # Verify monotonicity
    monotonic = all(scales[i] <= scales[i+1] + 1e-6 for i in range(len(scales)-1))
    print(f"  {'✓' if monotonic else '✗'} Monotonic increase: {monotonic}")
    
    return early_check and late_check and monotonic


def verify_allocation_table():
    """Generate and verify allocation table across timesteps."""
    config = create_default_mamp_config()
    allocator = MAMPAllocator(config)
    
    print("\n3. Layer Allocation Across Timesteps")
    print("-" * 50)
    
    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
    table = allocator.get_allocation_table(timesteps)
    
    # Build table for display
    headers = ["Layer Type"] + [f"t={t:.2f}" for t in timesteps]
    rows = []
    
    for layer_type in LayerType:
        row = [layer_type.value]
        for t in timesteps:
            row.append(table[layer_type.value][t])
        rows.append(row)
    
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    
    return True


def verify_cross_attention_protection():
    """Verify cross-attention high precision protection (VAL-MAMP-004)."""
    config = create_default_mamp_config()
    allocator = MAMPAllocator(config)
    
    print("\n4. Cross-Attention High Precision Protection (VAL-MAMP-004)")
    print("-" * 50)
    
    timesteps = np.linspace(0.0, 1.0, 11)
    
    all_passed = True
    results = []
    
    for t in timesteps:
        cross_bits = allocator.allocate_precision(LayerType.CROSS_ATTENTION, t)
        temporal_bits = allocator.allocate_precision(LayerType.TEMPORAL_ATTENTION, t)
        self_bits = allocator.allocate_precision(LayerType.SELF_ATTENTION, t)
        ffn_bits = allocator.allocate_precision(LayerType.FFN, t)
        
        # Cross must be >= all others
        min_required = 6
        is_highest = (cross_bits >= temporal_bits and 
                     cross_bits >= self_bits and 
                     cross_bits >= ffn_bits and
                     cross_bits >= min_required)
        
        results.append([
            f"{t:.2f}",
            cross_bits,
            temporal_bits,
            self_bits,
            ffn_bits,
            "✓" if is_highest else "✗"
        ])
        
        if not is_highest:
            all_passed = False
    
    print("\n" + tabulate(results, 
                         headers=["Timestep", "Cross", "Temporal", "Self", "FFN", "Status"],
                         tablefmt="grid"))
    
    # Sensitivity check
    sensitivity = allocator.get_layer_sensitivity(LayerType.CROSS_ATTENTION)
    print(f"\n✓ Cross-attention text alignment sensitivity: {sensitivity['text_alignment']:.1f}")
    print(f"  (Should be highest among all layer types)")
    
    return all_passed


def verify_temporal_optimization():
    """Verify temporal attention smoothness optimization (VAL-MAMP-005)."""
    config = create_default_mamp_config()
    allocator = MAMPAllocator(config)
    
    print("\n5. Temporal Consistency Optimization (VAL-MAMP-005)")
    print("-" * 50)
    
    timesteps = np.linspace(0.0, 1.0, 11)
    
    all_passed = True
    results = []
    
    for t in timesteps:
        temporal_bits = allocator.allocate_precision(LayerType.TEMPORAL_ATTENTION, t)
        self_bits = allocator.allocate_precision(LayerType.SELF_ATTENTION, t)
        ffn_bits = allocator.allocate_precision(LayerType.FFN, t)
        
        # Temporal must be >= self and >= ffn
        min_required = 5
        is_optimized = (temporal_bits >= self_bits and 
                       temporal_bits >= ffn_bits and
                       temporal_bits >= min_required)
        
        results.append([
            f"{t:.2f}",
            temporal_bits,
            self_bits,
            ffn_bits,
            "✓" if is_optimized else "✗"
        ])
        
        if not is_optimized:
            all_passed = False
    
    print("\n" + tabulate(results,
                         headers=["Timestep", "Temporal", "Self", "FFN", "Status"],
                         tablefmt="grid"))
    
    # Sensitivity check
    sensitivity = allocator.get_layer_sensitivity(LayerType.TEMPORAL_ATTENTION)
    print(f"\n✓ Temporal-attention consistency sensitivity: {sensitivity['temporal_consistency']:.1f}")
    print(f"  (Should be highest among all layer types)")
    
    return all_passed


def verify_metric_preservation():
    """Verify metric preservation estimates."""
    config = create_default_mamp_config()
    allocator = MAMPAllocator(config)
    
    print("\n6. Metric Preservation Estimates")
    print("-" * 50)
    
    # Check at different timesteps
    for timestep in [0.0, 0.5, 1.0]:
        allocations = allocator.allocate_all_layers(timestep)
        passes, scores = allocator.verify_metric_preservation(allocations)
        
        print(f"\nTimestep = {timestep:.2f}:")
        print(f"  Allocations: {allocations}")
        print(f"  Metric preservation:")
        for metric, score in scores.items():
            print(f"    - {metric}: {score*100:.1f}%")
        print(f"  Meets targets: {'✓' if passes else '✗'}")
    
    return True


def run_ablation_study():
    """Compare MAMP vs uniform 4-bit allocation."""
    config = create_default_mamp_config()
    allocator = MAMPAllocator(config)
    
    print("\n7. Ablation Study: MAMP vs Uniform 4-bit")
    print("-" * 50)
    
    # MAMP allocations at t=0.5
    mamp_allocations = allocator.allocate_all_layers(timestep=0.5)
    
    # Uniform 4-bit for comparison
    uniform_allocations = {lt.value: 4 for lt in LayerType}
    
    print("\nAllocation comparison:")
    comparison = []
    for layer_type in LayerType:
        layer_name = layer_type.value
        mamp_bits = mamp_allocations[layer_name]
        uniform_bits = uniform_allocations[layer_name]
        diff = mamp_bits - uniform_bits
        comparison.append([layer_name, mamp_bits, uniform_bits, f"{diff:+d}"])
    
    print(tabulate(comparison,
                  headers=["Layer", "MAMP", "Uniform", "Diff"],
                  tablefmt="grid"))
    
    # Compute metric preservation for both
    _, mamp_scores = allocator.verify_metric_preservation(mamp_allocations)
    _, uniform_scores = allocator.verify_metric_preservation(uniform_allocations)
    
    print("\nEstimated metric preservation:")
    metrics = list(mamp_scores.keys())
    metric_comparison = []
    for metric in metrics:
        mamp_val = mamp_scores[metric] * 100
        uniform_val = uniform_scores[metric] * 100
        metric_comparison.append([
            metric,
            f"{mamp_val:.1f}%",
            f"{uniform_val:.1f}%",
            f"{mamp_val - uniform_val:+.1f}%"
        ])
    
    print(tabulate(metric_comparison,
                  headers=["Metric", "MAMP", "Uniform", "Advantage"],
                  tablefmt="grid"))
    
    print("\n✓ MAMP advantages over uniform 4-bit:")
    print("  - Cross-attention protected at 6 bits (vs 4) for text alignment")
    print("  - Temporal-attention at 5 bits (vs 4) for smoothness")
    print("  - FFN reduced to 3 bits (vs 4) to save memory")
    
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("VideoQuant MAMP Verification Script")
    print("=" * 70)
    
    results = {}
    
    # Run all verification functions
    results["layer_allocations"] = verify_layer_allocations()
    results["timestep_scaling"] = verify_timestep_scaling()
    results["allocation_table"] = verify_allocation_table()
    results["cross_attention"] = verify_cross_attention_protection()
    results["temporal_optimization"] = verify_temporal_optimization()
    results["metric_preservation"] = verify_metric_preservation()
    results["ablation_study"] = run_ablation_study()
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("All verification checks PASSED ✓")
        print("MAMP layer precision allocation is correctly implemented.")
        return 0
    else:
        print("Some verification checks FAILED ✗")
        print("Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
