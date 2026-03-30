---
name: core-algorithm-worker
description: Implements core VideoQuant quantization algorithms (TPQ, SQJL, MAMP) with CPU-optimized kernels
---

# Core Algorithm Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

This worker handles implementation of VideoQuant's core quantization algorithms:
- Temporal-Polar Quantization (TPQ)
- Spatial-QJL (SQJL) 
- Metric-Aware Mixed Precision (MAMP)
- CPU-optimized kernels (SIMD/AVX2)

Features using this skill:
- TPQ polar transform implementation
- SQJL projection and sign-bit quantization
- MAMP layer-specific precision allocation
- Quantization/dequantization kernels
- Tensor packing/unpacking utilities

## Required Skills

None - pure Python/NumPy/PyTorch implementation.

## Work Procedure

### Phase 1: Test-Driven Development Setup

1. **Write failing tests first (RED)**
   - Create test file with comprehensive test cases
   - Tests must fail before implementation
   - Cover: correctness, edge cases, numerical accuracy, roundtrip

2. **Define tensor shapes and dtypes**
   - Video DiT tensors: [B, F, N, C] (Batch, Frames, Patches, Channels)
   - FP16 for activations, INT4/INT8 for quantized
   - Test with synthetic tensors matching video dimensions

### Phase 2: Core Algorithm Implementation

3. **Implement TPQ (Temporal-Polar Quantization)**
   - cartesian_to_polar(): Convert to radius + angle
   - recursive_polar_transform(): Log2(C) levels
   - adaptive_bit_allocation(): 60/40 radii/angle split
   - pack_quantized(): Bit-level tensor packing
   - Must pass VAL-TPQ-001 through VAL-TPQ-005

4. **Implement SQJL (Spatial-QJL)**
   - create_jl_projections(): Random projection matrices
   - spatial_jl_transform(): Apply projections preserving 2D
   - sign_quantize(): 1-bit sign encoding
   - unbiased_estimator(): Special estimator for attention
   - Must pass VAL-SQJL-001 through VAL-SQJL-004

5. **Implement MAMP (Metric-Aware Mixed Precision)**
   - Layer sensitivity configuration
   - Timestep-aware allocation
   - Metric preservation logic
   - Must pass VAL-MAMP-001 through VAL-MAMP-005

### Phase 3: CPU Optimization

6. **Implement CPU-optimized kernels**
   - Use NumPy vectorization
   - Consider Numba for JIT compilation
   - AVX2/SIMD where applicable (via PyTorch)
   - Profile and optimize hot paths

### Phase 4: Verification

7. **Run tests to verify (GREEN)**
   - All tests must pass
   - Roundtrip accuracy > 99%
   - No NaN/Inf in outputs
   - Performance benchmarks

8. **Manual verification**
   - Test on synthetic video tensors
   - Verify numerical stability
   - Check memory efficiency
   - Profile with cProfile

### Phase 5: Integration Prep

9. **Export clean interfaces**
   - Document tensor shapes expected
   - Provide configuration schemas
   - Add type hints
   - Create examples

## Example Handoff

```json
{
  "salientSummary": "Implemented TPQ with polar transform achieving 99.2% roundtrip accuracy and 2.7x effective compression. SQJL sign-bit quantization verified with zero memory overhead. MAMP layer profiles configured for DiT attention types.",
  "whatWasImplemented": "Three-stage quantization pipeline: (1) TPQ with recursive polar compression, (2) SQJL with JL projection and 1-bit sign encoding, (3) MAMP with layer-specific precision allocation. CPU-optimized kernels using Numba JIT.",
  "whatWasLeftUndone": "Integration with Diffusers pipeline pending. CUDA kernels not implemented (no GPU).",
  "verification": {
    "commandsRun": [
      {"command": "python -m pytest tests/test_tpq.py -v", "exitCode": 0, "observation": "5 tests passed, roundtrip accuracy 99.2%"},
      {"command": "python -m pytest tests/test_sqjl.py -v", "exitCode": 0, "observation": "4 tests passed, distance preservation 0.98"},
      {"command": "python -m pytest tests/test_mamp.py -v", "exitCode": 0, "observation": "5 tests passed"},
      {"command": "python benchmarks/compression_ratio.py", "exitCode": 0, "observation": "2.7x compression achieved at 3.5-bit average"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {"file": "tests/test_tpq.py", "cases": [
        {"name": "test_polar_transform_accuracy", "verifies": "VAL-TPQ-001"},
        {"name": "test_recursive_compression", "verifies": "VAL-TPQ-002"},
        {"name": "test_bit_allocation", "verifies": "VAL-TPQ-003"},
        {"name": "test_temporal_redundancy", "verifies": "VAL-TPQ-004"},
        {"name": "test_roundtrip_accuracy", "verifies": "VAL-TPQ-005"}
      ]},
      {"file": "tests/test_sqjl.py", "cases": [
        {"name": "test_jl_distance_preservation", "verifies": "VAL-SQJL-001"},
        {"name": "test_sign_bit_overhead", "verifies": "VAL-SQJL-002"},
        {"name": "test_unbiased_estimator", "verifies": "VAL-SQJL-003"},
        {"name": "test_spatial_preservation", "verifies": "VAL-SQJL-004"}
      ]},
      {"file": "tests/test_mamp.py", "cases": [
        {"name": "test_layer_precision", "verifies": "VAL-MAMP-001"},
        {"name": "test_timestep_allocation", "verifies": "VAL-MAMP-002"}
      ]}
    ]
  },
  "discoveredIssues": [
    {
      "severity": "low",
      "description": "Recursive polar transform O(N log N) - could optimize with iterative approach",
      "suggestedFix": "Consider GPU/CUDA for production"
    }
  ]
}
```

## When to Return to Orchestrator

- Numerical instability discovered (diverging errors, NaN)
- Algorithm correctness cannot be verified
- Performance is unacceptable (>10x slower than target)
- Tensor shapes don't match DiT requirements
