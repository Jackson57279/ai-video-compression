# VideoQuant User Testing Guide

## Validation Surface

### Primary Surface: Python API and CLI

VideoQuant is a library with two user-facing interfaces:

1. **Python API**: Direct programmatic access
2. **CLI Tools**: Command-line scripts for common operations

No web UI or TUI - validation focuses on:
- Library correctness
- Model inference quality
- Resource usage metrics

## Validation Tools

### 1. Python Test Suite

**Tool:** pytest
**Skill:** N/A (pure Python testing)

**Usage:**
```bash
pytest tests/ -v                    # All tests
pytest tests/test_tpq.py -v         # TPQ only
pytest tests/test_sqjl.py -v        # SQJL only
pytest tests/test_mamp.py -v        # MAMP only
pytest tests/test_pipeline.py -v    # Integration
pytest tests/test_metrics.py -v     # Quality metrics
```

**Expected Runtime:**
- Unit tests: < 30 seconds
- Integration tests: < 2 minutes (CPU)
- Full test suite: < 5 minutes (CPU)

**Resource Cost:** Low (CPU-bound, < 2GB RAM)

### 2. Benchmark Scripts

**Tool:** Custom Python scripts

**Available Benchmarks:**
```bash
python benchmarks/tpq_compression.py      # TPQ compression ratio
python benchmarks/memory_usage.py         # Memory profiling
python benchmarks/cpu_performance.py      # Speed benchmarks
python benchmarks/end_to_end.py            # Full pipeline
```

**Expected Runtime:**
- Compression: < 1 minute
- Memory: < 2 minutes
- Performance: < 5 minutes
- End-to-end: < 10 minutes

**Resource Cost:** Medium (up to 4GB RAM during model tests)

### 3. Quality Evaluation

**Tool:** Custom evaluation scripts

**Usage:**
```bash
python scripts/evaluate_quality.py --model wan2.1-1.3b
python scripts/compare_quality.py --baseline fp16 --quantized w4a4
```

**Expected Runtime:**
- Generate + evaluate: ~30 minutes (CPU, single video)
- Compare: ~60 minutes (CPU, baseline + quantized)

**Resource Cost:** High (8GB+ RAM during model inference)

## Validation Concurrency

**Max Concurrent Validators: 1**

**Rationale:**
- CPU-only environment (no GPU parallelization)
- Model inference is memory-intensive
- Wan2.1-1.3B uses ~8GB RAM during inference
- System has limited RAM, cannot run multiple model instances

**Resource Estimates:**
- Idle: ~500MB baseline
- Unit tests: +500MB
- Model inference: +8GB
- **Total peak:** ~9GB
- **Recommended max:** 1 validator at a time

## Test Categories

### Unit Tests (Fast)
- TPQ correctness
- SQJL distance preservation
- MAMP allocation
- Tensor packing
- Configuration loading

**Runtime:** < 30 seconds
**Automation:** Run on every feature

### Integration Tests (Medium)
- Full pipeline execution
- Diffusers integration
- Model loading
- Memory profiling

**Runtime:** < 5 minutes
**Automation:** Run at milestone completion

### Quality Tests (Slow)
- FID computation
- CLIPSIM evaluation
- Temporal consistency
- End-to-end video generation

**Runtime:** 30-120 minutes
**Automation:** Run at final validation only

## Manual Verification Checklist

For each completed feature:

- [ ] Unit tests pass
- [ ] No NaN or Inf in outputs
- [ ] Tensor shapes preserved
- [ ] Memory usage as expected
- [ ] Roundtrip accuracy > 99%

For milestone validation:

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Model loads successfully
- [ ] Video generates without errors
- [ ] Memory reduction meets target (4x)

For final validation:

- [ ] Quality metrics computed
- [ ] FID ≥ 99% of baseline
- [ ] CLIPSIM ≥ 99% of baseline
- [ ] Temporal ≥ 99% of baseline
- [ ] Visual inspection confirms quality

## Common Issues and Debugging

### Issue: Test timeout on CPU

**Solution:**
- Reduce test tensor sizes
- Use smaller frame counts (4 instead of 16)
- Skip heavy tests in quick mode: `pytest -m "not slow"`

### Issue: Out of memory

**Solution:**
- Clear PyTorch cache: `torch.cuda.empty_cache()` (if CUDA available)
- Delete intermediate tensors explicitly
- Run tests sequentially, not in parallel
- Use smaller test fixtures

### Issue: Numerical instability

**Solution:**
- Check for division by zero in scales
- Add epsilon to denominators
- Verify clamping to valid ranges
- Check for overflow in accumulations

## Environment-Specific Notes

**No GPU Available:**
- All inference on CPU (slow)
- No CUDA kernels
- PyTorch CPU-only build sufficient

**CPU Optimization:**
- Numba JIT for hot paths
- NumPy vectorization
- Avoid Python loops for large tensors

## Validation Command Reference

```bash
# Quick validation (unit tests only)
pytest tests/ -v -m "not slow"

# Full validation (including integration)
pytest tests/ -v

# Quality validation (slow, with model)
python scripts/evaluate_quality.py --full

# Memory benchmark
python scripts/benchmark_memory.py

# Generate test video
python scripts/generate_video.py \
    --model wan2.1-1.3b \
    --quantized \
    --prompt "a cat playing piano" \
    --num-frames 16
```
