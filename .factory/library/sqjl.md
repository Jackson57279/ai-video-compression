# Spatial-QJL (SQJL) Algorithm

## Overview

SQJL (Spatial-QJL) is the second stage of the VideoQuant two-stage quantization pipeline. It applies Johnson-Lindenstrauss (JL) random projection followed by sign-bit (1-bit) quantization for residual correction.

## Key Features

1. **Johnson-Lindenstrauss Projection**: Random projection that preserves pairwise distances within (1±ε) factor
2. **Sign-Bit Quantization**: 1-bit encoding (positive=1, negative=0) with zero metadata overhead
3. **Unbiased Attention Estimator**: Statistical estimator with near-zero systematic bias for attention computation
4. **Spatial Relationship Preservation**: Maintains 2D spatial topology from image/video patches

## Usage

```python
from videoquant import SQJLQuantizer, SQJLConfig

# Create quantizer with 256-dimensional projection
config = SQJLConfig(projection_dim=256)
quantizer = SQJLQuantizer(config)

# Quantize tensor
tensor = torch.randn(1000, 512)  # [num_vectors, feature_dim]
result = quantizer.quantize(tensor, output_dim=256)

# Result contains:
# - quantized_bits: 1-bit sign-quantized tensor (bool dtype)
# - metadata: global config only (zero per-element overhead)

# Dequantize
dequantized = quantizer.dequantize(result)
```

## Attention Estimation

SQJL is designed for attention score estimation with quantized keys:

```python
from videoquant import estimate_attention_with_sqjl

queries = torch.randn(batch, heads, seq_len, head_dim)
keys = torch.randn(batch, heads, seq_len, head_dim)

# Estimate attention with SQJL-quantized keys
attention_scores = estimate_attention_with_sqjl(queries, keys)
```

## Algorithm Details

### Johnson-Lindenstrauss Projection

The JL projection matrix P is created with:
- Entries drawn from N(0, 1/output_dim)
- Deterministic seeding for reproducibility
- Caching for efficiency

Distance preservation property:
```
(1-ε)||u-v||² ≤ ||Pu-Pv||² ≤ (1+ε)||u-v||²
```

For ε=0.1 and projection_dim=256, >95% of pairs should satisfy this bound.

### Sign-Bit Quantization

Each projected element is quantized to 1 bit:
- positive/zero → True (1)
- negative → False (0)

**Zero Metadata Overhead**: Unlike traditional quantization which stores scales and zero-points per tensor or per group, SQJL uses only global configuration with no per-element metadata.

### Unbiased Estimator

For attention score estimation with sign-quantized keys k_q:

```python
dequantized = sign(k) * √(π/2)
attention = dot(query, dequantized)
```

The scale factor √(π/2) is chosen so that for Gaussian-distributed inputs:
```
E[estimator(query, k_q)] ≈ dot(query, k)
```

## Memory Characteristics

| Metric | SQJL | Traditional INT8 | Group-wise INT4 |
|--------|------|------------------|-----------------|
| Bits per element | 1 | 8 | 4 |
| Metadata overhead | ~0% | ~0.0007% | ~3% |
| Compression vs FP16 | 8x-16x | 2x | 4x |

## Verification

All SQJL tests verify:
- **VAL-SQJL-001**: Distance preservation within (1±0.1) factor
- **VAL-SQJL-002**: Sign-bit uses exactly 1 bit per element, zero metadata
- **VAL-SQJL-003**: Unbiased estimator with low systematic bias (<0.05)
- **VAL-SQJL-004**: Spatial relationship preservation (positive correlation)

Run tests:
```bash
pytest tests/test_sqjl.py -v
python benchmarks/memory_overhead.py
```

## Integration

SQJL integrates with TPQ as part of the full pipeline:
1. TPQ quantizes activations with polar transform
2. SQJL adds residual correction via JL projection
3. MAMP applies layer-specific precision allocation

The combined pipeline achieves 4x+ compression with minimal quality loss.

## References

- Johnson-Lindenstrauss Lemma: https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
- TurboQuant paper (inspiration for sign-bit approach)
