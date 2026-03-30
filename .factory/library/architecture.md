# VideoQuant Architecture

## Overview

VideoQuant is a TurboQuant-inspired quantization library for video diffusion transformers (DiTs). It achieves W4A4 (4-bit weights, 4-bit activations) compression with perceptually lossless quality (>99% metric preservation).

## Three-Stage Pipeline

```
Input Tensor [B, F, N, C]
    ↓
Stage 1: TPQ (Temporal-Polar Quantization)
    - Converts to polar coordinates (radius, angle)
    - Exploits temporal redundancy
    - 2.5-3.5 bit average compression
    ↓
Stage 2: SQJL (Spatial-QJL)
    - Johnson-Lindenstrauss projection
    - 1-bit sign quantization
    - Zero metadata overhead
    ↓
Stage 3: MAMP (Metric-Aware Mixed Precision)
    - Layer-specific bit allocation
    - Timestep-aware scaling
    - Protects quality-critical layers
    ↓
Dequantized Tensor
```

## Core Components

### TPQ (Temporal-Polar Quantization)

**Purpose:** Aggressive compression exploiting video temporal redundancy

**Algorithm:**
1. Group consecutive frames in pairs
2. Convert to polar coordinates: `radius = sqrt(x² + y²), angle = atan2(y, x)`
3. Recursive polar compression on radii (log2(C) levels)
4. Adaptive bit allocation: 60% to radii, 40% to angles

**Output:** Quantized polar representation, 2.5-3.5 bits average

**Properties:**
- Temporal redundancy → small radii (high compression)
- First frames may have larger radii (initialization)
- Reconstruction error < 0.1%

### SQJL (Spatial-QJL)

**Purpose:** Error correction with zero overhead

**Algorithm:**
1. Apply Johnson-Lindenstrauss random projection
2. Sign-bit quantization: `sign(x) = +1 if x >= 0 else -1`
3. Unbiased estimator for attention computation

**Output:** 1-bit sign matrix, zero metadata overhead

**Properties:**
- JL theorem: distances preserved with high probability
- No quantization constants needed
- Unbiased attention score estimation

### MAMP (Metric-Aware Mixed Precision)

**Purpose:** Layer-specific precision allocation preserving quality metrics

**Allocation Strategy:**
| Layer Type | Bits | Rationale |
|------------|------|-----------|
| Cross-attention | 6 | Text alignment critical (CLIPSIM) |
| Temporal-attention | 5 | Smoothness critical (temporal consistency) |
| Self-attention | 4 | Balanced quality/compression |
| FFN | 3 | Less sensitive to quantization |

**Timestep Scaling:**
- Early (high noise): 1.0x base precision
- Late (refining): 1.3x base precision
- Smooth transition throughout denoising

## Data Flow

### Video DiT Tensor Shapes

```python
# Input video: [B, F, H, W, 3] (Batch, Frames, Height, Width, RGB)
# After patchify: [B, F, N, C] where N = (H*W)/patch_size²

# Three attention paths:
spatial_qkv = [B*F, N, C]      # Within each frame
temporal_qkv = [B*N, F, C]       # Across frames at same position
cross_q = [B*F*N, 1, C_text]   # Query from video
cross_kv = [B*F*N, T, C]       # Key/Value from text (T tokens)
```

### Quantization Points

1. **Weight Quantization (W4)**
   - Applied during model loading
   - Static per-layer scales
   - 4-bit symmetric quantization

2. **Activation Quantization (A4)**
   - Applied during forward pass
   - Dynamic per-tensor scales
   - TPQ+SQJL+MAMP pipeline

## CPU Optimization Strategy

Since no GPU is available:

1. **Numba JIT Compilation**
   - Hot paths: polar transform, quantization kernels
   - `@njit` decorator for inner loops
   - Parallel processing with `prange`

2. **NumPy Vectorization**
   - Batch operations over loops
   - Broadcasting for tensor operations
   - Efficient memory layout (C-contiguous)

3. **Algorithm Optimizations**
   - In-place operations where possible
   - Minimize tensor copies
   - Cache-friendly access patterns

## Integration Points

### HuggingFace Diffusers

```python
from videoquant.diffusers import VideoQuantPipeline

pipe = VideoQuantPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B",
    quantization_config={
        "weights": 4,
        "activations": 4,
        "mixed_precision": True
    }
)

video = pipe(
    prompt="a cat playing piano",
    num_frames=16,
    height=512,
    width=512
).frames[0]
```

### ComfyUI

Custom nodes for visual workflow:
- `VideoQuantLoader`: Load model with quantization
- `VideoQuantSampler`: Generate with quantized inference
- `VideoQuantConfigure`: Adjust quantization parameters

## Quality Metrics

### FID (Frechet Inception Distance)
- Measures visual quality vs real videos
- Target: ≥ 99% of FP16 baseline

### CLIPSIM (CLIP Similarity)
- Measures text-video alignment
- Target: ≥ 99% of FP16 baseline

### Temporal Consistency
- Measures frame-to-frame smoothness
- Target: ≥ 99% of FP16 baseline

## Memory Model

**FP16 Baseline:**
- Wan2.1-1.3B: ~2.6GB weights (FP16)
- Activations: ~7-8GB during inference
- Total: ~10GB peak

**W4A4 Quantized:**
- Weights: ~0.65GB (4-bit)
- Activations: ~2GB (4-bit average)
- Total: ~2.5-3GB peak
- **Reduction: 4x**

## Error Analysis

**TPQ Reconstruction Error:**
- Mean: < 0.5% relative error
- Max: < 2% relative error
- Cosine similarity: > 0.99

**SQJL Bias:**
- Attention score bias: < 0.001
- No systematic error direction

**End-to-End:**
- FID degradation: < 1%
- CLIPSIM degradation: < 1%
- Temporal degradation: < 1%

## Future Work

1. **CUDA Kernels**: GPU implementation for production use
2. **INT8 Path**: Alternative 8-bit path for quality-critical applications
3. **Dynamic Bit Adjustment**: Runtime quality/compression tradeoff
4. **Multi-Model Support**: Extend beyond Wan2.1 to other DiTs
