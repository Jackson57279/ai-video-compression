---
name: quality-metrics-worker
description: Implements and validates video quality metrics (FID, CLIPSIM, temporal consistency) for VideoQuant
---

# Quality Metrics Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

This worker implements quality evaluation for VideoQuant:
- FID (Frechet Inception Distance) computation
- CLIPSIM (CLIP similarity) for text-video alignment
- Temporal consistency metrics
- Benchmarking suite

Features using this skill:
- Metrics computation implementation
- Baseline vs quantized comparison
- Quality regression detection
- Evaluation dataset preparation

## Required Skills

None. Uses PyTorch, transformers, and custom metrics.

## Work Procedure

### Phase 1: Metrics Implementation

1. **Implement FID computation**
   - Load InceptionV3 or similar feature extractor
   - Compute statistics for generated videos
   - Calculate Frechet distance
   - Cache reference statistics

2. **Implement CLIPSIM**
   - Load CLIP model
   - Compute text-video similarity
   - Aggregate across frames
   - Handle video encoding

3. **Implement temporal consistency**
   - Frame difference analysis
   - Optical flow consistency
   - Motion smoothness metrics
   - Flickering detection

### Phase 2: Evaluation Pipeline

4. **Create evaluation harness**
   - Generate videos with FP16 baseline
   - Generate videos with quantization
   - Compute metrics for both
   - Compare and report

5. **Prepare test datasets**
   - Text prompts for evaluation
   - Reference videos if available
   - Standardized generation parameters

### Phase 3: Validation

6. **Verify metric preservation**
   - Run evaluation on test set
   - Verify ≥ 99% metric preservation
   - Document any degradation

7. **Statistical analysis**
   - Compute confidence intervals
   - Test statistical significance
   - Generate comparison plots

## Example Handoff

```json
{
  "salientSummary": "Implemented FID, CLIPSIM, and temporal consistency metrics. Evaluated Wan2.1-1.3B with W4A4 quantization: FID 99.1% preserved, CLIPSIM 99.3%, temporal 98.7%. All metrics exceed 99% target.",
  "whatWasImplemented": "Video quality metrics suite with FID using InceptionV3 features, CLIPSIM with CLIP ViT-L/14, temporal consistency via frame difference and optical flow. Benchmark script comparing FP16 vs W4A4.",
  "whatWasLeftUndone": "Large-scale evaluation on VBench dataset not completed (time constraints).",
  "verification": {
    "commandsRun": [
      {"command": "python -m pytest tests/test_metrics.py -v", "exitCode": 0, "observation": "All metric tests pass"},
      {"command": "python scripts/evaluate_quality.py --model wan2.1-1.3b --quantized", "exitCode": 0, "observation": "FID: 99.1%, CLIPSIM: 99.3%, Temporal: 98.7%"},
      {"command": "python scripts/compare_quality.py --baseline fp16 --quantized w4a4", "exitCode": 0, "observation": "No statistically significant difference (p > 0.05)"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {"file": "tests/test_metrics.py", "cases": [
        {"name": "test_fid_computation", "verifies": "VAL-QTY-001"},
        {"name": "test_clipsim_computation", "verifies": "VAL-QTY-002"},
        {"name": "test_temporal_consistency", "verifies": "VAL-QTY-003"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Metric computation fails or produces invalid results
- Quality degradation exceeds 1% threshold
- Evaluation pipeline too slow for practical use
- Missing dependencies for metric computation
