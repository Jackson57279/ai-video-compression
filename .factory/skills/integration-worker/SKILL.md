---
name: integration-worker
description: Integrates VideoQuant with HuggingFace Diffusers, Wan2.1 models, and ComfyUI nodes
---

# Integration Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

This worker handles VideoQuant integration with external frameworks:
- HuggingFace Diffusers pipeline integration
- Wan2.1-T2V-1.3B model loading and quantization
- ComfyUI custom nodes
- CLI tools and scripts

Features using this skill:
- Diffusers model wrapper with quantization
- Wan2.1 model download and setup
- Quantized pipeline execution
- ComfyUI node implementation

## Required Skills

None. Pure Python implementation using PyTorch and Diffusers.

## Work Procedure

### Phase 1: Framework Analysis

1. **Study target framework**
   - For Diffusers: Study pipeline structure, model loading, inference flow
   - For ComfyUI: Study node API, tensor handling, execution graph
   - Identify integration points for quantization hooks

### Phase 2: Integration Implementation

2. **Create framework adapter**
   - Diffusers: Custom pipeline or model wrapper
   - ComfyUI: Custom node class with quantization hooks
   - Maintain compatibility with existing features

3. **Implement model quantization**
   - Weight quantization on model load
   - Activation quantization hooks in forward pass
   - Configurable quantization parameters

4. **Add configuration support**
   - YAML/JSON config files
   - Command-line arguments
   - Runtime configuration changes

### Phase 3: Wan2.1 Model Integration

5. **Setup model integration**
   - Model download from HuggingFace
   - Checkpoint loading with quantization
   - Verify memory usage meets targets

6. **Test inference pipeline**
   - Text-to-video generation
   - Verify output quality
   - Measure memory and speed

### Phase 4: Testing and Validation

7. **Run integration tests**
   - Full pipeline execution
   - Error handling
   - Resource usage validation

8. **Manual verification**
   - Generate test videos
   - Compare quality metrics
   - Document any issues

## Example Handoff

```json
{
  "salientSummary": "Integrated VideoQuant with Diffusers pipeline. Wan2.1-1.3B loads with W4A4 quantization, achieving 4.2x memory reduction. CPU inference functional, 512x512x16 video generation takes ~8 minutes.",
  "whatWasImplemented": "DiffusersVideoQuantPipeline wrapper, Wan2.1 model integration with weight quantization, CLI tool for video generation, ComfyUI VideoQuantNode implementation.",
  "whatWasLeftUndone": "Multi-GPU support not implemented. Batch generation optimization pending.",
  "verification": {
    "commandsRun": [
      {"command": "python -m pytest tests/test_diffusers_integration.py -v", "exitCode": 0, "observation": "Pipeline loads and generates with quantization"},
      {"command": "python scripts/generate_video.py --model wan2.1-1.3b --quantized --prompt 'a cat playing'", "exitCode": 0, "observation": "Generated 16-frame video in 8min 12s"},
      {"command": "python scripts/benchmark_memory.py", "exitCode": 0, "observation": "Memory reduced from 10.2GB to 2.4GB (4.25x)"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {"file": "tests/test_diffusers_integration.py", "cases": [
        {"name": "test_pipeline_quantization", "verifies": "VAL-INT-002"},
        {"name": "test_model_loading", "verifies": "VAL-WAN-001"}
      ]},
      {"file": "tests/test_video_generation.py", "cases": [
        {"name": "test_quantized_generation", "verifies": "VAL-WAN-002"},
        {"name": "test_memory_reduction", "verifies": "VAL-WAN-003"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Framework API incompatibility discovered
- Model loading fails with quantization
- Integration causes unacceptable performance degradation
- Dependencies cannot be satisfied
