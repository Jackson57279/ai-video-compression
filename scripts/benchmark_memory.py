#!/usr/bin/env python3
"""
Memory Benchmarking Script for VideoQuant Wan2.1 Inference.

This script benchmarks memory usage during Wan2.1-T2V-1.3B inference
with and without quantization to verify the 4x memory reduction target.

Usage:
    python scripts/benchmark_memory.py --model wan2.1-1.3b
    python scripts/benchmark_memory.py --quantized
    python scripts/benchmark_memory.py --baseline

Validates:
    - VAL-WAN-003: Memory Reduction Verification (≤ 25% of FP16)
"""

import argparse
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from videoquant.integration import (
        VideoQuantDiffusersPipeline,
        VideoQuantDiffusersConfig,
    )
    from diffusers import DiffusionPipeline
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure diffusers and videoquant are installed.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark memory usage for Wan2.1 inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wan2.1-1.3b",
        help="Model variant to benchmark",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Benchmark W4A4 quantized inference",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Benchmark FP16 baseline (no quantization)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu, cuda:0, etc.)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video width",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=25,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cat playing with a ball",
        help="Test prompt for generation",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=2.5,
        help="Maximum memory target in GB for quantized (4x reduction from ~10GB)",
    )
    
    return parser.parse_args()


def format_bytes(bytes_val: float) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def format_gb(bytes_val: float) -> str:
    """Format bytes to GB string."""
    return f"{bytes_val / (1024**3):.2f} GB"


def get_memory_usage():
    """Get current memory usage in bytes."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    except ImportError:
        return 0


def get_gpu_memory():
    """Get GPU memory usage if available."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
        }
    return None


class MemoryProfiler:
    """Profile memory usage during inference."""
    
    def __init__(self):
        self.measurements: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        
    def start(self):
        """Start profiling."""
        self.start_time = time.time()
        self.measurements = []
        
        # Reset GPU stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def record(self, label: str):
        """Record a memory measurement."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        measurement = {
            "label": label,
            "elapsed_time": elapsed,
            "cpu_memory": get_memory_usage(),
        }
        
        gpu_mem = get_gpu_memory()
        if gpu_mem:
            measurement["gpu_memory"] = gpu_mem
        
        self.measurements.append(measurement)
        return measurement
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of memory profiling."""
        if not self.measurements:
            return {}
        
        cpu_memories = [m["cpu_memory"] for m in self.measurements]
        
        summary = {
            "total_time": self.measurements[-1]["elapsed_time"] if self.measurements else 0,
            "cpu_memory_peak": max(cpu_memories) if cpu_memories else 0,
            "cpu_memory_start": cpu_memories[0] if cpu_memories else 0,
        }
        
        # Get GPU summary if available
        gpu_memories = []
        for m in self.measurements:
            if "gpu_memory" in m:
                gpu_memories.append(m["gpu_memory"]["max_allocated"])
        
        if gpu_memories:
            summary["gpu_memory_peak"] = max(gpu_memories)
        
        return summary


def benchmark_quantized(
    model_id: str,
    device: str,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int,
    prompt: str,
) -> Dict[str, Any]:
    """Benchmark W4A4 quantized inference."""
    print(f"\n{'='*70}")
    print("BENCHMARK: W4A4 Quantized Inference")
    print(f"{'='*70}")
    
    profiler = MemoryProfiler()
    
    # Configuration
    config = VideoQuantDiffusersConfig.default_w4a4()
    config.device = device
    config.dtype = "fp16"
    
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Resolution: {height}x{width}, {num_frames} frames")
    print(f"Steps: {num_steps}")
    print(f"Prompt: '{prompt}'")
    print()
    
    try:
        # Load model
        profiler.start()
        print("Loading model with W4A4 quantization...")
        
        pipe = VideoQuantDiffusersPipeline.from_pretrained(
            model_id,
            videoquant_config=config,
            torch_dtype=torch.float16,
            device=device,
        )
        
        load_measurement = profiler.record("model_loaded")
        print(f"✓ Model loaded in {load_measurement['elapsed_time']:.2f}s")
        print(f"  Memory: {format_bytes(load_measurement['cpu_memory'])}")
        
        # Run inference
        print(f"\nRunning inference...")
        
        # Warmup
        print("  Warmup pass...")
        _ = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=1,  # Single step for warmup
        )
        
        warmup_measurement = profiler.record("warmup_complete")
        print(f"  ✓ Warmup complete in {warmup_measurement['elapsed_time']:.2f}s")
        
        # Full inference
        print(f"  Full inference ({num_steps} steps)...")
        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
        )
        
        inference_measurement = profiler.record("inference_complete")
        print(f"  ✓ Inference complete in {inference_measurement['elapsed_time']:.2f}s")
        
        # Get summary
        summary = profiler.get_summary()
        
        print(f"\n{'='*70}")
        print("W4A4 Quantized Results")
        print(f"{'='*70}")
        print(f"Load time: {load_measurement['elapsed_time']:.2f}s")
        print(f"Total time: {summary['total_time']:.2f}s")
        print(f"Peak CPU memory: {format_bytes(summary['cpu_memory_peak'])}")
        print(f"Peak CPU memory: {format_gb(summary['cpu_memory_peak'])}")
        
        if 'gpu_memory_peak' in summary:
            print(f"Peak GPU memory: {format_bytes(summary['gpu_memory_peak'])}")
        
        # Verify quantization was applied
        if pipe.quantizer:
            stats = pipe.quantizer.get_stats()
            print(f"\nQuantization active: {stats['enabled']}")
            print(f"  TPQ: {stats['config']['enable_tpq']}")
            print(f"  SQJL: {stats['config']['enable_sqjl']}")
            print(f"  MAMP: {stats['config']['enable_mamp']}")
        
        return {
            "mode": "w4a4_quantized",
            "load_time": load_measurement['elapsed_time'],
            "inference_time": inference_measurement['elapsed_time'] - warmup_measurement['elapsed_time'],
            "total_time": summary['total_time'],
            "peak_memory_bytes": summary['cpu_memory_peak'],
            "peak_memory_gb": summary['cpu_memory_peak'] / (1024**3),
            "success": True,
        }
        
    except Exception as e:
        print(f"\n✗ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {
            "mode": "w4a4_quantized",
            "success": False,
            "error": str(e),
        }


def benchmark_baseline(
    model_id: str,
    device: str,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int,
    prompt: str,
) -> Dict[str, Any]:
    """Benchmark FP16 baseline inference."""
    print(f"\n{'='*70}")
    print("BENCHMARK: FP16 Baseline (No Quantization)")
    print(f"{'='*70}")
    
    profiler = MemoryProfiler()
    
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Resolution: {height}x{width}, {num_frames} frames")
    print(f"Steps: {num_steps}")
    print(f"Prompt: '{prompt}'")
    print()
    
    try:
        # Load model
        profiler.start()
        print("Loading model (FP16 baseline)...")
        
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipe = pipe.to(device)
        
        load_measurement = profiler.record("model_loaded")
        print(f"✓ Model loaded in {load_measurement['elapsed_time']:.2f}s")
        print(f"  Memory: {format_bytes(load_measurement['cpu_memory'])}")
        
        # Run inference
        print(f"\nRunning inference...")
        
        # Warmup
        print("  Warmup pass...")
        _ = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=1,
        )
        
        warmup_measurement = profiler.record("warmup_complete")
        print(f"  ✓ Warmup complete in {warmup_measurement['elapsed_time']:.2f}s")
        
        # Full inference
        print(f"  Full inference ({num_steps} steps)...")
        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
        )
        
        inference_measurement = profiler.record("inference_complete")
        print(f"  ✓ Inference complete in {inference_measurement['elapsed_time']:.2f}s")
        
        # Get summary
        summary = profiler.get_summary()
        
        print(f"\n{'='*70}")
        print("FP16 Baseline Results")
        print(f"{'='*70}")
        print(f"Load time: {load_measurement['elapsed_time']:.2f}s")
        print(f"Total time: {summary['total_time']:.2f}s")
        print(f"Peak CPU memory: {format_bytes(summary['cpu_memory_peak'])}")
        print(f"Peak CPU memory: {format_gb(summary['cpu_memory_peak'])}")
        
        return {
            "mode": "fp16_baseline",
            "load_time": load_measurement['elapsed_time'],
            "inference_time": inference_measurement['elapsed_time'] - warmup_measurement['elapsed_time'],
            "total_time": summary['total_time'],
            "peak_memory_bytes": summary['cpu_memory_peak'],
            "peak_memory_gb": summary['cpu_memory_peak'] / (1024**3),
            "success": True,
        }
        
    except Exception as e:
        print(f"\n✗ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {
            "mode": "fp16_baseline",
            "success": False,
            "error": str(e),
        }


def compare_results(quantized_results: Dict, baseline_results: Dict, max_memory_gb: float):
    """Compare quantized vs baseline results."""
    print(f"\n{'='*70}")
    print("COMPARISON: W4A4 Quantized vs FP16 Baseline")
    print(f"{'='*70}")
    
    if not quantized_results.get("success"):
        print("✗ Quantized benchmark failed - cannot compare")
        return False
    
    if not baseline_results.get("success"):
        print("⚠ Baseline benchmark failed - using estimated values")
        # Estimate baseline at ~10GB for FP16
        baseline_memory_gb = 10.0
    else:
        baseline_memory_gb = baseline_results["peak_memory_gb"]
    
    quantized_memory_gb = quantized_results["peak_memory_gb"]
    
    # Calculate reduction
    if baseline_memory_gb > 0:
        memory_reduction = baseline_memory_gb / quantized_memory_gb
        memory_savings_pct = (1 - quantized_memory_gb / baseline_memory_gb) * 100
    else:
        memory_reduction = 0
        memory_savings_pct = 0
    
    print(f"\nMemory Usage:")
    print(f"  FP16 Baseline:     {baseline_memory_gb:.2f} GB")
    print(f"  W4A4 Quantized:    {quantized_memory_gb:.2f} GB")
    print(f"  Reduction:         {memory_reduction:.2f}x ({memory_savings_pct:.1f}% reduction)")
    
    # Target validation
    print(f"\nTarget Validation:")
    print(f"  Target memory:     ≤ {max_memory_gb:.2f} GB")
    print(f"  Actual memory:     {quantized_memory_gb:.2f} GB")
    
    target_met = quantized_memory_gb <= max_memory_gb
    
    if target_met:
        print(f"  ✓ PASS: Memory target met")
    else:
        print(f"  ⚠ WARNING: Memory exceeds target by {quantized_memory_gb - max_memory_gb:.2f} GB")
    
    # 4x reduction validation
    print(f"\n4x Memory Reduction:")
    if memory_reduction >= 4.0:
        print(f"  ✓ PASS: {memory_reduction:.2f}x reduction achieved (target: 4x)")
        four_x_met = True
    else:
        print(f"  ⚠ WARNING: Only {memory_reduction:.2f}x reduction (target: 4x)")
        print(f"    Expected memory: {baseline_memory_gb / 4:.2f} GB")
        print(f"    Actual memory:   {quantized_memory_gb:.2f} GB")
        four_x_met = False
    
    # Timing comparison
    if baseline_results.get("success") and quantized_results.get("success"):
        print(f"\nTiming Comparison:")
        print(f"  Baseline load:     {baseline_results['load_time']:.2f}s")
        print(f"  Quantized load:    {quantized_results['load_time']:.2f}s")
        
        baseline_inference = baseline_results.get('inference_time', 0)
        quantized_inference = quantized_results.get('inference_time', 0)
        
        if baseline_inference > 0 and quantized_inference > 0:
            speedup = baseline_inference / quantized_inference
            print(f"  Baseline inference: {baseline_inference:.2f}s")
            print(f"  Quantized inference: {quantized_inference:.2f}s")
            print(f"  Speedup:            {speedup:.2f}x")
    
    # Final validation
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    if target_met and four_x_met:
        print("✓ VAL-WAN-003: Memory Reduction Verification - PASSED")
        print(f"  Quantized model uses {memory_reduction:.2f}x less memory than FP16")
        print(f"  Peak memory: {quantized_memory_gb:.2f} GB (target: ≤ {max_memory_gb:.2f} GB)")
        return True
    else:
        print("⚠ VAL-WAN-003: Memory Reduction Verification - PARTIAL")
        if not target_met:
            print(f"  Memory target not met: {quantized_memory_gb:.2f} GB > {max_memory_gb:.2f} GB")
        if not four_x_met:
            print(f"  4x reduction not achieved: {memory_reduction:.2f}x < 4x")
        return False


def main():
    """Main entry point."""
    args = parse_args()
    
    # If neither --quantized nor --baseline specified, run both
    run_both = not (args.quantized or args.baseline)
    
    baseline_results = None
    quantized_results = None
    
    try:
        # Run baseline first (if requested or running both)
        if args.baseline or run_both:
            baseline_results = benchmark_baseline(
                args.model_id,
                args.device,
                args.num_frames,
                args.height,
                args.width,
                args.num_inference_steps,
                args.prompt,
            )
        
        # Run quantized (if requested or running both)
        if args.quantized or run_both:
            quantized_results = benchmark_quantized(
                args.model_id,
                args.device,
                args.num_frames,
                args.height,
                args.width,
                args.num_inference_steps,
                args.prompt,
            )
        
        # Compare if we have both results
        if run_both and baseline_results and quantized_results:
            success = compare_results(quantized_results, baseline_results, args.max_memory_gb)
            return 0 if success else 1
        
        # Return success if at least one benchmark succeeded
        if (baseline_results and baseline_results.get("success")) or \
           (quantized_results and quantized_results.get("success")):
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
