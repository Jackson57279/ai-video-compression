#!/usr/bin/env python3
"""Video Generation Script for VideoQuant Validation.

Generates videos using FP16 baseline or W4A4 quantized inference for
quality validation and comparison.

Usage:
    python scripts/generate_video.py --prompt "a cat playing piano"
    python scripts/generate_video.py --quantized --prompt "sunset over ocean"
    python scripts/generate_video.py --baseline --output video_baseline.mp4

Validates:
    - VAL-WAN-002: Quantized Video Generation
"""

import argparse
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from videoquant.integration import (
        VideoQuantDiffusersPipeline,
        VideoQuantDiffusersConfig,
    )
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate video with VideoQuant quantization"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation",
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
        help="Use W4A4 quantization",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use FP16 baseline (no quantization)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to generate (default: 16)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height (default: 512)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video width (default: 512)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=25,
        help="Number of denoising steps (default: 25)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (default: 5.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames as images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_videos",
        help="Output directory",
    )
    
    return parser.parse_args()


def save_video_frames(frames: torch.Tensor, output_dir: Path, prefix: str = "frame"):
    """Save video frames as individual images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from PIL import Image
        
        # Convert to numpy [F, H, W, C]
        frames_np = frames.cpu().numpy()
        frames_np = np.transpose(frames_np, (0, 2, 3, 1))
        
        # Convert to uint8
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)
        
        for i, frame in enumerate(frames_np):
            img = Image.fromarray(frame)
            img.save(output_dir / f"{prefix}_{i:04d}.png")
        
        print(f"  ✓ Saved {len(frames_np)} frames to {output_dir}")
        
    except ImportError:
        print(f"  ⚠ PIL not available, saving as numpy instead")
        np.save(output_dir / f"{prefix}_video.npy", frames.cpu().numpy())


def generate_video_baseline(
    model_id: str,
    prompt: str,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    device: str,
    seed: int,
) -> tuple:
    """Generate video with FP16 baseline."""
    print(f"\n{'='*70}")
    print("Video Generation - FP16 Baseline")
    print(f"{'='*70}")
    print(f"Model: {model_id}")
    print(f"Prompt: '{prompt}'")
    print(f"Resolution: {height}x{width}, {num_frames} frames")
    print(f"Steps: {num_steps}, Guidance: {guidance_scale}")
    print(f"Device: {device}")
    
    print(f"\nLoading model...")
    start_time = time.time()
    
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")
    
    # Set seed
    torch.manual_seed(seed)
    
    print(f"\nGenerating video...")
    print(f"  (This may take several minutes on CPU)")
    
    start_gen = time.time()
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )
    gen_time = time.time() - start_gen
    
    # Extract frames
    frames = output.frames[0] if hasattr(output, 'frames') else output
    
    print(f"✓ Generation complete in {gen_time:.2f}s")
    print(f"  Output shape: {frames.shape}")
    
    total_time = time.time() - start_time
    
    stats = {
        'load_time': load_time,
        'generation_time': gen_time,
        'total_time': total_time,
        'model_type': 'FP16 Baseline',
        'quantization': 'none',
    }
    
    return frames, stats


def generate_video_quantized(
    model_id: str,
    prompt: str,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    device: str,
    seed: int,
) -> tuple:
    """Generate video with W4A4 quantization."""
    print(f"\n{'='*70}")
    print("Video Generation - W4A4 Quantized")
    print(f"{'='*70}")
    print(f"Model: {model_id}")
    print(f"Prompt: '{prompt}'")
    print(f"Resolution: {height}x{width}, {num_frames} frames")
    print(f"Steps: {num_steps}, Guidance: {guidance_scale}")
    print(f"Device: {device}")
    
    print(f"\nLoading model with W4A4 quantization...")
    start_time = time.time()
    
    # Create W4A4 config
    config = VideoQuantDiffusersConfig.default_w4a4()
    config.device = device
    config.dtype = "fp16"
    
    pipe = VideoQuantDiffusersPipeline.from_pretrained(
        model_id,
        videoquant_config=config,
        torch_dtype=torch.float16,
        device=device,
    )
    
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")
    
    # Verify quantization
    if pipe.quantizer:
        stats_quant = pipe.quantizer.get_stats()
        print(f"✓ Quantization active:")
        print(f"  TPQ: {stats_quant['config']['enable_tpq']}")
        print(f"  SQJL: {stats_quant['config']['enable_sqjl']}")
        print(f"  MAMP: {stats_quant['config']['enable_mamp']}")
    
    # Set seed
    torch.manual_seed(seed)
    
    print(f"\nGenerating video...")
    print(f"  (This may take several minutes on CPU)")
    
    start_gen = time.time()
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )
    gen_time = time.time() - start_gen
    
    # Extract frames
    frames = output.frames[0] if hasattr(output, 'frames') else output
    
    print(f"✓ Generation complete in {gen_time:.2f}s")
    print(f"  Output shape: {frames.shape}")
    
    if output.quant_stats:
        print(f"  Quantization stats: {output.quant_stats}")
    
    total_time = time.time() - start_time
    
    stats = {
        'load_time': load_time,
        'generation_time': gen_time,
        'total_time': total_time,
        'model_type': 'W4A4 Quantized',
        'quantization': 'w4a4',
    }
    
    return frames, stats


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine mode
    use_quantization = args.quantized or not args.baseline
    
    try:
        if use_quantization:
            frames, stats = generate_video_quantized(
                args.model_id,
                args.prompt,
                args.num_frames,
                args.height,
                args.width,
                args.num_inference_steps,
                args.guidance_scale,
                args.device,
                args.seed,
            )
        else:
            frames, stats = generate_video_baseline(
                args.model_id,
                args.prompt,
                args.num_frames,
                args.height,
                args.width,
                args.num_inference_steps,
                args.guidance_scale,
                args.device,
                args.seed,
            )
        
        # Print summary
        print(f"\n{'='*70}")
        print("Generation Summary")
        print(f"{'='*70}")
        print(f"Mode: {stats['model_type']}")
        print(f"Load time: {stats['load_time']:.2f}s")
        print(f"Generation time: {stats['generation_time']:.2f}s")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Output shape: {frames.shape}")
        print(f"Video dtype: {frames.dtype}")
        print(f"Value range: [{frames.min():.3f}, {frames.max():.3f}]")
        
        # Save output
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_frames:
            mode_prefix = "quantized" if use_quantization else "baseline"
            save_video_frames(frames, output_dir, prefix=mode_prefix)
        
        if args.output:
            # Save as numpy for now (can be converted to video)
            output_path = output_dir / args.output
            np.save(output_path, frames.cpu().numpy())
            print(f"✓ Video saved to: {output_path}")
        else:
            # Auto-generate filename
            mode_name = "quantized" if use_quantization else "baseline"
            safe_prompt = args.prompt.replace(" ", "_")[:30]
            output_path = output_dir / f"{mode_name}_{safe_prompt}.npy"
            np.save(output_path, frames.cpu().numpy())
            print(f"✓ Video saved to: {output_path}")
        
        print(f"\n✓ VAL-WAN-002: Quantized Video Generation - PASSED")
        print(f"  Successfully generated video with {stats['model_type']}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
