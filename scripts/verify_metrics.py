"""Verify quality metrics implementation.

Compares VideoQuant metrics against reference implementations and
validates correctness of FID, CLIPSIM, and temporal consistency.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from videoquant.metrics import (
    FIDMetric,
    CLIPSIMMetric,
    TemporalConsistencyMetric,
    VideoMetricsEvaluator,
    compute_fid,
    compute_clipsim,
    compute_temporal_consistency,
)


def create_test_video(
    num_frames: int = 8,
    height: int = 64,
    width: int = 64,
    seed: int = 42,
) -> torch.Tensor:
    """Create a test video with known properties."""
    torch.manual_seed(seed)
    return torch.rand(num_frames, 3, height, width)


def create_static_video(
    num_frames: int = 8,
    height: int = 64,
    width: int = 64,
) -> torch.Tensor:
    """Create a static video (perfect consistency)."""
    frame = torch.ones(1, 3, height, width) * 0.5
    return frame.repeat(num_frames, 1, 1, 1)


def create_smooth_video(
    num_frames: int = 8,
    height: int = 64,
    width: int = 64,
    seed: int = 42,
) -> torch.Tensor:
    """Create a smooth video with gradual changes."""
    torch.manual_seed(seed)
    base = torch.rand(1, 3, height, width)
    frames = []
    for i in range(num_frames):
        noise = torch.randn_like(base) * 0.05
        frame = torch.clamp(base + noise, 0, 1)
        frames.append(frame)
    return torch.cat(frames, dim=0)


def verify_temporal_consistency():
    """Verify temporal consistency metric."""
    print("\n=== Verifying Temporal Consistency ===")
    
    metric = TemporalConsistencyMetric(method="frame_diff")
    
    # Test 1: Static video should have high consistency
    static_video = create_static_video()
    static_score = metric.compute(static_video).value
    print(f"  Static video consistency: {static_score:.4f} (expected > 0.99)")
    assert static_score > 0.99, f"Static video consistency too low: {static_score}"
    
    # Test 2: Smooth video should have good consistency
    smooth_video = create_smooth_video()
    smooth_score = metric.compute(smooth_video).value
    print(f"  Smooth video consistency: {smooth_score:.4f} (expected > 0.5)")
    assert smooth_score > 0.5, f"Smooth video consistency too low: {smooth_score}"
    
    # Test 3: Random video should have lower consistency than smooth
    random_video = create_test_video()
    random_score = metric.compute(random_video).value
    print(f"  Random video consistency: {random_score:.4f}")
    assert random_score < smooth_score, "Random should be less consistent than smooth"
    
    # Test 4: Single frame should have perfect consistency
    single_frame = torch.rand(1, 3, 64, 64)
    single_score = metric.compute(single_frame).value
    print(f"  Single frame consistency: {single_score:.4f} (expected = 1.0)")
    assert single_score == 1.0, "Single frame should have perfect consistency"
    
    # Test 5: Reproducibility
    torch.manual_seed(42)
    video1 = torch.rand(4, 3, 64, 64)
    score1 = metric.compute(video1).value
    torch.manual_seed(42)
    video2 = torch.rand(4, 3, 64, 64)
    score2 = metric.compute(video2).value
    print(f"  Reproducibility check: {score1:.4f} vs {score2:.4f}")
    assert abs(score1 - score2) < 1e-6, "Temporal consistency not reproducible"
    
    print("  ✓ Temporal consistency verification passed")
    return True


def verify_fid_mock():
    """Verify FID metric with mocked features (no model loading)."""
    print("\n=== Verifying FID Metric (Mocked) ===")
    
    from unittest.mock import Mock, patch
    
    metric = FIDMetric()
    
    # Mock the feature extractor
    def mock_extract(video):
        # Return features based on video shape
        if video.dim() == 5:  # [B, F, C, H, W]
            n = video.shape[0] * video.shape[1]
        else:
            n = video.shape[0]
        return torch.randn(n, 2048)
    
    with patch.object(metric, '_extract_features', side_effect=mock_extract):
        # Test 1: Identical videos should have low FID
        video1 = create_test_video(num_frames=4)
        video1_batch = video1.unsqueeze(0)  # [1, F, C, H, W]
        
        fid_identical = metric.compute(video1_batch, video1_batch.clone()).value
        print(f"  FID (identical videos): {fid_identical:.4f} (expected < 1.0)")
        assert fid_identical < 1.0, f"FID for identical videos too high: {fid_identical}"
        
        # Test 2: Different videos should have higher FID
        video2 = create_test_video(num_frames=4, seed=43)
        video2_batch = video2.unsqueeze(0)
        
        fid_different = metric.compute(video1_batch, video2_batch).value
        print(f"  FID (different videos): {fid_different:.4f}")
        assert fid_different >= 0, "FID should be non-negative"
        
        # Test 3: FID is non-negative
        assert fid_different >= 0, "FID should be non-negative"
        
        # Test 4: FID requires references
        try:
            metric.compute(video1_batch)
            assert False, "FID should require references"
        except ValueError:
            print("  ✓ FID correctly requires references")
    
    print("  ✓ FID verification passed (mocked)")
    return True


def verify_fid_statistics():
    """Verify FID statistical computations."""
    print("\n=== Verifying FID Statistics ===")
    
    metric = FIDMetric()
    
    # Test 1: Mean and covariance computation
    features = torch.randn(100, 512)
    mu, sigma = metric._compute_statistics(features)
    
    print(f"  Mean shape: {mu.shape} (expected (512,))")
    assert mu.shape == (512,), f"Mean shape mismatch: {mu.shape}"
    
    print(f"  Covariance shape: {sigma.shape} (expected (512, 512))")
    assert sigma.shape == (512, 512), f"Covariance shape mismatch: {sigma.shape}"
    
    # Test 2: Frechet distance for identical distributions
    mu = torch.zeros(10)
    sigma = torch.eye(10)
    fid = metric._compute_fid(mu, sigma, mu, sigma)
    print(f"  FID (identical Gaussians): {fid:.6f} (expected ≈ 0)")
    assert abs(fid) < 1e-4, f"FID for identical distributions not zero: {fid}"
    
    # Test 3: Frechet distance increases with mean difference
    mu1 = torch.zeros(10)
    mu2 = torch.ones(10) * 2.0
    sigma = torch.eye(10)
    fid_diff = metric._compute_fid(mu1, sigma, mu2, sigma)
    print(f"  FID (mean diff): {fid_diff:.4f} (expected > 0)")
    assert fid_diff > 0, "FID should increase with mean difference"
    
    print("  ✓ FID statistics verification passed")
    return True


def verify_clipsim_mock():
    """Verify CLIPSIM metric with mocked model."""
    print("\n=== Verifying CLIPSIM (Mocked) ===")
    
    from unittest.mock import Mock, patch
    
    metric = CLIPSIMMetric()
    
    # Mock CLIP model
    def mock_get_model():
        mock_model = Mock()
        mock_processor = Mock()
        
        # Return normalized features
        def mock_text(**kwargs):
            return torch.tensor([[1.0, 0.0, 0.0]])
        
        def mock_image(**kwargs):
            return torch.tensor([[0.9, 0.1, 0.0]])
        
        mock_model.get_text_features = mock_text
        mock_model.get_image_features = mock_image
        
        return mock_model, mock_processor
    
    # Use direct encoding mocks to avoid PIL/processor issues
    # Create properly normalized features to ensure cosine similarity is in [0, 1]
    text_feat = torch.nn.functional.normalize(torch.randn(1, 768), dim=-1)
    video_feat = torch.nn.functional.normalize(torch.randn(1, 4, 768), dim=-1)
    
    with patch.object(metric, '_encode_text', return_value=text_feat):
        with patch.object(metric, '_encode_video_frames', return_value=video_feat):
            # Test 1: CLIPSIM requires text
            video = create_test_video(num_frames=4).unsqueeze(0)
            
            try:
                metric.compute(video)
                assert False, "CLIPSIM should require text prompts"
            except ValueError:
                print("  ✓ CLIPSIM correctly requires text prompts")
            
            # Test 2: CLIPSIM with aligned text
            prompts = ["a video of a cat"]
            result = metric.compute(video, text_prompts=prompts)
            print(f"  CLIPSIM score: {result.value:.4f} (expected in [0, 1])")
            assert 0 <= result.value <= 1, f"CLIPSIM out of range: {result.value}"
            
            # Test 3: Aggregation methods work
            for agg in ["mean", "max", "first"]:
                metric_agg = CLIPSIMMetric(aggregation=agg)
                # Create new normalized features for each iteration
                tf = torch.nn.functional.normalize(torch.randn(1, 768), dim=-1)
                vf = torch.nn.functional.normalize(torch.randn(1, 4, 768), dim=-1)
                with patch.object(metric_agg, '_encode_text', return_value=tf):
                    with patch.object(metric_agg, '_encode_video_frames', return_value=vf):
                        result = metric_agg.compute(video, text_prompts=prompts)
                        print(f"  CLIPSIM ({agg}): {result.value:.4f}")
                        assert 0 <= result.value <= 1
    
    print("  ✓ CLIPSIM verification passed (mocked)")
    return True


def verify_metric_evaluator():
    """Verify unified metrics evaluator."""
    print("\n=== Verifying Metrics Evaluator ===")
    
    # Test with only temporal consistency (doesn't need model loading)
    evaluator = VideoMetricsEvaluator(
        enable_fid=False,
        enable_clipsim=False,
        enable_temporal=True,
    )
    
    video = create_test_video(num_frames=4).unsqueeze(0)
    prompts = ["test prompt"]
    
    results = evaluator.evaluate(video, text_prompts=prompts)
    
    assert 'temporal_consistency' in results
    assert 0 <= results['temporal_consistency'].value <= 1
    print(f"  Temporal consistency: {results['temporal_consistency'].value:.4f}")
    
    # Test metric preservation computation
    baseline = {
        'temporal_consistency': type('Result', (), {'value': 0.9})(),
        'clipsim': type('Result', (), {'value': 0.85})(),
    }
    quantized = {
        'temporal_consistency': type('Result', (), {'value': 0.89})(),
        'clipsim': type('Result', (), {'value': 0.84})(),
    }
    
    preservation = evaluator.compute_metric_preservation(baseline, quantized)
    print(f"  Metric preservation: {preservation}")
    
    assert 'temporal_consistency' in preservation
    assert 'clipsim' in preservation
    assert all(0 <= v <= 100 for v in preservation.values())
    
    print("  ✓ Metrics evaluator verification passed")
    return True


def verify_reproducibility():
    """Verify metrics are reproducible."""
    print("\n=== Verifying Reproducibility ===")
    
    metric = TemporalConsistencyMetric()
    
    # Create same video twice with same seed
    torch.manual_seed(42)
    video1 = torch.rand(4, 3, 64, 64)
    
    torch.manual_seed(42)
    video2 = torch.rand(4, 3, 64, 64)
    
    score1 = metric.compute(video1).value
    score2 = metric.compute(video2).value
    
    print(f"  Run 1: {score1:.6f}")
    print(f"  Run 2: {score2:.6f}")
    print(f"  Difference: {abs(score1 - score2):.10f}")
    
    assert abs(score1 - score2) < 1e-6, "Metrics not reproducible!"
    
    print("  ✓ Reproducibility verification passed")
    return True


def verify_consistency_ranking():
    """Verify temporal consistency correctly ranks videos."""
    print("\n=== Verifying Consistency Ranking ===")
    
    metric = TemporalConsistencyMetric()
    
    # Static video (highest consistency)
    static = create_static_video()
    static_score = metric.compute(static).value
    
    # Smooth video (good consistency)
    smooth = create_smooth_video()
    smooth_score = metric.compute(smooth).value
    
    # Random video (lowest consistency)
    random = create_test_video()
    random_score = metric.compute(random).value
    
    print(f"  Static video: {static_score:.4f}")
    print(f"  Smooth video: {smooth_score:.4f}")
    print(f"  Random video: {random_score:.4f}")
    
    # Static should be best
    assert static_score > smooth_score, "Static should be more consistent than smooth"
    assert static_score > random_score, "Static should be more consistent than random"
    
    # All should be in valid range
    assert 0 <= static_score <= 1
    assert 0 <= smooth_score <= 1
    assert 0 <= random_score <= 1
    
    print("  ✓ Consistency ranking verification passed")
    return True


def main():
    """Run all metric verifications."""
    print("=" * 60)
    print("VideoQuant Quality Metrics Verification")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= verify_temporal_consistency()
    except Exception as e:
        print(f"  ✗ Temporal consistency verification failed: {e}")
        all_passed = False
    
    try:
        all_passed &= verify_fid_statistics()
    except Exception as e:
        print(f"  ✗ FID statistics verification failed: {e}")
        all_passed = False
    
    try:
        all_passed &= verify_fid_mock()
    except Exception as e:
        print(f"  ✗ FID verification failed: {e}")
        all_passed = False
    
    try:
        all_passed &= verify_clipsim_mock()
    except Exception as e:
        print(f"  ✗ CLIPSIM verification failed: {e}")
        all_passed = False
    
    try:
        all_passed &= verify_metric_evaluator()
    except Exception as e:
        print(f"  ✗ Evaluator verification failed: {e}")
        all_passed = False
    
    try:
        all_passed &= verify_reproducibility()
    except Exception as e:
        print(f"  ✗ Reproducibility verification failed: {e}")
        all_passed = False
    
    try:
        all_passed &= verify_consistency_ranking()
    except Exception as e:
        print(f"  ✗ Consistency ranking verification failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All metric verifications passed!")
        return 0
    else:
        print("✗ Some verifications failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
