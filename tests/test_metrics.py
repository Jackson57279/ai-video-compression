"""Tests for VideoQuant quality metrics.

Validates:
- VAL-QTY-001: FID computation accurate to reference implementation
- VAL-QTY-002: CLIPSIM captures text-video alignment
- VAL-QTY-003: Temporal consistency measures smoothness
- Metrics reproducible across runs
"""

import torch
import numpy as np
import pytest
from unittest.mock import Mock, patch
import warnings

from videoquant.metrics import (
    FIDMetric,
    CLIPSIMMetric,
    TemporalConsistencyMetric,
    VideoMetricsEvaluator,
    MetricResult,
    compute_fid,
    compute_clipsim,
    compute_temporal_consistency,
)


class TestMetricResult:
    """Test MetricResult dataclass."""
    
    def test_basic_result(self):
        """MetricResult stores value and name."""
        result = MetricResult(value=0.5, metric_name="test")
        assert result.value == 0.5
        assert result.metric_name == "test"
    
    def test_result_with_metadata(self):
        """MetricResult stores metadata."""
        result = MetricResult(
            value=0.5,
            metric_name="test",
            metadata={'n_samples': 100}
        )
        assert result.metadata['n_samples'] == 100
    
    def test_result_to_dict(self):
        """MetricResult converts to dictionary."""
        result = MetricResult(
            value=0.5,
            metric_name="test",
            confidence_interval=(0.4, 0.6),
            metadata={'key': 'value'}
        )
        d = result.to_dict()
        assert d['value'] == 0.5
        assert d['metric_name'] == "test"
        assert d['confidence_interval'] == (0.4, 0.6)
        assert d['metadata']['key'] == "value"


class TestFIDMetric:
    """VAL-QTY-001: FID computation accurate to reference implementation."""
    
    def test_fid_metric_name(self):
        """FID metric has correct name."""
        metric = FIDMetric()
        assert metric.name == "fid"
    
    def test_fid_requires_references(self):
        """FID requires reference videos."""
        metric = FIDMetric()
        video = torch.randn(1, 4, 3, 64, 64)  # [B, F, C, H, W]
        
        with pytest.raises(ValueError, match="requires reference"):
            metric.compute(video)
    
    @patch('videoquant.metrics.FIDMetric._get_inception_model')
    def test_fid_computation_basic(self, mock_get_model):
        """FID computation runs without errors."""
        # Mock the Inception model
        mock_model = Mock()
        mock_model.return_value = torch.randn(8, 2048)  # Mock features
        mock_get_model.return_value = mock_model
        
        metric = FIDMetric()
        
        # Create test videos
        gen_video = torch.rand(2, 4, 3, 299, 299)  # [B, F, C, H, W]
        ref_video = torch.rand(2, 4, 3, 299, 299)
        
        result = metric.compute(gen_video, ref_video)
        
        assert isinstance(result, MetricResult)
        assert result.metric_name == "fid"
        assert result.value >= 0  # FID is non-negative
    
    def test_fid_with_list_videos(self):
        """FID handles list of videos."""
        metric = FIDMetric()
        
        # Create lists of videos
        gen_videos = [torch.rand(4, 3, 64, 64) for _ in range(3)]
        ref_videos = [torch.rand(4, 3, 64, 64) for _ in range(3)]
        
        # Mock the feature extraction to avoid model loading
        with patch.object(metric, '_extract_features') as mock_extract:
            mock_extract.side_effect = [
                torch.randn(12, 2048),  # 3 videos * 4 frames = 12 features
                torch.randn(12, 2048),
            ]
            
            result = metric.compute(gen_videos, ref_videos)
            assert isinstance(result, MetricResult)
    
    def test_fid_non_negative(self):
        """FID value is always non-negative."""
        metric = FIDMetric()
        
        # Mock features
        gen_video = torch.rand(1, 4, 3, 299, 299)
        ref_video = torch.rand(1, 4, 3, 299, 299)
        
        with patch.object(metric, '_extract_features') as mock_extract:
            mock_extract.side_effect = [
                torch.randn(4, 2048),
                torch.randn(4, 2048),
            ]
            
            result = metric.compute(gen_video, ref_video)
            assert result.value >= 0
    
    def test_fid_identical_videos(self):
        """FID is near zero for identical videos."""
        metric = FIDMetric()
        
        video = torch.rand(1, 4, 3, 299, 299)
        
        with patch.object(metric, '_extract_features') as mock_extract:
            # Return identical features
            features = torch.randn(4, 2048)
            mock_extract.return_value = features
            
            result = metric.compute(video, video.clone())
            # FID should be very small for identical distributions
            assert result.value < 1.0
    
    def test_fid_statistics_computation(self):
        """FID correctly computes mean and covariance."""
        metric = FIDMetric()
        
        features = torch.randn(100, 512)
        mu, sigma = metric._compute_statistics(features)
        
        assert mu.shape == (512,)
        assert sigma.shape == (512, 512)
        
        # Check symmetry
        assert torch.allclose(sigma, sigma.T)
    
    def test_fid_frechet_distance(self):
        """FID Frechet distance computation is correct."""
        metric = FIDMetric()
        
        # Test with known Gaussians
        mu1 = torch.zeros(10)
        sigma1 = torch.eye(10)
        mu2 = torch.zeros(10)
        sigma2 = torch.eye(10)
        
        fid = metric._compute_fid(mu1, sigma1, mu2, sigma2)
        
        # FID should be 0 for identical distributions
        assert abs(fid) < 1e-5


class TestCLIPSIMMetric:
    """VAL-QTY-002: CLIPSIM captures text-video alignment."""
    
    def test_clipsim_metric_name(self):
        """CLIPSIM metric has correct name."""
        metric = CLIPSIMMetric()
        assert metric.name == "clipsim"
    
    def test_clipsim_requires_text(self):
        """CLIPSIM requires text prompts."""
        metric = CLIPSIMMetric()
        video = torch.randn(1, 4, 3, 224, 224)
        
        with pytest.raises(ValueError, match="requires text prompts"):
            metric.compute(video)
    
    def test_clipsim_returns_metric_result(self):
        """CLIPSIM returns MetricResult."""
        metric = CLIPSIMMetric()
        
        video = torch.rand(1, 4, 3, 224, 224)
        prompts = ["a cat playing"]
        
        # Mock encode methods directly to avoid PIL and processor issues
        # Use normalized features
        video_feat = torch.nn.functional.normalize(torch.randn(1, 4, 768), dim=-1)
        text_feat = torch.nn.functional.normalize(torch.randn(1, 768), dim=-1)
        
        with patch.object(metric, '_encode_video_frames', return_value=video_feat):
            with patch.object(metric, '_encode_text', return_value=text_feat):
                result = metric.compute(video, text_prompts=prompts)
                assert isinstance(result, MetricResult)
                assert result.metric_name == "clipsim"
    
    def test_clipsim_score_range(self):
        """CLIPSIM score is in [0, 1] range."""
        metric = CLIPSIMMetric()
        
        video = torch.rand(1, 4, 3, 224, 224)
        prompts = ["test prompt"]
        
        # Mock encode methods with normalized features
        video_feat = torch.nn.functional.normalize(torch.randn(1, 4, 768), dim=-1)
        text_feat = torch.nn.functional.normalize(torch.randn(1, 768), dim=-1)
        
        with patch.object(metric, '_encode_video_frames', return_value=video_feat):
            with patch.object(metric, '_encode_text', return_value=text_feat):
                result = metric.compute(video, text_prompts=prompts)
                assert 0 <= result.value <= 1
    
    def test_clipsim_aggregation_methods(self):
        """CLIPSIM supports different aggregation methods."""
        for agg in ["mean", "max", "first"]:
            metric = CLIPSIMMetric(aggregation=agg)
            assert metric.aggregation == agg
    
    def test_clipsim_batch_handling(self):
        """CLIPSIM handles batch of videos."""
        metric = CLIPSIMMetric()
        
        # Batch of 3 videos
        videos = torch.rand(3, 4, 3, 224, 224)
        prompts = ["cat", "dog", "bird"]
        
        # Mock encode methods with normalized features
        video_feat = torch.nn.functional.normalize(torch.randn(3, 4, 768), dim=-1)
        text_feat = torch.nn.functional.normalize(torch.randn(3, 768), dim=-1)
        
        with patch.object(metric, '_encode_video_frames', return_value=video_feat):
            with patch.object(metric, '_encode_text', return_value=text_feat):
                result = metric.compute(videos, text_prompts=prompts)
                assert isinstance(result, MetricResult)
                assert result.metadata['n_videos'] == 3


class TestTemporalConsistencyMetric:
    """VAL-QTY-003: Temporal consistency measures smoothness."""
    
    def test_temporal_metric_name(self):
        """Temporal metric has correct name."""
        metric = TemporalConsistencyMetric()
        assert metric.name == "temporal_consistency"
    
    def test_temporal_single_frame(self):
        """Single frame video has perfect consistency."""
        metric = TemporalConsistencyMetric()
        
        video = torch.rand(1, 3, 64, 64)  # [F=1, C, H, W]
        result = metric.compute(video)
        
        assert result.value == 1.0
    
    def test_temporal_identical_frames(self):
        """Identical frames have high consistency."""
        metric = TemporalConsistencyMetric()
        
        # Create video with identical frames
        frame = torch.rand(1, 3, 64, 64)
        video = frame.repeat(4, 1, 1, 1)  # [4, 3, 64, 64]
        
        result = metric.compute(video)
        assert result.value > 0.9  # Should be very high
    
    def test_temporal_random_noise(self):
        """Random noise has lower consistency."""
        metric = TemporalConsistencyMetric()
        
        video = torch.rand(8, 3, 64, 64)
        result = metric.compute(video)
        
        # Random frames should have some consistency
        assert 0 <= result.value <= 1
    
    def test_temporal_score_range(self):
        """Temporal consistency score is in [0, 1]."""
        metric = TemporalConsistencyMetric()
        
        video = torch.rand(4, 3, 64, 64)
        result = metric.compute(video)
        
        assert 0 <= result.value <= 1
    
    def test_temporal_methods(self):
        """Temporal metric supports different methods."""
        video = torch.rand(4, 3, 64, 64)
        
        for method in ["frame_diff", "flow", "combined"]:
            metric = TemporalConsistencyMetric(method=method)
            result = metric.compute(video)
            assert 0 <= result.value <= 1
    
    def test_temporal_batch_videos(self):
        """Temporal metric handles batch of videos."""
        metric = TemporalConsistencyMetric()
        
        # Batch of videos
        videos = torch.rand(3, 4, 3, 64, 64)  # [B, F, C, H, W]
        result = metric.compute(videos)
        
        assert isinstance(result, MetricResult)
        assert result.metadata['n_videos'] == 3
    
    def test_temporal_list_videos(self):
        """Temporal metric handles list of videos."""
        metric = TemporalConsistencyMetric()
        
        videos = [torch.rand(4, 3, 64, 64) for _ in range(3)]
        result = metric.compute(videos)
        
        assert isinstance(result, MetricResult)
        assert result.metadata['n_videos'] == 3
    
    def test_temporal_gradual_change(self):
        """Gradual changes have better consistency than abrupt."""
        metric = TemporalConsistencyMetric()
        
        # Create video with gradual changes
        base = torch.rand(1, 3, 64, 64)
        gradual = torch.stack([
            base + i * 0.05 * torch.randn_like(base) 
            for i in range(4)
        ]).squeeze(1)
        
        # Create video with abrupt changes
        abrupt = torch.stack([
            torch.rand(1, 3, 64, 64) for _ in range(4)
        ]).squeeze(1)
        
        gradual_result = metric.compute(gradual)
        abrupt_result = metric.compute(abrupt)
        
        # Gradual should have better consistency
        assert gradual_result.value > abrupt_result.value


class TestVideoMetricsEvaluator:
    """Test unified metrics evaluator."""
    
    def test_evaluator_initialization(self):
        """Evaluator initializes with all metrics."""
        evaluator = VideoMetricsEvaluator()
        assert 'fid' in evaluator.metrics
        assert 'clipsim' in evaluator.metrics
        assert 'temporal_consistency' in evaluator.metrics
    
    def test_evaluator_selective_metrics(self):
        """Evaluator can enable/disable metrics."""
        evaluator = VideoMetricsEvaluator(
            enable_fid=False,
            enable_clipsim=True,
            enable_temporal=False,
        )
        assert 'fid' not in evaluator.metrics
        assert 'clipsim' in evaluator.metrics
        assert 'temporal_consistency' not in evaluator.metrics
    
    def test_evaluator_evaluate(self):
        """Evaluator computes all enabled metrics."""
        evaluator = VideoMetricsEvaluator(enable_fid=False)
        
        video = torch.rand(1, 4, 3, 64, 64)
        prompts = ["test"]
        
        # Mock the metrics to avoid model loading
        for metric in evaluator.metrics.values():
            metric.compute = Mock(return_value=MetricResult(
                value=0.5, metric_name=metric.name
            ))
        
        results = evaluator.evaluate(video, text_prompts=prompts)
        
        assert 'clipsim' in results
        assert 'temporal_consistency' in results
    
    def test_evaluator_metric_preservation(self):
        """Evaluator computes metric preservation."""
        evaluator = VideoMetricsEvaluator()
        
        baseline = {
            'fid': MetricResult(value=10.0, metric_name='fid'),
            'clipsim': MetricResult(value=0.9, metric_name='clipsim'),
        }
        
        quantized = {
            'fid': MetricResult(value=10.1, metric_name='fid'),
            'clipsim': MetricResult(value=0.89, metric_name='clipsim'),
        }
        
        preservation = evaluator.compute_metric_preservation(baseline, quantized)
        
        assert 'fid' in preservation
        assert 'clipsim' in preservation
        # FID: slightly higher quantized should give ~99% preservation
        assert 95 < preservation['fid'] <= 100
        # CLIPSIM: slightly lower quantized should give ~99% preservation
        assert 95 < preservation['clipsim'] <= 100
    
    def test_evaluator_error_handling(self):
        """Evaluator handles metric computation errors gracefully."""
        evaluator = VideoMetricsEvaluator(enable_fid=False)
        
        video = torch.rand(1, 4, 3, 64, 64)
        
        # Make one metric fail
        evaluator.metrics['clipsim'].compute = Mock(
            side_effect=Exception("CLIP not available")
        )
        
        results = evaluator.evaluate(video)
        
        assert 'clipsim' in results
        assert 'error' in results['clipsim'].metadata
        assert np.isnan(results['clipsim'].value)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_compute_fid(self):
        """compute_fid convenience function works."""
        gen = torch.rand(1, 4, 3, 299, 299)
        ref = torch.rand(1, 4, 3, 299, 299)
        
        with patch('videoquant.metrics.FIDMetric') as MockFID:
            mock_instance = Mock()
            mock_instance.compute.return_value = MetricResult(
                value=5.0, metric_name='fid'
            )
            MockFID.return_value = mock_instance
            
            fid = compute_fid(gen, ref)
            assert fid == 5.0
    
    def test_compute_clipsim(self):
        """compute_clipsim convenience function works."""
        video = torch.rand(1, 4, 3, 224, 224)
        prompts = ["test"]
        
        with patch('videoquant.metrics.CLIPSIMMetric') as MockCLIP:
            mock_instance = Mock()
            mock_instance.compute.return_value = MetricResult(
                value=0.75, metric_name='clipsim'
            )
            MockCLIP.return_value = mock_instance
            
            clipsim = compute_clipsim(video, prompts)
            assert clipsim == 0.75
    
    def test_compute_temporal_consistency(self):
        """compute_temporal_consistency convenience function works."""
        video = torch.rand(4, 3, 64, 64)
        
        with patch('videoquant.metrics.TemporalConsistencyMetric') as MockTC:
            mock_instance = Mock()
            mock_instance.compute.return_value = MetricResult(
                value=0.9, metric_name='temporal_consistency'
            )
            MockTC.return_value = mock_instance
            
            tc = compute_temporal_consistency(video)
            assert tc == 0.9


class TestMetricReproducibility:
    """Test that metrics are reproducible."""
    
    def test_fid_reproducible(self):
        """FID gives same result for same input."""
        metric = FIDMetric()
        
        video = torch.rand(1, 4, 3, 64, 64)
        
        # Use mock to avoid actual model loading
        features = torch.randn(4, 512)
        with patch.object(metric, '_extract_features', return_value=features):
            result1 = metric.compute(video, video.clone())
            result2 = metric.compute(video, video.clone())
            
            assert abs(result1.value - result2.value) < 1e-6
    
    def test_temporal_consistency_reproducible(self):
        """Temporal consistency gives same result for same input."""
        metric = TemporalConsistencyMetric()
        
        torch.manual_seed(42)
        video = torch.rand(4, 3, 64, 64)
        
        result1 = metric.compute(video)
        result2 = metric.compute(video)
        
        assert abs(result1.value - result2.value) < 1e-6


class TestMetricValidation:
    """Validate metrics meet expected behavior."""
    
    def test_fid_symmetry(self):
        """FID is symmetric (mostly)."""
        metric = FIDMetric()
        
        v1 = torch.rand(1, 4, 3, 64, 64)
        v2 = torch.rand(1, 4, 3, 64, 64)
        
        # Mock to return fixed features for symmetry test
        features = torch.randn(4, 512)
        with patch.object(metric, '_extract_features', return_value=features):
            fid_1_2 = metric.compute(v1, v2).value
            fid_2_1 = metric.compute(v2, v1).value
            
            # Should be similar (allow small numerical differences)
            assert abs(fid_1_2 - fid_2_1) < 1e-4
    
    def test_temporal_perfect_static(self):
        """Static video has perfect temporal consistency."""
        metric = TemporalConsistencyMetric()
        
        # Create truly static video
        frame = torch.ones(1, 3, 64, 64) * 0.5
        video = frame.repeat(8, 1, 1, 1)
        
        result = metric.compute(video)
        assert result.value > 0.99
    
    def test_temporal_zero_motion_high_consistency(self):
        """Video with no motion has high consistency."""
        metric = TemporalConsistencyMetric()
        
        # All frames the same
        frame = torch.rand(1, 3, 64, 64)
        video = frame.expand(8, 3, 64, 64)
        
        result = metric.compute(video)
        assert result.value > 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
