"""
Tests for Metric-Aware Mixed Precision (MAMP) core algorithm.

Validates:
- VAL-MAMP-001: Layer-Type Precision Assignment
- VAL-MAMP-002: Timestep-Aware Allocation
- VAL-MAMP-004: Cross-Attention High Precision
- VAL-MAMP-005: Temporal Consistency Optimization
"""

import torch
import numpy as np
import pytest
from videoquant.core.mamp import (
    MAMPAllocator,
    MAMPQuantizer,
    MAMPConfig,
    LayerType,
    create_default_mamp_config,
    get_precision_for_layer,
)


class TestLayerPrecisionAssignment:
    """VAL-MAMP-001: Layer-Type Precision Assignment"""
    
    def test_cross_attention_gets_highest_precision(self):
        """Cross-attention should receive 6 bits (highest)."""
        allocator = MAMPAllocator()
        
        bits = allocator.get_base_bits(LayerType.CROSS_ATTENTION)
        assert bits == 6, f"Cross-attention should have 6 bits, got {bits}"
    
    def test_temporal_attention_gets_5_bits(self):
        """Temporal-attention should receive 5 bits."""
        allocator = MAMPAllocator()
        
        bits = allocator.get_base_bits(LayerType.TEMPORAL_ATTENTION)
        assert bits == 5, f"Temporal-attention should have 5 bits, got {bits}"
    
    def test_self_attention_gets_4_bits(self):
        """Self-attention should receive 4 bits."""
        allocator = MAMPAllocator()
        
        bits = allocator.get_base_bits(LayerType.SELF_ATTENTION)
        assert bits == 4, f"Self-attention should have 4 bits, got {bits}"
    
    def test_ffn_gets_3_bits(self):
        """FFN should receive 3 bits (lowest)."""
        allocator = MAMPAllocator()
        
        bits = allocator.get_base_bits(LayerType.FFN)
        assert bits == 3, f"FFN should have 3 bits, got {bits}"
    
    def test_layer_precision_hierarchy(self):
        """Verify precision hierarchy: cross > temporal > self > ffn."""
        allocator = MAMPAllocator()
        
        cross_bits = allocator.get_base_bits(LayerType.CROSS_ATTENTION)
        temporal_bits = allocator.get_base_bits(LayerType.TEMPORAL_ATTENTION)
        self_bits = allocator.get_base_bits(LayerType.SELF_ATTENTION)
        ffn_bits = allocator.get_base_bits(LayerType.FFN)
        
        assert cross_bits >= temporal_bits, \
            f"Cross ({cross_bits}) should be >= temporal ({temporal_bits})"
        assert temporal_bits >= self_bits, \
            f"Temporal ({temporal_bits}) should be >= self ({self_bits})"
        assert self_bits >= ffn_bits, \
            f"Self ({self_bits}) should be >= ffn ({ffn_bits})"
    
    def test_default_config_matches_specification(self):
        """Default config should match validation contract specifications."""
        config = create_default_mamp_config()
        
        assert config.cross_attention_bits == 6, "Cross-attention must be 6 bits"
        assert config.temporal_attention_bits == 5, "Temporal-attention must be 5 bits"
        assert config.self_attention_bits == 4, "Self-attention must be 4 bits"
        assert config.ffn_bits == 3, "FFN must be 3 bits"
    
    def test_allocate_all_layers_returns_all_types(self):
        """allocate_all_layers() should return allocations for all layer types."""
        allocator = MAMPAllocator()
        allocations = allocator.allocate_all_layers(timestep=0.5)
        
        expected_types = ["cross_attention", "temporal_attention", "self_attention", "ffn"]
        for layer_type in expected_types:
            assert layer_type in allocations, f"Missing allocation for {layer_type}"
            assert isinstance(allocations[layer_type], int), \
                f"Allocation for {layer_type} should be an integer"
            assert 2 <= allocations[layer_type] <= 8, \
                f"Allocation for {layer_type} should be in valid range [2, 8]"
    
    def test_string_layer_type_lookup(self):
        """Should accept string layer type names."""
        allocator = MAMPAllocator()
        
        # Test with string names
        assert allocator.get_base_bits("cross_attention") == 6
        assert allocator.get_base_bits("temporal_attention") == 5
        assert allocator.get_base_bits("self_attention") == 4
        assert allocator.get_base_bits("ffn") == 3


class TestTimestepAwareAllocation:
    """VAL-MAMP-002: Timestep-Aware Allocation"""
    
    def test_early_timestep_scale_is_1_0(self):
        """Early timestep (t=0) should have scale = 1.0."""
        allocator = MAMPAllocator()
        
        scale = allocator.compute_timestep_scale(0.0)
        assert abs(scale - 1.0) < 0.01, f"Early timestep scale should be 1.0, got {scale}"
    
    def test_late_timestep_scale_is_1_3(self):
        """Late timestep (t=1) should have scale = 1.3."""
        allocator = MAMPAllocator()
        
        scale = allocator.compute_timestep_scale(1.0)
        assert abs(scale - 1.3) < 0.01, f"Late timestep scale should be 1.3, got {scale}"
    
    def test_timestep_scale_increases_monotonically(self):
        """Scale should increase monotonically from early to late timesteps."""
        allocator = MAMPAllocator()
        
        timesteps = np.linspace(0.0, 1.0, 21)
        scales = [allocator.compute_timestep_scale(t) for t in timesteps]
        
        # Check monotonic increase
        for i in range(len(scales) - 1):
            assert scales[i] <= scales[i + 1] + 1e-6, \
                f"Scale should increase monotonically: {scales[i]} > {scales[i+1]} at t={timesteps[i]}"
    
    def test_midpoint_timestep_scale_is_midpoint(self):
        """Midpoint timestep (t=0.5) should have scale approximately at midpoint."""
        allocator = MAMPAllocator()
        
        scale = allocator.compute_timestep_scale(0.5)
        expected_midpoint = (1.0 + 1.3) / 2  # 1.15
        assert abs(scale - expected_midpoint) < 0.05, \
            f"Midpoint scale should be ~{expected_midpoint}, got {scale}"
    
    def test_timestep_precision_allocation_increases(self):
        """Allocated bits should increase from early to late timesteps."""
        allocator = MAMPAllocator()
        
        # Test for each layer type
        for layer_type in LayerType:
            early_bits = allocator.allocate_precision(layer_type, timestep=0.0)
            late_bits = allocator.allocate_precision(layer_type, timestep=1.0)
            
            assert late_bits >= early_bits, \
                f"{layer_type.value}: Late bits ({late_bits}) should be >= early bits ({early_bits})"
    
    def test_timestep_clamping(self):
        """Timesteps outside [0,1] should be clamped."""
        allocator = MAMPAllocator()
        
        # Below 0 should be clamped to 0
        scale_low = allocator.compute_timestep_scale(-0.5)
        scale_zero = allocator.compute_timestep_scale(0.0)
        assert abs(scale_low - scale_zero) < 0.01, "Negative timesteps should be clamped to 0"
        
        # Above 1 should be clamped to 1
        scale_high = allocator.compute_timestep_scale(1.5)
        scale_one = allocator.compute_timestep_scale(1.0)
        assert abs(scale_high - scale_one) < 0.01, "Timesteps > 1 should be clamped to 1"
    
    def test_sigmoid_schedule_smooth_transition(self):
        """Sigmoid schedule should provide smooth transition."""
        config = MAMPConfig(timestep_schedule="sigmoid")
        allocator = MAMPAllocator(config)
        
        timesteps = np.linspace(0.0, 1.0, 11)
        scales = [allocator.compute_timestep_scale(t) for t in timesteps]
        
        # Should still start near 1.0 and end near 1.3
        assert abs(scales[0] - 1.0) < 0.15, "Sigmoid early scale should be near 1.0"
        assert abs(scales[-1] - 1.3) < 0.15, "Sigmoid late scale should be near 1.3"
        
        # Should have steepest change near center
        diffs = [scales[i+1] - scales[i] for i in range(len(scales)-1)]
        max_diff_idx = diffs.index(max(diffs))
        assert 4 <= max_diff_idx <= 6, "Steepest change should be near center (t=0.5)"


class TestCrossAttentionHighPrecision:
    """VAL-MAMP-004: Cross-Attention High Precision Protection"""
    
    def test_cross_attention_minimum_6_bits(self):
        """Cross-attention should never go below 6 bits at any timestep."""
        allocator = MAMPAllocator()
        
        for t in np.linspace(0.0, 1.0, 11):
            bits = allocator.allocate_precision(LayerType.CROSS_ATTENTION, t)
            assert bits >= 6, f"Cross-attention bits ({bits}) below minimum 6 at t={t}"
    
    def test_cross_attention_always_highest_precision(self):
        """Cross-attention should always have highest or equal precision among all layers."""
        allocator = MAMPAllocator()
        
        for t in np.linspace(0.0, 1.0, 11):
            cross_bits = allocator.allocate_precision(LayerType.CROSS_ATTENTION, t)
            
            for layer_type in [LayerType.TEMPORAL_ATTENTION, LayerType.SELF_ATTENTION, LayerType.FFN]:
                layer_bits = allocator.allocate_precision(layer_type, t)
                assert cross_bits >= layer_bits, \
                    f"At t={t}: Cross ({cross_bits}) should be >= {layer_type.value} ({layer_bits})"
    
    def test_cross_attention_sensitivity_text_alignment(self):
        """Cross-attention should have highest text alignment sensitivity."""
        allocator = MAMPAllocator()
        
        cross_sens = allocator.get_layer_sensitivity(LayerType.CROSS_ATTENTION)
        temporal_sens = allocator.get_layer_sensitivity(LayerType.TEMPORAL_ATTENTION)
        self_sens = allocator.get_layer_sensitivity(LayerType.SELF_ATTENTION)
        ffn_sens = allocator.get_layer_sensitivity(LayerType.FFN)
        
        assert cross_sens["text_alignment"] > temporal_sens["text_alignment"], \
            "Cross-attention should have higher text alignment sensitivity than temporal"
        assert cross_sens["text_alignment"] > self_sens["text_alignment"], \
            "Cross-attention should have higher text alignment sensitivity than self"
        assert cross_sens["text_alignment"] > ffn_sens["text_alignment"], \
            "Cross-attention should have higher text alignment sensitivity than ffn"
    
    def test_cross_attention_late_timestep_at_least_7_bits(self):
        """At late timesteps, cross-attention should have at least 7 bits (6 * 1.3 ≈ 7.8)."""
        allocator = MAMPAllocator()
        
        bits = allocator.allocate_precision(LayerType.CROSS_ATTENTION, timestep=1.0)
        assert bits >= 7, f"Late timestep cross-attention should have >= 7 bits, got {bits}"


class TestTemporalConsistencyOptimization:
    """VAL-MAMP-005: Temporal Consistency Optimization"""
    
    def test_temporal_attention_minimum_5_bits(self):
        """Temporal-attention should have at least 5 bits at any timestep."""
        allocator = MAMPAllocator()
        
        for t in np.linspace(0.0, 1.0, 11):
            bits = allocator.allocate_precision(LayerType.TEMPORAL_ATTENTION, t)
            assert bits >= 5, f"Temporal-attention bits ({bits}) below minimum 5 at t={t}"
    
    def test_temporal_attention_higher_than_self(self):
        """Temporal-attention should always have >= bits than self-attention."""
        allocator = MAMPAllocator()
        
        for t in np.linspace(0.0, 1.0, 11):
            temporal_bits = allocator.allocate_precision(LayerType.TEMPORAL_ATTENTION, t)
            self_bits = allocator.allocate_precision(LayerType.SELF_ATTENTION, t)
            assert temporal_bits >= self_bits, \
                f"At t={t}: Temporal ({temporal_bits}) should be >= self ({self_bits})"
    
    def test_temporal_attention_higher_than_ffn(self):
        """Temporal-attention should always have >= bits than FFN."""
        allocator = MAMPAllocator()
        
        for t in np.linspace(0.0, 1.0, 11):
            temporal_bits = allocator.allocate_precision(LayerType.TEMPORAL_ATTENTION, t)
            ffn_bits = allocator.allocate_precision(LayerType.FFN, t)
            assert temporal_bits >= ffn_bits, \
                f"At t={t}: Temporal ({temporal_bits}) should be >= ffn ({ffn_bits})"
    
    def test_temporal_attention_sensitivity_highest_for_consistency(self):
        """Temporal-attention should have highest temporal consistency sensitivity."""
        allocator = MAMPAllocator()
        
        temporal_sens = allocator.get_layer_sensitivity(LayerType.TEMPORAL_ATTENTION)
        cross_sens = allocator.get_layer_sensitivity(LayerType.CROSS_ATTENTION)
        self_sens = allocator.get_layer_sensitivity(LayerType.SELF_ATTENTION)
        ffn_sens = allocator.get_layer_sensitivity(LayerType.FFN)
        
        assert temporal_sens["temporal_consistency"] > cross_sens["temporal_consistency"], \
            "Temporal-attention should have higher consistency sensitivity than cross"
        assert temporal_sens["temporal_consistency"] > self_sens["temporal_consistency"], \
            "Temporal-attention should have higher consistency sensitivity than self"
        assert temporal_sens["temporal_consistency"] > ffn_sens["temporal_consistency"], \
            "Temporal-attention should have higher consistency sensitivity than ffn"


class TestMAMPConfigValidation:
    """MAMP configuration validation tests."""
    
    def test_invalid_cross_attention_bits_raises_error(self):
        """Config with cross < temporal should raise error."""
        config = MAMPConfig(cross_attention_bits=4, temporal_attention_bits=5)
        
        with pytest.raises(AssertionError):
            MAMPAllocator(config)
    
    def test_invalid_timestep_scale_order_raises_error(self):
        """Config with late < early scale should raise error."""
        config = MAMPConfig(timestep_scale_early=1.3, timestep_scale_late=1.0)
        
        with pytest.raises(AssertionError):
            MAMPAllocator(config)
    
    def test_invalid_min_max_bits_raises_error(self):
        """Config with min > max bits should raise error."""
        config = MAMPConfig(min_bits=8, max_bits=4)
        
        with pytest.raises(AssertionError):
            MAMPAllocator(config)
    
    def test_custom_config_layer_bits(self):
        """Custom config should respect provided bit allocations."""
        config = MAMPConfig(
            cross_attention_bits=7,
            temporal_attention_bits=6,
            self_attention_bits=5,
            ffn_bits=4,
        )
        allocator = MAMPAllocator(config)
        
        assert allocator.get_base_bits(LayerType.CROSS_ATTENTION) == 7
        assert allocator.get_base_bits(LayerType.TEMPORAL_ATTENTION) == 6
        assert allocator.get_base_bits(LayerType.SELF_ATTENTION) == 5
        assert allocator.get_base_bits(LayerType.FFN) == 4


class TestMAMPQuantizer:
    """MAMPQuantizer high-level interface tests."""
    
    def test_quantizer_returns_correct_bits(self):
        """Quantizer should return correct bit allocation."""
        quantizer = MAMPQuantizer()
        
        tensor = torch.randn(2, 16, 256, 512)
        # Use timestep=0.0 to get base bits without scaling
        result = quantizer.quantize_for_layer(tensor, LayerType.CROSS_ATTENTION, timestep=0.0)
        
        assert result["bits"] == 6, f"Expected 6 bits for cross-attention, got {result['bits']}"
        assert result["layer_type"] == "cross_attention"
    
    def test_quantizer_returns_layer_profile(self):
        """get_layer_profile() should return complete allocation info."""
        quantizer = MAMPQuantizer()
        
        profile = quantizer.get_layer_profile(timestep=0.5)
        
        assert "allocations" in profile
        assert "metric_preservation" in profile
        assert "meets_targets" in profile
        assert profile["timestep"] == 0.5
        
        # Check all layer types present
        for layer_type in LayerType:
            assert layer_type.value in profile["allocations"]
    
    def test_get_precision_for_layer_convenience_function(self):
        """Convenience function should return correct precision."""
        
        # Test each layer type at timestep=0.0 (base precision)
        assert get_precision_for_layer("cross_attention", 0.0) == 6
        assert get_precision_for_layer("temporal_attention", 0.0) == 5
        assert get_precision_for_layer("self_attention", 0.0) == 4
        assert get_precision_for_layer("ffn", 0.0) == 3


class TestMetricPreservation:
    """Metric preservation computation and validation."""
    
    def test_compute_metric_impact_returns_valid_ratios(self):
        """Metric impact should return preservation ratios in [0, 1]."""
        allocator = MAMPAllocator()
        
        impacts = allocator.compute_metric_impact(LayerType.CROSS_ATTENTION, bits=6)
        
        for metric, ratio in impacts.items():
            assert 0.0 <= ratio <= 1.0, \
                f"Metric {metric} ratio {ratio} not in [0, 1]"
    
    def test_higher_bits_better_preservation(self):
        """Higher bits should give better metric preservation."""
        allocator = MAMPAllocator()
        
        impacts_low = allocator.compute_metric_impact(LayerType.SELF_ATTENTION, bits=2)
        impacts_high = allocator.compute_metric_impact(LayerType.SELF_ATTENTION, bits=6)
        
        # Each metric should be better preserved with higher bits
        for metric in impacts_low:
            assert impacts_high[metric] >= impacts_low[metric] - 0.01, \
                f"Higher bits should give better preservation for {metric}"
    
    def test_verify_metric_preservation_returns_boolean(self):
        """Verification should return boolean pass/fail."""
        allocator = MAMPAllocator()
        
        allocations = allocator.allocate_all_layers(timestep=0.5)
        passes, scores = allocator.verify_metric_preservation(allocations)
        
        assert isinstance(passes, bool)
        assert isinstance(scores, dict)
        assert len(scores) > 0


class TestAllocationTable:
    """Allocation table generation tests."""
    
    def test_get_allocation_table_structure(self):
        """Allocation table should have correct nested structure."""
        allocator = MAMPAllocator()
        
        table = allocator.get_allocation_table(timesteps=[0.0, 0.5, 1.0])
        
        # Should have entry for each layer type
        for layer_type in LayerType:
            assert layer_type.value in table
            # Each should have entries for timesteps
            assert 0.0 in table[layer_type.value]
            assert 0.5 in table[layer_type.value]
            assert 1.0 in table[layer_type.value]
    
    def test_allocation_table_shows_timestep_progression(self):
        """Table should show increasing bits with timestep."""
        allocator = MAMPAllocator()
        
        table = allocator.get_allocation_table(timesteps=[0.0, 0.5, 1.0])
        
        for layer_type in LayerType:
            bits_early = table[layer_type.value][0.0]
            bits_late = table[layer_type.value][1.0]
            assert bits_late >= bits_early, \
                f"{layer_type.value}: bits should increase from early ({bits_early}) to late ({bits_late})"


class TestEdgeCases:
    """Edge case and boundary tests."""
    
    def test_extreme_timesteps(self):
        """Test with extreme timestep values."""
        allocator = MAMPAllocator()
        
        # Very early (should be clamped to 0)
        bits_early = allocator.allocate_precision(LayerType.CROSS_ATTENTION, timestep=-100)
        assert bits_early == allocator.allocate_precision(LayerType.CROSS_ATTENTION, timestep=0.0)
        
        # Very late (should be clamped to 1)
        bits_late = allocator.allocate_precision(LayerType.CROSS_ATTENTION, timestep=100)
        assert bits_late == allocator.allocate_precision(LayerType.CROSS_ATTENTION, timestep=1.0)
    
    def test_bits_clamping_at_extreme_scales(self):
        """Bits should be clamped to min/max even with extreme scales."""
        config = MAMPConfig(
            cross_attention_bits=6,
            min_bits=4,
            max_bits=8,
        )
        allocator = MAMPAllocator(config)
        
        # Even with extreme early scale, should respect min_bits
        bits = allocator.allocate_precision(LayerType.CROSS_ATTENTION, timestep=0.0)
        assert bits >= config.min_bits
        
        # Even with extreme late scale, should respect max_bits  
        bits = allocator.allocate_precision(LayerType.CROSS_ATTENTION, timestep=1.0)
        assert bits <= config.max_bits
    
    def test_convenience_function_with_various_timesteps(self):
        """get_precision_for_layer should work with various timesteps."""
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            bits = get_precision_for_layer("cross_attention", t)
            assert isinstance(bits, int)
            assert 2 <= bits <= 16
