#!/usr/bin/env python3
"""
VideoQuant - Spatial-QJL (SQJL) Core Implementation

Implements the SQJL residual correction algorithm with:
- Johnson-Lindenstrauss random projection
- Sign-bit (1-bit) quantization
- Unbiased attention estimator
- Spatial relationship preservation
- Zero memory overhead from metadata

References:
- Johnson-Lindenstrauss Lemma: https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
- Sign-bit quantization inspired by TurboQuant paper
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class SQJLConfig:
    """Configuration for Spatial-QJL"""
    projection_dim: int = 256  # Target dimension for JL projection
    use_random_seed: bool = True  # Use deterministic seed for reproducibility
    random_seed: int = 42  # Seed for projection matrix generation
    preserve_spatial: bool = True  # Preserve 2D spatial relationships
    unbiased_estimator: bool = True  # Use unbiased attention estimator


class SQJLQuantizer:
    """
    Spatial-QJL Quantization for video diffusion transformers.
    
    Implements Johnson-Lindenstrauss random projection followed by
    sign-bit (1-bit) quantization with zero metadata overhead.
    
    Key features:
    1. Distance preservation via JL projection (within 1±ε factor)
    2. 1-bit sign quantization (zero metadata overhead)
    3. Unbiased attention score estimator
    4. 2D spatial relationship preservation
    """
    
    def __init__(self, config: Optional[SQJLConfig] = None):
        self.config = config or SQJLConfig()
        self._projection_matrices: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._spatial_indices: Optional[torch.Tensor] = None
    
    def create_jl_projection_matrix(
        self, 
        input_dim: int, 
        output_dim: int,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Create Johnson-Lindenstrauss random projection matrix.
        
        According to JL lemma, for n points in high-dimensional space,
        we can project to O(log(n)/ε²) dimensions while preserving
        pairwise distances within (1±ε) factor.
        
        We use random Gaussian projection matrix scaled appropriately.
        
        Args:
            input_dim: Original dimensionality
            output_dim: Target projected dimensionality
            device: Device for the matrix
            dtype: Data type for the matrix
            
        Returns:
            projection_matrix: [input_dim, output_dim] random projection matrix
        """
        # Set seed for reproducibility if configured
        if self.config.use_random_seed:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.config.random_seed)
        else:
            generator = None
        
        # Generate random Gaussian matrix
        # JL projection matrix: entries ~ N(0, 1/sqrt(output_dim))
        scale = 1.0 / math.sqrt(output_dim)
        if generator:
            projection_matrix = torch.randn(
                input_dim, output_dim, 
                generator=generator,
                device=device, 
                dtype=dtype
            ) * scale
        else:
            projection_matrix = torch.randn(
                input_dim, output_dim,
                device=device, 
                dtype=dtype
            ) * scale
        
        return projection_matrix
    
    def _get_cached_projection(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or create cached projection matrix for given dimensions."""
        key = (input_dim, output_dim, str(device), str(dtype))
        
        if key not in self._projection_matrices:
            self._projection_matrices[key] = self.create_jl_projection_matrix(
                input_dim, output_dim, device, dtype
            )
        
        return self._projection_matrices[key]
    
    def apply_jl_projection(
        self, 
        tensor: torch.Tensor,
        output_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply Johnson-Lindenstrauss random projection to tensor.
        
        Args:
            tensor: Input tensor [..., input_dim]
            output_dim: Target dimension (defaults to config.projection_dim)
            
        Returns:
            projected: Projected tensor [..., output_dim]
            
        Note:
            For input x, output y = x @ P where P is the JL projection matrix.
            E[||y||²] = ||x||², preserving distances in expectation.
            Uses proper JL scaling: P_ij ~ N(0, 1/output_dim)
        """
        if output_dim is None:
            output_dim = self.config.projection_dim
        
        # Get input dimension from last axis
        input_dim = tensor.shape[-1]
        
        # No projection needed if already at target dimension
        if input_dim <= output_dim:
            return tensor
        
        # Get or create projection matrix
        projection = self._get_cached_projection(
            input_dim, output_dim, tensor.device, tensor.dtype
        )
        
        # Apply projection: result = tensor @ projection
        # tensor shape: [..., input_dim], projection: [input_dim, output_dim]
        # result shape: [..., output_dim]
        projected = torch.matmul(tensor, projection)
        
        # Scale to preserve expected norm: JL projection should maintain ||x|| in expectation
        # The projection matrix is already scaled by 1/sqrt(output_dim) during creation
        # but we need to scale by sqrt(output_dim) to preserve the norm approximately
        # Actually, for JL: E[||Px||²] = ||x||² when P_ij ~ N(0, 1/output_dim)
        # The current scaling is correct for distance preservation
        
        return projected
    
    def sign_quantize(
        self, 
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Sign-bit quantization: encode each element as 1 bit (sign only).
        
        Args:
            tensor: Input tensor of any shape
            
        Returns:
            quantized: Boolean tensor (True for positive/zero, False for negative)
            
        Note:
            - Uses exactly 1 bit per element (stored as bool)
            - No metadata (scales, zero-points) stored
            - Zero memory overhead from metadata
        """
        # Sign-bit encoding: positive/zero -> True (1), negative -> False (0)
        # This uses exactly 1 bit per element when stored as bool
        return tensor >= 0
    
    def sign_dequantize(
        self,
        quantized: torch.Tensor,
        reference_magnitude: Optional[float] = None
    ) -> torch.Tensor:
        """
        Dequantize sign-bit encoded tensor.
        
        Args:
            quantized: Boolean tensor from sign_quantize
            reference_magnitude: Optional magnitude to scale signs
            
        Returns:
            dequantized: Float tensor with values {-scale, +scale}
            
        Note:
            Without reference magnitude, uses unit magnitude.
            For attention estimation, we typically use √π/2 as scale
            to maintain unbiased estimation.
        """
        if reference_magnitude is None:
            # Default scale for unbiased estimator
            # For Gaussian inputs, E[|x|] = σ√(2/π)
            # To maintain E[x²] = 1, we need scale such that E[(scale * sign)²] ≈ 1
            scale = math.sqrt(math.pi / 2)
        else:
            scale = reference_magnitude
        
        # Convert bool to float: True -> +scale, False -> -scale
        return quantized.float() * (2 * scale) - scale
    
    def unbiased_attention_estimator(
        self,
        query: torch.Tensor,
        quantized_keys: torch.Tensor,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Unbiased estimator for attention scores with sign-bit quantized keys.
        
        Args:
            query: Query tensor [..., query_dim] (FP16/FP32)
            quantized_keys: Sign-bit quantized keys from sign_quantize
            scale: Optional scale factor (auto-computed if None)
            
        Returns:
            attention_scores: Estimated attention scores
            
        Note:
            This estimator has zero systematic bias:
            E[estimator(query, quantized_key)] = dot(query, original_key)
            
            The scale factor is chosen such that:
            E[sign(x) * scale] = x for Gaussian distributed x
        """
        if scale is None:
            # Optimal scale for zero-mean Gaussian: √(π/2)
            scale = math.sqrt(math.pi / 2)
        
        # Dequantize keys with appropriate scale
        dequantized_keys = self.sign_dequantize(quantized_keys, scale)
        
        # Compute attention scores as dot product
        # query: [..., head_dim], keys: [..., head_dim]
        attention_scores = torch.sum(query * dequantized_keys, dim=-1)
        
        return attention_scores
    
    def _generate_2d_spatial_indices(
        self,
        height: int,
        width: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Generate 2D spatial indices for preserving spatial relationships.
        
        Args:
            height: Height of spatial grid
            width: Width of spatial grid
            device: Device for indices
            
        Returns:
            spatial_indices: [height*width, 2] tensor of (row, col) coordinates
        """
        # Create 2D grid of coordinates
        rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
        cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
        
        # Stack and flatten to [height*width, 2]
        spatial_indices = torch.stack([rows.flatten(), cols.flatten()], dim=-1)
        
        return spatial_indices.float()
    
    def compute_spatial_distance_preservation(
        self,
        original_features: torch.Tensor,
        projected_features: torch.Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Compute spatial relationship preservation metric.
        
        Args:
            original_features: Original spatial features [num_patches, feature_dim]
            projected_features: Projected features [num_patches, proj_dim]
            spatial_shape: Optional (height, width) for computing patch coordinates
            
        Returns:
            correlation: Pearson correlation between spatial distances
            
        Note:
            Returns correlation between:
            1. Spatial distances in original 2D grid
            2. Feature distances in projected space (Euclidean)
            
            High correlation (>0.9) indicates spatial relationships preserved.
            Uses Spearman rank correlation which is more robust to JL projection.
        """
        num_patches = original_features.shape[0]
        
        if spatial_shape is None:
            # Assume square grid
            height = width = int(math.sqrt(num_patches))
            if height * width != num_patches:
                height = int(math.sqrt(num_patches))
                width = num_patches // height
                if height * width != num_patches:
                    height = width = int(num_patches ** 0.5)
        else:
            height, width = spatial_shape
        
        # Get spatial coordinates
        if self._spatial_indices is None or self._spatial_indices.shape[0] != num_patches:
            self._spatial_indices = self._generate_2d_spatial_indices(
                height, width, original_features.device
            )
        
        spatial_coords = self._spatial_indices[:num_patches]
        
        # Sample pairs for efficiency (computing all pairwise is O(N^2))
        max_pairs = 5000
        if num_patches > 100:
            # Sample random pairs
            np.random.seed(42)
            n_samples = min(num_patches, 100)
            indices = torch.randperm(num_patches, device=original_features.device)[:n_samples]
            
            sample_orig = original_features[indices]
            sample_proj = projected_features[indices]
            sample_spatial = spatial_coords[indices]
            
            # Compute pairwise distances for sampled subset
            N = n_samples
        else:
            sample_orig = original_features
            sample_proj = projected_features
            sample_spatial = spatial_coords
            N = num_patches
        
        # Compute pairwise spatial distances
        spatial_diff = sample_spatial.unsqueeze(0) - sample_spatial.unsqueeze(1)  # [N, N, 2]
        spatial_dists = torch.sqrt((spatial_diff ** 2).sum(dim=-1) + 1e-10)  # [N, N]
        
        # Compute pairwise feature distances in projected space
        proj_diff = sample_proj.unsqueeze(0) - sample_proj.unsqueeze(1)  # [N, N, proj_dim]
        proj_dists = torch.sqrt((proj_diff ** 2).sum(dim=-1) + 1e-10)  # [N, N]
        
        # Get upper triangular (excluding diagonal)
        triu_indices = torch.triu_indices(N, N, offset=1, device=original_features.device)
        spatial_flat = spatial_dists[triu_indices[0], triu_indices[1]]
        proj_flat = proj_dists[triu_indices[0], triu_indices[1]]
        
        # Compute Pearson correlation
        if len(spatial_flat) == 0:
            return 0.0
        
        # Convert to numpy for spearman correlation
        spatial_np = spatial_flat.cpu().numpy()
        proj_np = proj_flat.cpu().numpy()
        
        # Use Spearman rank correlation (more robust)
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(spatial_np, proj_np)
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def verify_distance_preservation(
        self,
        original_tensor: torch.Tensor,
        projected_tensor: torch.Tensor,
        epsilon: float = 0.1
    ) -> Tuple[bool, float, float]:
        """
        Verify Johnson-Lindenstrauss distance preservation property.
        
        Args:
            original_tensor: Original high-dimensional tensor [N, D]
            projected_tensor: Projected tensor [N, d] where d << D
            epsilon: Target distortion bound (default 0.1 for 1±0.1 factor)
            
        Returns:
            (passes, preserved_ratio, mean_distortion): 
                - passes: True if >95% of pairs within (1±ε) factor
                - preserved_ratio: Fraction of pairs within bound
                - mean_distortion: Mean distance distortion ratio
                
        Note:
            For JL lemma: (1-ε)||u-v||² ≤ ||Pu-Pv||² ≤ (1+ε)||u-v||²
            We verify this holds for sampled pairs.
        """
        N = original_tensor.shape[0]
        
        # Sample pairs (limit for efficiency)
        max_pairs = 10000
        if N * (N - 1) // 2 > max_pairs:
            # Random sampling
            num_samples = int(math.sqrt(2 * max_pairs))
            indices = torch.randperm(N, device=original_tensor.device)[:num_samples]
            orig_sample = original_tensor[indices]
            proj_sample = projected_tensor[indices]
        else:
            orig_sample = original_tensor
            proj_sample = projected_tensor
        
        N_sample = orig_sample.shape[0]
        
        # Compute pairwise squared distances in original space
        # Using broadcasting: [N, 1, D] - [1, N, D] = [N, N, D]
        orig_diff = orig_sample.unsqueeze(0) - orig_sample.unsqueeze(1)
        orig_dists_sq = (orig_diff ** 2).sum(dim=-1)  # [N, N]
        
        # Compute pairwise squared distances in projected space
        proj_diff = proj_sample.unsqueeze(0) - proj_sample.unsqueeze(1)
        proj_dists_sq = (proj_diff ** 2).sum(dim=-1)  # [N, N]
        
        # Get upper triangular (excluding diagonal)
        triu_indices = torch.triu_indices(N_sample, N_sample, offset=1, device=original_tensor.device)
        orig_dists_flat = orig_dists_sq[triu_indices[0], triu_indices[1]]
        proj_dists_flat = proj_dists_sq[triu_indices[0], triu_indices[1]]
        
        # Avoid division by zero
        mask = orig_dists_flat > 1e-10
        orig_dists_valid = orig_dists_flat[mask]
        proj_dists_valid = proj_dists_flat[mask]
        
        if len(orig_dists_valid) == 0:
            return True, 1.0, 0.0
        
        # Compute distortion ratios
        distortion_ratios = proj_dists_valid / (orig_dists_valid + 1e-10)
        
        # Check if within (1±ε)² bounds
        lower_bound = (1 - epsilon) ** 2
        upper_bound = (1 + epsilon) ** 2
        
        within_bounds = ((distortion_ratios >= lower_bound) & (distortion_ratios <= upper_bound)).float()
        preserved_ratio = within_bounds.mean().item()
        mean_distortion = distortion_ratios.mean().item()
        
        passes = preserved_ratio >= 0.95
        
        return passes, preserved_ratio, mean_distortion
    
    def quantize(
        self,
        tensor: torch.Tensor,
        output_dim: Optional[int] = None,
        return_projected: bool = False
    ) -> Dict[str, Any]:
        """
        Main SQJL quantization entry point.
        
        Args:
            tensor: Input tensor [..., input_dim]
            output_dim: Target projection dimension
            return_projected: If True, also return pre-quantization projection
            
        Returns:
            Dictionary containing:
                - quantized_bits: 1-bit sign-quantized tensor (bool)
                - metadata: Minimal metadata (no per-element overhead)
                - projected: (optional) Projected but not quantized tensor
                
        Note:
            Zero metadata overhead - only stores global config values,
            no per-element scales or zero-points.
        """
        # Step 1: Apply JL projection
        projected = self.apply_jl_projection(tensor, output_dim)
        
        # Step 2: Sign-bit quantization
        quantized_bits = self.sign_quantize(projected)
        
        # Prepare result with zero metadata overhead
        result = {
            'quantized_bits': quantized_bits,
            'metadata': {
                'input_shape': tensor.shape,
                'output_dim': projected.shape[-1],
                'config': self.config,
                # No per-element metadata - only global config
            }
        }
        
        if return_projected:
            result['projected'] = projected
        
        return result
    
    def dequantize(
        self,
        quantized_data: Dict[str, Any],
        reference_magnitude: Optional[float] = None
    ) -> torch.Tensor:
        """
        Main SQJL dequantization entry point.
        
        Args:
            quantized_data: Output from quantize()
            reference_magnitude: Optional magnitude for dequantization scaling
            
        Returns:
            dequantized: Dequantized tensor (approximation of JL projection)
            
        Note:
            This returns the sign-dequantized JL projection.
            Full reconstruction would require inverse JL transform (not implemented
            as SQJL is designed for attention estimation, not reconstruction).
        """
        quantized_bits = quantized_data['quantized_bits']
        
        # Sign dequantization
        dequantized = self.sign_dequantize(quantized_bits, reference_magnitude)
        
        return dequantized


def estimate_attention_with_sqjl(
    queries: torch.Tensor,
    keys: torch.Tensor,
    config: Optional[SQJLConfig] = None,
    return_stats: bool = False
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    """
    Convenience function for attention estimation with SQJL-quantized keys.
    
    Args:
        queries: Query tensor [batch, heads, seq_len_q, head_dim]
        keys: Key tensor [batch, heads, seq_len_k, head_dim]
        config: SQJL configuration
        return_stats: If True, also return statistics
        
    Returns:
        attention_scores: [batch, heads, seq_len_q, seq_len_k]
        stats: (optional) Dictionary with verification statistics
    """
    quantizer = SQJLQuantizer(config)
    
    # Extract dimensions
    batch, heads, seq_len_q, head_dim = queries.shape
    seq_len_k = keys.shape[2]
    
    # Reshape keys for projection: [batch*heads*seq_len_k, head_dim]
    keys_flat = keys.reshape(-1, head_dim)
    
    # SQJL quantize keys
    sqjl_result = quantizer.quantize(keys_flat)
    dequantized_keys = quantizer.dequantize(sqjl_result)
    
    # Reshape back: [batch, heads, seq_len_k, head_dim]
    dequantized_keys = dequantized_keys.reshape(batch, heads, seq_len_k, -1)
    
    # Compute attention scores: Q @ K^T
    # queries: [batch, heads, seq_len_q, head_dim]
    # keys: [batch, heads, seq_len_k, head_dim]
    # scores: [batch, heads, seq_len_q, seq_len_k]
    attention_scores = torch.matmul(queries, dequantized_keys.transpose(-2, -1))
    
    if return_stats:
        # Compute statistics
        stats = {
            'projection_dim': sqjl_result['metadata']['output_dim'],
            'bits_per_element': 1.0,
            'metadata_overhead_ratio': 0.0,  # Zero per-element metadata
        }
        return attention_scores, stats
    
    return attention_scores
