#!/usr/bin/env python3
"""
VideoQuant - Temporal-Polar Quantization (TPQ) Core Implementation

Implements the TPQ algorithm with:
- Cartesian to polar coordinate transformation
- Recursive polar compression
- Adaptive bit allocation (60/40 radii/angle split)
- Tensor packing and unpacking
"""

import torch
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class TPQConfig:
    """Configuration for Temporal-Polar Quantization"""
    target_bits: float = 3.5  # Target average bits per element
    radii_allocation: float = 0.6  # Fraction of bits for radii (60%)
    angle_allocation: float = 0.4  # Fraction of bits for angles (40%)
    temporal_group_size: int = 2  # Group consecutive frames in pairs
    enable_recursive: bool = True  # Enable recursive polar compression
    max_recursive_levels: int = 6  # Maximum recursion depth for channels (enough for 64 channels)


class TPQQuantizer:
    """
    Temporal-Polar Quantization for video diffusion transformers.
    
    Implements TurboQuant-inspired polar coordinate quantization
    adapted for temporal video data.
    """
    
    def __init__(self, config: Optional[TPQConfig] = None):
        self.config = config or TPQConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        assert 0 < self.config.radii_allocation < 1, "radii_allocation must be in (0, 1)"
        assert abs(self.config.radii_allocation + self.config.angle_allocation - 1.0) < 1e-6, \
            "radii_allocation + angle_allocation must equal 1.0"
        assert self.config.temporal_group_size == 2, "temporal_group_size must be 2 for polar transform"
        assert self.config.target_bits > 0, "target_bits must be positive"
    
    def cartesian_to_polar(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates (x, y) to polar coordinates (radius, angle).
        
        Args:
            x: First component [B, F, N, C] or any shape
            y: Second component [B, F, N, C] or any shape
            
        Returns:
            radius: Magnitude of vector (>= 0)
            angle: Direction of vector in [-pi, pi]
            
        Note:
            radius = sqrt(x^2 + y^2)
            angle = atan2(y, x)  # PyTorch's atan2 preserves quadrant
        """
        # Compute radius: sqrt(x^2 + y^2)
        radius = torch.sqrt(x**2 + y**2 + 1e-10)  # Add epsilon for numerical stability
        
        # Compute angle: atan2(y, x) preserves quadrant information
        angle = torch.atan2(y, x)
        
        return radius, angle
    
    def polar_to_cartesian(self, radius: torch.Tensor, angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert polar coordinates (radius, angle) back to Cartesian (x, y).
        
        Args:
            radius: Magnitude of vector (>= 0)
            angle: Direction of vector in radians
            
        Returns:
            x: First component
            y: Second component
        """
        x = radius * torch.cos(angle)
        y = radius * torch.sin(angle)
        return x, y
    
    def _group_temporal_pairs(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Group consecutive frames in pairs for polar transformation.
        
        Input shape: [B, F, N, C] where F must be even
        Output shape: [B, F/2, N, C, 2] where last dim is (frame_t, frame_{t+1})
        """
        B, F, N, C = tensor.shape
        assert F % 2 == 0, f"Number of frames must be even, got {F}"
        
        # Reshape to [B, F/2, 2, N, C] then permute to [B, F/2, N, C, 2]
        grouped = tensor.view(B, F // 2, 2, N, C)
        return grouped.permute(0, 1, 3, 4, 2)  # [B, F/2, N, C, 2]
    
    def _ungroup_temporal_pairs(self, paired_tensor: torch.Tensor, original_frames: int) -> torch.Tensor:
        """
        Ungroup temporal pairs back to original frame structure.
        
        Input shape: [B, F/2, N, C, 2] 
        Output shape: [B, F, N, C]
        """
        B, F_half, N, C, _ = paired_tensor.shape
        # Permute back to [B, F/2, 2, N, C] then reshape to [B, F, N, C]
        tensor = paired_tensor.permute(0, 1, 4, 2, 3)  # [B, F/2, 2, N, C]
        # Flatten the frame dimensions: [B, F_half*2, N, C]
        return tensor.reshape(B, F_half * 2, N, C)
    
    def recursive_polar_transform(
        self, 
        radii: torch.Tensor,
        angles: torch.Tensor,
        level: int = 0,
        max_levels: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply recursive polar transformation to radii at each level.
        
        Args:
            radii: Current radii tensor [B, F/2, N, C] or similar
            angles: Accumulated angles from previous levels (temporal angles at start)
            level: Current recursion level
            max_levels: Maximum number of recursion levels (auto-computed if None)
            
        Returns:
            final_radius: Single radius value after all recursions
            all_angles: Concatenated angles from all levels including temporal
            
        Note:
            Each level halves the radii dimension and produces C/2 angles.
            For C channels (power of 2), we apply log2(C) levels.
        """
        *batch_dims, C = radii.shape
        
        # Auto-compute max levels based on input channels
        if max_levels is None:
            if C > 1 and (C & (C - 1)) == 0:
                # Power of 2: go all the way to 1
                max_levels = int(math.log2(C))
            else:
                max_levels = self.config.max_recursive_levels
        
        # Base case: single channel, no more recursion possible, or reached max levels
        if C <= 1 or level >= max_levels:
            # Return accumulated angles (which starts with temporal angles)
            if angles is None or angles.numel() == 0:
                # If no angles provided, create empty tensor with correct batch dims
                angles = torch.zeros(*batch_dims, 0, device=radii.device, dtype=radii.dtype)
            return radii, angles
        
        if C % 2 != 0:
            # Handle odd number of channels by padding
            padding = torch.zeros(*batch_dims, 1, device=radii.device, dtype=radii.dtype)
            radii = torch.cat([radii, padding], dim=-1)
            C = C + 1
        
        # Split channels into pairs
        C_half = C // 2
        radii_reshaped = radii.reshape(*batch_dims, C_half, 2)
        
        # Extract pairs for polar transform
        x = radii_reshaped[..., 0]  # [B, F/2, N, C/2]
        y = radii_reshaped[..., 1]  # [B, F/2, N, C/2]
        
        # Apply polar transform to radii pairs
        new_radii, new_angles = self.cartesian_to_polar(x, y)
        
        # Accumulate angles: add new_angles to the end
        if angles is None or angles.numel() == 0:
            all_angles = new_angles
        else:
            all_angles = torch.cat([angles, new_angles], dim=-1)
        
        # Recurse with incremented level
        return self.recursive_polar_transform(new_radii, all_angles, level + 1, max_levels)
    
    def inverse_recursive_polar_transform(
        self,
        final_radius: torch.Tensor,
        all_angles: torch.Tensor,
        original_channels: int
    ) -> torch.Tensor:
        """
        Reconstruct radii from recursive polar representation.
        
        The all_angles tensor contains:
        - First 'original_channels' angles: temporal angles from initial polar transform
        - Remaining angles: from recursive compression levels
        
        Args:
            final_radius: Single radius value [B, F/2, N, 1]
            all_angles: Concatenated angles (temporal + recursive)
            original_channels: Original channel dimension
            
        Returns:
            reconstructed_radii: Full radii tensor [B, F/2, N, C]
        """
        if original_channels <= 1:
            # No recursion happened, return as-is (but reshape to match expected)
            return final_radius.reshape(*final_radius.shape[:-1], original_channels)
        
        # Calculate number of recursion levels applied
        # log2(original_channels) levels, each producing C/(2^level) angles
        # But we stop when we can't halve anymore
        num_levels = int(math.log2(original_channels))
        
        # Figure out how many angles came from recursive compression
        # At level i (0-indexed), we produce C/(2^(i+1)) angles
        # Total recursive angles = C/2 + C/4 + C/8 + ... = C - 1 (for power of 2)
        recursive_angles_count = original_channels - 1
        
        # Extract recursive angles (last 'recursive_angles_count' elements)
        # Temporal angles are first 'original_channels' elements
        if all_angles.shape[-1] >= recursive_angles_count:
            recursive_angles = all_angles[..., -recursive_angles_count:]
        else:
            # Not enough angles, pad with zeros
            deficit = recursive_angles_count - all_angles.shape[-1]
            padding = torch.zeros(*all_angles.shape[:-1], deficit, 
                                 device=all_angles.device, dtype=all_angles.dtype)
            recursive_angles = torch.cat([all_angles, padding], dim=-1) if all_angles.numel() > 0 else padding
        
        # Start with final radius and reconstruct
        radii = final_radius
        
        # Process levels in reverse (from deepest to shallowest)
        # Level i has 2^i radii (in the forward direction)
        # So in reverse, we start from 1 radii and expand to 2, 4, 8, ...
        angle_offset = recursive_angles_count
        
        for level in range(num_levels - 1, -1, -1):
            # At this level, we had C/(2^(level+1)) angles
            num_angles_this_level = original_channels // (2 ** (level + 1))
            
            # Extract angles for this level
            angle_start = angle_offset - num_angles_this_level
            if angle_start < 0:
                angle_start = 0
            angle_end = angle_offset
            level_angles = recursive_angles[..., angle_start:angle_end] if recursive_angles.numel() > 0 else \
                          torch.zeros(*radii.shape[:-1], num_angles_this_level, device=radii.device, dtype=radii.dtype)
            angle_offset = angle_start
            
            # Reconstruct: each radius becomes two (x, y) via polar_to_cartesian
            x = radii * torch.cos(level_angles)
            y = radii * torch.sin(level_angles)
            
            # Interleave x and y to get double the channels
            # Stack and reshape: [..., C] -> [..., 2*C]
            radii = torch.stack([x, y], dim=-1).reshape(*x.shape[:-1], -1)
        
        return radii[..., :original_channels]
    
    def adaptive_bit_allocation(
        self,
        radii: torch.Tensor,
        angles: torch.Tensor,
        total_bits: Optional[float] = None
    ) -> Tuple[int, int]:
        """
        Allocate bits between radii and angles based on information content.
        
        Implements 60/40 radii/angle split as per TurboQuant paper.
        Radii carry more information (magnitude) and get more bits.
        
        Args:
            radii: Radii tensor
            angles: Angles tensor  
            total_bits: Target average bits per element (defaults to config)
            
        Returns:
            radii_bits: Number of bits for radii quantization
            angle_bits: Number of bits for angle quantization
        """
        if total_bits is None:
            total_bits = self.config.target_bits
        
        # Total bits for the pair of elements (one radii, one angle)
        total_pair_bits = total_bits * 2
        
        # Calculate bit allocation based on configured ratio
        # Radii get 60%, angles get 40%
        radii_bits_exact = total_pair_bits * self.config.radii_allocation
        angle_bits_exact = total_pair_bits * self.config.angle_allocation
        
        # Round to integers, ensuring minimum 1 bit each
        radii_bits = max(1, int(round(radii_bits_exact)))
        angle_bits = max(1, int(round(angle_bits_exact)))
        
        # Adjust if sum exceeds target significantly
        # Allow some flexibility for the ratio
        current_total = radii_bits + angle_bits
        target_total = int(round(total_pair_bits))
        
        if current_total > target_total + 1:
            # Need to reduce - prefer reducing the one with more bits
            excess = current_total - target_total
            if radii_bits > angle_bits:
                radii_bits = max(1, radii_bits - excess)
            else:
                angle_bits = max(1, angle_bits - excess)
        
        return radii_bits, angle_bits
    
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        bits: int,
        symmetric: bool = True
    ) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
        """
        Quantize a tensor to specified bits.
        
        Args:
            tensor: Input tensor to quantize
            bits: Number of bits (1-16)
            symmetric: Use symmetric quantization (zero-centered)
            
        Returns:
            quantized: Quantized tensor as integers
            scale: Scale factor for dequantization
            zero_point: Zero point offset (0 for symmetric)
            original_scale: The scale/max value for information
        """
        num_levels = 2 ** bits
        
        if symmetric:
            # Symmetric quantization: [-max, +max] -> quantized range
            abs_max = tensor.abs().max()
            
            # Avoid division by zero
            if abs_max < 1e-10:
                return torch.zeros_like(tensor, dtype=torch.int32), 1.0, 0.0, torch.zeros_like(tensor)
            
            # Scale: value per quantization level
            # For n bits, we have 2^(n-1) - 1 positive and negative levels
            # (excluding one value for symmetry)
            max_quant_val = (num_levels // 2) - 1
            scale = abs_max / max_quant_val if max_quant_val > 0 else abs_max
            
            # Quantize: round(tensor / scale)
            quantized = torch.round(tensor / scale).clamp(-(num_levels // 2), num_levels // 2 - 1)
            zero_point = 0.0
        else:
            # Asymmetric quantization
            t_min, t_max = tensor.min(), tensor.max()
            scale = (t_max - t_min) / (num_levels - 1) if t_max > t_min else 1.0
            zero_point = -t_min / scale
            quantized = torch.round((tensor / scale) + zero_point).clamp(0, num_levels - 1)
        
        return quantized.to(torch.int32), float(scale), float(zero_point), tensor
    
    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scale: float,
        zero_point: float,
        symmetric: bool = True
    ) -> torch.Tensor:
        """
        Dequantize tensor back to floating point.
        
        Args:
            quantized: Quantized integer tensor
            scale: Scale factor from quantization
            zero_point: Zero point offset
            symmetric: Whether symmetric quantization was used
            
        Returns:
            dequantized: Dequantized floating point tensor
        """
        if symmetric:
            return quantized.float() * scale
        else:
            return (quantized.float() - zero_point) * scale
    
    def pack_quantized(
        self,
        radii_quantized: torch.Tensor,
        angles_quantized: torch.Tensor,
        radii_bits: int,
        angle_bits: int
    ) -> Dict[str, torch.Tensor]:
        """
        Pack quantized radii and angles into bit-packed storage.
        
        Args:
            radii_quantized: Integer tensor of quantized radii
            angles_quantized: Integer tensor of quantized angles
            radii_bits: Bits used per radii element
            angle_bits: Bits used per angle element
            
        Returns:
            packed: Dictionary with packed tensors and metadata
        """
        # Store metadata
        packed = {
            'radii_bits': radii_bits,
            'angle_bits': angle_bits,
            'radii_shape': radii_quantized.shape,
            'angles_shape': angles_quantized.shape,
            'packed_data': None,  # Actual bit packing would go here
            'radii_data': radii_quantized,  # For now, store as-is
            'angles_data': angles_quantized,  # For now, store as-is
        }
        
        return packed
    
    def quantize(self, tensor: torch.Tensor) -> Dict[str, any]:
        """
        Main quantization entry point: FP16 tensor -> TPQ representation.
        
        Args:
            tensor: Input tensor [B, F, N, C] (Batch, Frames, Patches, Channels)
            
        Returns:
            quantized_data: Dictionary containing:
                - packed: Bit-packed quantized data
                - metadata: Quantization parameters (scales, zero_points)
                - shape_info: Original tensor shape
        """
        original_shape = tensor.shape
        B, F, N, C = original_shape
        
        # Ensure even number of frames
        if F % 2 != 0:
            # Pad with zeros
            padding = torch.zeros(B, 1, N, C, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=1)
            F = F + 1
            was_padded = True
        else:
            was_padded = False
        
        # Step 1: Group temporal pairs
        grouped = self._group_temporal_pairs(tensor)  # [B, F/2, N, C, 2]
        
        # Step 2: Polar transform on temporal pairs
        x = grouped[..., 0]  # frame_t
        y = grouped[..., 1]  # frame_{t+1}
        radii, angles = self.cartesian_to_polar(x, y)  # [B, F/2, N, C]
        
        # Step 3: Recursive polar compression on radii (if enabled and C is power of 2)
        if self.config.enable_recursive and C > 1 and (C & (C - 1)) == 0:  # Power of 2 check
            final_radius, all_angles = self.recursive_polar_transform(radii, angles)
        else:
            final_radius = radii
            all_angles = angles
        
        # Step 4: Adaptive bit allocation
        radii_bits, angle_bits = self.adaptive_bit_allocation(final_radius, all_angles)
        
        # Step 5: Quantize radii and angles
        radii_quant, radii_scale, radii_zp, _ = self.quantize_tensor(final_radius, radii_bits, symmetric=True)
        angles_quant, angles_scale, angles_zp, _ = self.quantize_tensor(all_angles, angle_bits, symmetric=True)
        
        # Step 6: Pack quantized data
        packed = self.pack_quantized(radii_quant, angles_quant, radii_bits, angle_bits)
        
        # Store metadata for dequantization
        metadata = {
            'original_shape': original_shape,
            'padded': was_padded,
            'radii_scale': radii_scale,
            'radii_zero_point': radii_zp,
            'angles_scale': angles_scale,
            'angles_zero_point': angles_zp,
            'radii_bits': radii_bits,
            'angle_bits': angle_bits,
            'original_channels': C,
            'enable_recursive': self.config.enable_recursive,
        }
        
        return {
            'packed': packed,
            'metadata': metadata,
        }
    
    def dequantize(self, quantized_data: Dict[str, any]) -> torch.Tensor:
        """
        Main dequantization entry point: TPQ representation -> FP16 tensor.
        
        Args:
            quantized_data: Dictionary from quantize()
            
        Returns:
            tensor: Dequantized floating point tensor [B, F, N, C]
        """
        packed = quantized_data['packed']
        metadata = quantized_data['metadata']
        
        # Extract quantized data
        radii_quant = packed['radii_data']
        angles_quant = packed['angles_data']
        
        # Dequantize radii and angles
        radii = self.dequantize_tensor(
            radii_quant, 
            metadata['radii_scale'], 
            metadata['radii_zero_point'],
            symmetric=True
        )
        angles = self.dequantize_tensor(
            angles_quant,
            metadata['angles_scale'],
            metadata['angles_zero_point'],
            symmetric=True
        )
        
        # Inverse recursive polar transform (if enabled)
        if metadata['enable_recursive'] and metadata['original_channels'] > 1:
            reconstructed_radii = self.inverse_recursive_polar_transform(
                radii, angles, metadata['original_channels']
            )
        else:
            reconstructed_radii = radii
        
        # Inverse polar transform to get frame pairs
        # We need to split angles: part for reconstruction, part for temporal
        C = metadata['original_channels']
        temporal_angles = angles[..., :C] if angles.shape[-1] >= C else angles
        
        x, y = self.polar_to_cartesian(reconstructed_radii, temporal_angles)
        
        # Stack to form paired tensor [B, F/2, N, C, 2]
        paired = torch.stack([x, y], dim=-1)
        
        # Ungroup temporal pairs
        original_shape = metadata['original_shape']
        tensor = self._ungroup_temporal_pairs(paired, original_shape[1])
        
        # Remove padding if it was added
        if metadata.get('padded', False) and tensor.shape[1] > original_shape[1]:
            tensor = tensor[:, :original_shape[1], :, :]
        
        return tensor.reshape(original_shape)
