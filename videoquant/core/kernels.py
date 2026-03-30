"""
VideoQuant - CPU-Optimized Kernels

Implements CPU-optimized kernels using:
- Numba JIT compilation for hot paths
- NumPy vectorization
- PyTorch SIMD operations (via torch's optimized backends)
- Memory-efficient operations

Target performance:
- Polar transform < 50ms for [4, 16, 256, 512] tensor
- JL projection < 100ms for typical dimensions
- Overall < 2x slower than theoretical GPU speed
"""

import torch
import numpy as np
from typing import Tuple, Optional
import math

# Try to import Numba - if available, use JIT compilation
try:
    from numba import njit, prange, jit
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators for when Numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)


# =============================================================================
# Numba JIT-compiled kernels for hot paths
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True, fastmath=True)
    def _cartesian_to_polar_numba(x: np.ndarray, y: np.ndarray,
                                   radius: np.ndarray, angle: np.ndarray):
        """Numba-accelerated cartesian to polar conversion."""
        for i in prange(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(x.shape[3]):
                        xi = x[i, j, k, l]
                        yi = y[i, j, k, l]
                        radius[i, j, k, l] = math.sqrt(xi * xi + yi * yi + 1e-10)
                        angle[i, j, k, l] = math.atan2(yi, xi)

    @njit(parallel=True, cache=True, fastmath=True)
    def _polar_to_cartesian_numba(radius: np.ndarray, angle: np.ndarray,
                                   x: np.ndarray, y: np.ndarray):
        """Numba-accelerated polar to cartesian conversion."""
        for i in prange(radius.shape[0]):
            for j in range(radius.shape[1]):
                for k in range(radius.shape[2]):
                    for l in range(radius.shape[3]):
                        r = radius[i, j, k, l]
                        a = angle[i, j, k, l]
                        x[i, j, k, l] = r * math.cos(a)
                        y[i, j, k, l] = r * math.sin(a)

    @njit(parallel=True, cache=True, fastmath=True)
    def _quantize_symmetric_numba(tensor: np.ndarray, bits: int,
                                   abs_max: float) -> np.ndarray:
        """Numba-accelerated symmetric quantization."""
        max_quant_val = (2 ** (bits - 1)) - 1
        scale = abs_max / max_quant_val if max_quant_val > 0 else abs_max
        if scale < 1e-10:
            return np.zeros_like(tensor, dtype=np.int32)

        result = np.empty_like(tensor, dtype=np.int32)
        for i in prange(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    for l in range(tensor.shape[3]):
                        val = tensor[i, j, k, l]
                        quantized = round(val / scale)
                        # Clamp to range
                        min_val = -(2 ** (bits - 1))
                        max_val = 2 ** (bits - 1) - 1
                        if quantized < min_val:
                            quantized = min_val
                        elif quantized > max_val:
                            quantized = max_val
                        result[i, j, k, l] = int(quantized)
        return result

    @njit(parallel=True, cache=True, fastmath=True)
    def _dequantize_symmetric_numba(quantized: np.ndarray, scale: float,
                                    result: np.ndarray):
        """Numba-accelerated symmetric dequantization."""
        for i in prange(quantized.shape[0]):
            for j in range(quantized.shape[1]):
                for k in range(quantized.shape[2]):
                    for l in range(quantized.shape[3]):
                        result[i, j, k, l] = float(quantized[i, j, k, l]) * scale

    @njit(parallel=True, cache=True, fastmath=True)
    def _jl_projection_numba(tensor: np.ndarray, projection: np.ndarray,
                             result: np.ndarray):
        """Numba-accelerated JL projection via matrix multiplication."""
        batch = tensor.shape[0]
        input_dim = tensor.shape[1]
        output_dim = projection.shape[1]

        for i in prange(batch):
            for j in range(output_dim):
                sum_val = 0.0
                for k in range(input_dim):
                    sum_val += tensor[i, k] * projection[k, j]
                result[i, j] = sum_val

    @njit(parallel=True, cache=True, fastmath=True)
    def _sign_quantize_numba(tensor: np.ndarray, result: np.ndarray):
        """Numba-accelerated sign-bit quantization."""
        for i in prange(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    for l in range(tensor.shape[3]):
                        result[i, j, k, l] = tensor[i, j, k, l] >= 0

    @njit(parallel=True, cache=True, fastmath=True)
    def _sign_dequantize_numba(quantized: np.ndarray, scale: float,
                                result: np.ndarray):
        """Numba-accelerated sign-bit dequantization."""
        for i in prange(quantized.shape[0]):
            for j in range(quantized.shape[1]):
                for k in range(quantized.shape[2]):
                    for l in range(quantized.shape[3]):
                        if quantized[i, j, k, l]:
                            result[i, j, k, l] = scale
                        else:
                            result[i, j, k, l] = -scale

    @njit(parallel=True, cache=True, fastmath=True)
    def _pack_bits_numba(quantized: np.ndarray, bits: int,
                         packed: np.ndarray):
        """Numba-accelerated bit packing."""
        elements_per_byte = 8 // bits
        total_elements = quantized.size

        for i in prange(0, total_elements, elements_per_byte):
            packed_val = 0
            for j in range(elements_per_byte):
                idx = i + j
                if idx < total_elements:
                    val = quantized.flat[idx]
                    # Ensure value fits in bits
                    max_val = (1 << bits) - 1
                    if val < 0:
                        val = 0
                    elif val > max_val:
                        val = max_val
                    packed_val |= (int(val) & max_val) << (j * bits)
            packed[i // elements_per_byte] = packed_val

    @njit(parallel=True, cache=True, fastmath=True)
    def _unpack_bits_numba(packed: np.ndarray, bits: int, total_elements: int,
                           result: np.ndarray):
        """Numba-accelerated bit unpacking."""
        elements_per_byte = 8 // bits
        max_val = (1 << bits) - 1

        for i in prange(packed.size):
            packed_val = packed[i]
            for j in range(elements_per_byte):
                idx = i * elements_per_byte + j
                if idx < total_elements:
                    val = (packed_val >> (j * bits)) & max_val
                    result.flat[idx] = val


# =============================================================================
# PyTorch-optimized operations (SIMD via PyTorch's optimized backends)
# =============================================================================

def cartesian_to_polar_optimized(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized cartesian to polar conversion.

    Uses PyTorch's vectorized operations with automatic SIMD optimization.

    Args:
        x: First component [B, F, N, C] or any shape
        y: Second component [B, F, N, C] or any shape

    Returns:
        radius: Magnitude of vector (>= 0)
        angle: Direction of vector in [-pi, pi]
    """
    # PyTorch's sqrt and atan2 are already optimized with SIMD
    # We add a small epsilon for numerical stability
    radius = torch.sqrt(x * x + y * y + 1e-10)
    angle = torch.atan2(y, x)
    return radius, angle


def polar_to_cartesian_optimized(radius: torch.Tensor, angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized polar to cartesian conversion.

    Uses PyTorch's vectorized trigonometric operations.

    Args:
        radius: Magnitude of vector (>= 0)
        angle: Direction of vector in radians

    Returns:
        x: First component
        y: Second component
    """
    # PyTorch's cos and sin are already optimized with SIMD
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    return x, y


def quantize_symmetric_optimized(tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float]:
    """
    Optimized symmetric quantization.

    Uses PyTorch's vectorized operations for scale computation and quantization.

    Args:
        tensor: Input tensor to quantize
        bits: Number of bits (1-16)

    Returns:
        quantized: Quantized tensor as integers
        scale: Scale factor for dequantization
    """
    num_levels = 2 ** bits
    abs_max = tensor.abs().max()

    # Avoid division by zero
    if abs_max < 1e-10:
        return torch.zeros_like(tensor, dtype=torch.int32), 1.0

    max_quant_val = (num_levels // 2) - 1
    scale = abs_max / max_quant_val if max_quant_val > 0 else abs_max

    # Vectorized quantization
    quantized = torch.round(tensor / scale).clamp(-(num_levels // 2), num_levels // 2 - 1)

    return quantized.to(torch.int32), float(scale)


def dequantize_symmetric_optimized(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Optimized symmetric dequantization.

    Args:
        quantized: Quantized integer tensor
        scale: Scale factor from quantization

    Returns:
        dequantized: Dequantized floating point tensor
    """
    return quantized.float() * scale


def jl_projection_optimized(tensor: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """
    Optimized Johnson-Lindenstrauss projection.

    Uses PyTorch's optimized matrix multiplication (calls BLAS/LAPACK).

    Args:
        tensor: Input tensor [..., input_dim]
        projection: Projection matrix [input_dim, output_dim]

    Returns:
        projected: Projected tensor [..., output_dim]
    """
    # PyTorch's matmul is highly optimized and uses BLAS
    return torch.matmul(tensor, projection)


def sign_quantize_optimized(tensor: torch.Tensor) -> torch.Tensor:
    """
    Optimized sign-bit quantization.

    Args:
        tensor: Input tensor of any shape

    Returns:
        quantized: Boolean tensor (True for positive/zero, False for negative)
    """
    return tensor >= 0


def sign_dequantize_optimized(quantized: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """
    Optimized sign-bit dequantization.

    Args:
        quantized: Boolean tensor from sign_quantize
        scale: Optional magnitude to scale signs (defaults to sqrt(pi/2))

    Returns:
        dequantized: Float tensor with values {-scale, +scale}
    """
    if scale is None:
        scale = math.sqrt(math.pi / 2)

    # Vectorized: True -> +scale, False -> -scale
    return quantized.float() * (2 * scale) - scale


# =============================================================================
# Unified kernel interface with automatic dispatch
# =============================================================================

class CPUOptimizedKernels:
    """
    CPU-optimized kernels for VideoQuant quantization operations.

    Provides a unified interface that automatically selects the best
    implementation based on tensor size and available optimizations.

    Usage:
        >>> kernels = CPUOptimizedKernels()
        >>> radius, angle = kernels.cartesian_to_polar(x, y)
        >>> projected = kernels.jl_projection(tensor, proj_matrix)
    """

    def __init__(self, use_numba: bool = True, use_jit: bool = True):
        """
        Initialize optimized kernels.

        Args:
            use_numba: Whether to use Numba JIT compilation when available
            use_jit: Whether to use JIT compilation (torch.jit)
        """
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.use_jit = use_jit
        self._numba_cache = {}

        # Warm up Numba kernels if available
        if self.use_numba:
            self._warmup_numba()

    def _warmup_numba(self):
        """Warm up Numba kernels with small tensors to trigger JIT compilation."""
        if not NUMBA_AVAILABLE:
            return

        # Small tensors to trigger compilation
        x = np.random.randn(2, 2, 2, 2).astype(np.float32)
        y = np.random.randn(2, 2, 2, 2).astype(np.float32)
        radius = np.empty_like(x)
        angle = np.empty_like(x)

        try:
            _cartesian_to_polar_numba(x, y, radius, angle)
            _polar_to_cartesian_numba(radius, angle, x, y)
            _quantize_symmetric_numba(x, 4, 1.0)
            _sign_quantize_numba(x, np.empty_like(x, dtype=np.bool_))
        except Exception:
            pass  # Warmup failures are non-fatal

    def cartesian_to_polar(self, x: torch.Tensor, y: torch.Tensor,
                           use_numba: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to polar coordinates.

        Args:
            x: First component
            y: Second component
            use_numba: Override to use Numba (default: auto-select)

        Returns:
            radius, angle: Polar coordinates
        """
        # For large tensors, try Numba if available
        if (use_numba or (use_numba is None and self.use_numba)) and x.numel() > 10000:
            try:
                # Convert to numpy for Numba processing
                x_np = x.cpu().numpy()
                y_np = y.cpu().numpy()
                radius_np = np.empty_like(x_np)
                angle_np = np.empty_like(y_np)

                _cartesian_to_polar_numba(x_np, y_np, radius_np, angle_np)

                return (torch.from_numpy(radius_np).to(x.device, x.dtype),
                        torch.from_numpy(angle_np).to(y.device, y.dtype))
            except Exception:
                pass  # Fall back to PyTorch

        return cartesian_to_polar_optimized(x, y)

    def polar_to_cartesian(self, radius: torch.Tensor, angle: torch.Tensor,
                           use_numba: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert polar coordinates to Cartesian coordinates."""
        if (use_numba or (use_numba is None and self.use_numba)) and radius.numel() > 10000:
            try:
                r_np = radius.cpu().numpy()
                a_np = angle.cpu().numpy()
                x_np = np.empty_like(r_np)
                y_np = np.empty_like(a_np)

                _polar_to_cartesian_numba(r_np, a_np, x_np, y_np)

                return (torch.from_numpy(x_np).to(radius.device, radius.dtype),
                        torch.from_numpy(y_np).to(angle.device, angle.dtype))
            except Exception:
                pass

        return polar_to_cartesian_optimized(radius, angle)

    def quantize_symmetric(self, tensor: torch.Tensor, bits: int,
                          use_numba: Optional[bool] = None) -> Tuple[torch.Tensor, float]:
        """Symmetric quantization with optimized implementation."""
        if (use_numba or (use_numba is None and self.use_numba)) and tensor.numel() > 10000:
            try:
                t_np = tensor.cpu().numpy()
                abs_max = float(np.abs(t_np).max())
                quantized_np = _quantize_symmetric_numba(t_np, bits, abs_max)

                max_quant_val = (2 ** (bits - 1)) - 1
                scale = abs_max / max_quant_val if max_quant_val > 0 else abs_max

                return torch.from_numpy(quantized_np).to(tensor.device, torch.int32), scale
            except Exception:
                pass

        return quantize_symmetric_optimized(tensor, bits)

    def dequantize_symmetric(self, quantized: torch.Tensor, scale: float,
                             use_numba: Optional[bool] = None) -> torch.Tensor:
        """Symmetric dequantization with optimized implementation."""
        if (use_numba or (use_numba is None and self.use_numba)) and quantized.numel() > 10000:
            try:
                q_np = quantized.cpu().numpy()
                result_np = np.empty_like(q_np, dtype=np.float32)
                _dequantize_symmetric_numba(q_np, float(scale), result_np)
                return torch.from_numpy(result_np).to(quantized.device, torch.float32)
            except Exception:
                pass

        return dequantize_symmetric_optimized(quantized, scale)

    def jl_projection(self, tensor: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        """
        Johnson-Lindenstrauss projection.

        Uses PyTorch's optimized matmul (which uses BLAS).
        """
        return jl_projection_optimized(tensor, projection)

    def sign_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sign-bit quantization."""
        return sign_quantize_optimized(tensor)

    def sign_dequantize(self, quantized: torch.Tensor,
                        scale: Optional[float] = None) -> torch.Tensor:
        """Sign-bit dequantization."""
        return sign_dequantize_optimized(quantized, scale)

    def compute_quantization_error(self, original: torch.Tensor,
                                   quantized: torch.Tensor) -> Tuple[float, float]:
        """
        Compute quantization error metrics.

        Args:
            original: Original tensor
            quantized: Quantized/dequantized tensor

        Returns:
            (cosine_similarity, l2_relative_error)
        """
        orig_flat = original.reshape(1, -1)
        quant_flat = quantized.reshape(1, -1)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_flat, quant_flat, dim=1
        ).item()

        # L2 relative error
        l2_orig = torch.norm(original)
        l2_error = torch.norm(original - quantized)
        l2_rel = (l2_error / (l2_orig + 1e-10)).item()

        return cos_sim, l2_rel


# =============================================================================
# Convenience functions
# =============================================================================

# Global kernel instance for convenience
_default_kernels = None

def get_kernels() -> CPUOptimizedKernels:
    """Get the default CPU optimized kernels instance."""
    global _default_kernels
    if _default_kernels is None:
        _default_kernels = CPUOptimizedKernels()
    return _default_kernels


def reset_kernels():
    """Reset the default kernels instance (useful for testing)."""
    global _default_kernels
    _default_kernels = None


# Direct optimized function exports
__all__ = [
    'CPUOptimizedKernels',
    'get_kernels',
    'reset_kernels',
    'cartesian_to_polar_optimized',
    'polar_to_cartesian_optimized',
    'quantize_symmetric_optimized',
    'dequantize_symmetric_optimized',
    'jl_projection_optimized',
    'sign_quantize_optimized',
    'sign_dequantize_optimized',
    'NUMBA_AVAILABLE',
]
