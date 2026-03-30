# VideoQuant Environment

## Python Requirements

- Python 3.10 or higher (3.14.2 available in this environment)
- pip 26.0.1 or higher

## Hardware

- CPU-only (no GPU available)
- No CUDA/NVCC
- System memory: TBD based on host
- Target: Run Wan2.1-1.3B on CPU with < 8GB RAM

## Core Dependencies

```
torch >= 2.0.0 (CPU-only build)
numpy >= 1.24.0
numba >= 0.58.0
diffusers >= 0.30.0
transformers >= 4.40.0
accelerate >= 0.25.0
```

## Development Dependencies

```
pytest >= 7.0.0
pytest-cov >= 4.0.0
black >= 23.0.0
ruff >= 0.1.0
mypy >= 1.5.0
```

## Optional Dependencies

```
comfyui (for custom nodes, install in ComfyUI environment)
```

## Model Downloads

**Wan2.1-T2V-1.3B:**
- HuggingFace: `Wan-AI/Wan2.1-T2V-1.3B`
- License: Apache 2.0
- Size: ~5GB download, ~2.6GB FP16 weights
- Quantized size: ~0.65GB (4-bit weights)

## Environment Variables

Optional configuration:

```bash
export VIDEOQUANT_CACHE_DIR="~/.cache/videoquant"
export VIDEOQUANT_DEFAULT_BITS=4
export VIDEOQUANT_CPU_THREADS=4
```

## Platform Notes

### Linux (Current Environment)
- Native Python 3.14.2
- PyTorch CPU wheels available
- Numba works with LLVM

### macOS (Not Tested)
- May require different PyTorch index
- Numba may have limitations

### Windows (Not Tested)
- WSL2 recommended
- Native Windows support possible but untested

## Installation

```bash
# Clone repository
git clone <repo-url>
cd videoquant

# Run setup script
source .factory/init.sh

# Or manual setup
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Verification

```bash
# Check Python version
python3 --version  # Should be 3.10+

# Check PyTorch (CPU)
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"  # Should print False for CUDA

# Check Numba
python3 -c "import numba; print(numba.__version__)"

# Run quick tests
pytest tests/test_tpq.py -v
```

## Known Issues

1. **No GPU**: All CUDA-related code will fail
2. **CPU Speed**: Wan2.1 inference ~8-10 minutes per video on CPU
3. **Memory**: Need ~8GB free RAM for model inference
