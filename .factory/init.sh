#!/bin/bash
# VideoQuant Environment Setup
# Idempotent setup script

set -e

echo "Setting up VideoQuant environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "Error: Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version check passed ($PYTHON_VERSION)"

# Check for pip
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found"
    exit 1
fi

echo "✓ pip3 available"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy numba

# Install ML/Diffusion dependencies
echo "Installing ML dependencies..."
pip install diffusers transformers accelerate

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov black ruff mypy

# Install project in editable mode
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing VideoQuant in editable mode..."
    pip install -e .
fi

# Create necessary directories
mkdir -p videoquant/{core,integration,metrics,kernels}
mkdir -p tests
mkdir -p benchmarks
mkdir -p scripts
mkdir -p configs

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""
