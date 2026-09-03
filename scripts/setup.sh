#!/bin/bash
set -e

echo "========================================="
echo "  YOLO Fine-tune Setup (Rocky Linux)"
echo "========================================="

# Check if running as root or with sudo
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# ultralytics requires Python 3.8+. Rocky 8 defaults to 3.6, which is too old.
# Prefer the newest available python3.x (3.11/3.12/3.9) via dnf.
echo "[1/5] Installing system dependencies..."
$SUDO dnf install -y git wget unzip

PY=""
for cand in python3.12 python3.11 python3.10 python3.9 python3.8; do
    if command -v $cand &> /dev/null; then
        PY=$cand
        break
    fi
done

if [ -z "$PY" ]; then
    echo "Looking for a modern Python package (needs >= 3.8)..."
    # Enable python3.11 module/tools on Rocky 8/9
    if $SUDO dnf module list python3* 2>/dev/null | grep -q python39; then
        $SUDO dnf module install -y python39 2>/dev/null || $SUDO dnf install -y python39 python39-pip
        PY="python3.9"
    else
        # Fallback: ensure at least python3 exists
        $SUDO dnf install -y python3 python3-pip
        PY="python3"
    fi
fi

echo "  Using Python: $PY ($($PY --version 2>&1))"

# Verify version is >= 3.8
PY_MAJOR=$($PY -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PY -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]; }; then
    echo "ERROR: $PY is Python $PY_MAJOR.$PY_MINOR, but ultralytics needs >= 3.8."
    echo "Install a newer Python (>= 3.8), e.g.:"
    echo "  sudo dnf module install -y python39"
    exit 1
fi

# Create virtual environment
echo "[2/5] Creating virtual environment..."
$PY -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install Python packages
echo "[3/5] Installing Python packages..."
pip install -r requirements.txt

# Setup Roboflow API credentials
echo "[4/5] Setting up Roboflow API..."
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo ""
    echo "  Roboflow API key not found."
    echo "  1. Create a free account at https://app.roboflow.com"
    echo "  2. Click your username -> Settings -> API Key (copy it)"
    echo "  3. Run: export ROBOFLOW_API_KEY=your_key_here"
    echo "     # add to ~/.bashrc to persist"
    echo ""
else
    echo "  Roboflow API key found in environment."
fi

# Make scripts executable
echo "[5/5] Making scripts executable..."
chmod +x scripts/*.sh

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Set ROBOFLOW_API_KEY (if not done)"
echo "  2. Run: make download    # Download dataset"
echo "  3. Run: make train       # Start training"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python src/train.py train"
echo ""
