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

# Install system dependencies
echo "[1/5] Installing system dependencies..."
$SUDO dnf install -y python3 python3-pip git wget unzip

# Create virtual environment
echo "[2/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "[3/5] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup Roboflow API credentials
echo "[4/5] Setting up Roboflow API..."
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo ""
    echo "  Roboflow API key not found."
    echo "  1. Go to https://app.roboflow.com (free signup)"
    echo "  2. Click your username -> Settings -> API Key"
    echo "  3. Export it: export ROBOFLOW_API_KEY=your_key_here"
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
echo "  1. Get free API key from https://app.roboflow.com"
echo "  2. Run: export ROBOFLOW_API_KEY=your_key"
echo "  3. Run: make download    # Download dataset"
echo "  4. Run: make train       # Start training"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  export ROBOFLOW_API_KEY=your_key"
echo "  python src/train.py train"
echo ""
