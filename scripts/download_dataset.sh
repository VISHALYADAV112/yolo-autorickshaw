#!/bin/bash
set -e

echo "========================================="
echo "  Downloading Autorickshaw Dataset"
echo "  Source: Roboflow Universe (CC BY 4.0)"
echo "  Dataset: Auto rickshaw by VIT"
echo "  1,941 images, YOLO format"
echo "========================================="

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Check roboflow package
if ! python3 -c "import roboflow" 2>/dev/null; then
    echo "Error: roboflow package not found. Run: make setup"
    exit 1
fi

# Check API key
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "Error: ROBOFLOW_API_KEY not set."
    echo ""
    echo "  1. Go to https://app.roboflow.com (free signup)"
    echo "  2. Click your username -> Settings -> API Key"
    echo "  3. Run: export ROBOFLOW_API_KEY=your_key"
    exit 1
fi

# Download dataset using Roboflow
echo "[1/3] Downloading dataset from Roboflow..."
mkdir -p dataset

python3 -c "
from roboflow import Roboflow

rf = Roboflow(api_key='${ROBOFLOW_API_KEY}')
project = rf.workspace('vit-0mr4g').project('auto-rickshaw-oobkc')
version = project.version(9)
dataset = version.download('yolov8', location='./dataset')

print(f'Dataset downloaded to: {dataset.location}')
print(f'Classes: {dataset.classes}')
"

# Check if dataset needs restructuring
echo "[2/3] Checking dataset structure..."
if [ -d "dataset/train" ] && [ -d "dataset/valid" ]; then
    echo "  Roboflow format detected (train/valid/test)"
    # Rename valid -> val for YOLO compatibility
    if [ -d "dataset/valid" ] && [ ! -d "dataset/val" ]; then
        mv dataset/valid dataset/val
        echo "  Renamed 'valid' -> 'val'"
    fi
    # Check for test set and merge into val if small
    if [ -d "dataset/test" ]; then
        TEST_COUNT=$(find dataset/test -name "*.jpg" -o -name "*.png" | wc -l)
        echo "  Test set has $TEST_COUNT images (will be used for validation)"
    fi
fi

echo "[3/3] Dataset ready!"
echo ""
echo "Dataset structure:"
ls -la dataset/
echo ""
echo "Train images: $(find dataset/train -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l)"
echo "Val images: $(find dataset/val -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l)"
if [ -d "dataset/test" ]; then
    echo "Test images: $(find dataset/test -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l)"
fi
echo ""
echo "Run 'make train' to start training."
