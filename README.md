# YOLO Auto-Rickshaw Fine-tuning

Fine-tune **YOLOv8 medium** to detect auto rickshaws on a Rocky Linux server.

## Dataset

- **Source:** [Roboflow Universe - Auto-Rickshaw (v3)](https://universe.roboflow.com/autorickshaw-detection/auto-rickshaw-9fpsm)
- **Images:** 4,514 (augmented), license CC BY 4.0
- **Format:** YOLOv8 (downloads already split into train/val/test)
- **Classes:** Auto-Rickshaw, Auto Rickshaw

> Note: The original DataCluster Labs autorickshaw dataset on Kaggle only
> provides a sample (100 images); the full set requires a paid license.

## Requirements

- Rocky Linux server
- Python 3.9+
- Free Roboflow account (for API key)

## Quick Start

```bash
git clone https://github.com/VISHALYADAV112/yolo-autorickshaw
cd yolo-autorickshaw

# 1. Install dependencies
make setup

# 2. Set your Roboflow API key
#    Get it from https://app.roboflow.com -> Settings -> API Key
export ROBOFLOW_API_KEY=your_key_here

# 3. Download dataset
make download

# 4. Train YOLOv8 medium
make train
```

## Commands

```bash
make setup                         # Install all dependencies
make download                      # Download dataset from Roboflow
make train                         # Train YOLOv8 medium (100 epochs)
make train EPOCHS=50 BATCH=8       # Customize training
make train-small                   # YOLOv8s (faster)
make train-large                   # YOLOv8l (more accurate)
make eval                          # Evaluate trained model
make predict SOURCE=image.jpg      # Run inference on an image/folder
make export                        # Export best.pt to ONNX
make clean                         # Remove training runs
```

## Train YOLOv8 Medium (custom)

```bash
make train MODEL=yolov8m.pt EPOCHS=100 IMG_SIZE=640 BATCH=16 DEVICE=0
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `MODEL` | `yolov8m.pt` | Base weights |
| `EPOCHS` | `100` | Training epochs |
| `IMG_SIZE` | `640` | Input image size |
| `BATCH` | `16` | Batch size |
| `DEVICE` | `0` | GPU id, or `cpu` |

## GPU Notes

- Training auto-selects GPU with `device=0` (CUDA). For CPU: `make train DEVICE=cpu`.
- YOLOv8 downloads pre-trained COCO weights automatically on first run.

## Project Structure

```
├── Makefile              # Command shortcuts
├── requirements.txt      # Python dependencies
├── configs/data.yaml     # Fallback dataset config template
├── scripts/
│   ├── setup.sh          # One-time server setup
│   └── download_dataset.sh  # Fetch dataset + normalize to YOLO format
└── src/
    └── train.py          # Train / eval / predict CLI
```

## Output

Trained weights and plots are saved to `runs/autorickshaw/`:

```
runs/autorickshaw/weights/best.pt   # Best model
runs/autorickshaw/weights/last.pt   # Last epoch model
runs/autorickshaw/results.png       # Loss + metric plots
```
