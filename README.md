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

- Rocky Linux server (setup auto-installs Python 3.9+)
- Python 3.8+ (ultralytics requirement)
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
make train                         # Train YOLOv8m (L40S-tuned defaults)
make train-fast                    # Quick experiment (20 epochs, imgsz 640)
make train-multi DEVICE="0,1"      # Multi-GPU training
make eval                          # Evaluate trained model
make predict SOURCE=image.jpg      # Run inference on an image/folder
make export                        # Export best.pt to ONNX
make clean                         # Remove training runs
```

## Train YOLOv8 Medium (custom)

```bash
make train MODEL=yolov8m.pt EPOCHS=100 IMG_SIZE=1280 BATCH=-1 DEVICE=0 WORKERS=16
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `MODEL` | `yolov8m.pt` | Base weights |
| `EPOCHS` | `100` | Training epochs |
| `IMG_SIZE` | `1280` | Input image size (larger = more accurate, slower) |
| `BATCH` | `-1` | Batch size (`-1` = auto-select max that fits VRAM) |
| `DEVICE` | `0` | GPU id, or `0,1,2,3` for multi-GPU, or `cpu` |
| `WORKERS` | `16` | Data loader workers |

## GPU Notes

- **L40S tuning**: defaults use `imgsz=1280` for higher accuracy, `batch=-1`
  to auto-fill the 48GB VRAM, and `cache=ram` to prefetch the whole dataset
  into RAM (fine with 256GB) for faster epochs.
- Multi-GPU: `make train-multi DEVICE="0,1"` splits batch per GPU.
- Set `WORKERS` to your CPU core count for maximum data-loading parallelism.
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
