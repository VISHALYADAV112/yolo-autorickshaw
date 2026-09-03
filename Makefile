.PHONY: setup download train eval predict clean help

PYTHON := python3
VENV := venv
MODEL := yolov8m.pt
EPOCHS := 100
IMG_SIZE := 640
BATCH := 16
DEVICE := 0

help: ## Show this help
	@echo "Usage: make <command>"
	@echo ""
	@echo "Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup: ## Install dependencies and setup environment
	bash scripts/setup.sh

download: ## Download dataset from Roboflow (requires ROBOFLOW_API_KEY)
	bash scripts/download_dataset.sh

train: ## Train YOLOv8 medium (customizable: make train EPOCHS=50 BATCH=8)
	$(VENV)/bin/python src/train.py train \
		--model $(MODEL) \
		--epochs $(EPOCHS) \
		--imgsz $(IMG_SIZE) \
		--batch $(BATCH) \
		--device $(DEVICE)

train-small: ## Train YOLOv8 small (faster, less accurate)
	$(VENV)/bin/python src/train.py train --model yolov8s.pt --epochs $(EPOCHS) --batch 32

train-large: ## Train YOLOv8 large (slower, more accurate)
	$(VENV)/bin/python src/train.py train --model yolov8l.pt --epochs $(EPOCHS) --batch 8

eval: ## Evaluate trained model
	$(VENV)/bin/python src/train.py eval --model runs/autorickshaw/weights/best.pt

predict: ## Run inference on image/folder (make predict SOURCE=path/to/image)
	$(VENV)/bin/python src/train.py predict --model runs/autorickshaw/weights/best.pt --source $(SOURCE)

export: ## Export model to ONNX
	$(VENV)/bin/python -c "from ultralytics import YOLO; YOLO('runs/autorickshaw/weights/best.pt').export(format='onnx', imgsz=$(IMG_SIZE))"

clean: ## Remove training runs and cache
	rm -rf runs/autorickshaw/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
