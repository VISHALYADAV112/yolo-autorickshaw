.PHONY: setup download train train-bg eval predict dashboard dashboard-bg export clean help

PYTHON := python3
VENV := venv
MODEL := yolov8m.pt
EPOCHS := 100
IMG_SIZE := 1280
BATCH := -1
DEVICE := 0
WORKERS := 16
DATA := dataset/data.yaml
PORT := 7860
HOST := $(shell hostname -I 2>/dev/null | awk '{print $$1}')
FORMAT := onnx

help: ## Show this help
	@echo "Usage: make <command>"
	@echo ""
	@echo "Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup: ## Install dependencies and setup environment
	bash scripts/setup.sh

download: ## Download dataset from Roboflow (requires ROBOFLOW_API_KEY)
	bash scripts/download_dataset.sh

train: ## Train YOLOv8 medium (L40S defaults: imgsz 1280, batch auto, cache RAM)
	$(VENV)/bin/python src/train.py train \
		--model $(MODEL) \
		--data $(DATA) \
		--epochs $(EPOCHS) \
		--imgsz $(IMG_SIZE) \
		--batch $(BATCH) \
		--device $(DEVICE) \
		--workers $(WORKERS) \
		--cache ram

train-bg: ## Train in background (survives SSH disconnect): nohup + log to train.log
	nohup $(VENV)/bin/python src/train.py train \
		--model $(MODEL) \
		--data $(DATA) \
		--epochs $(EPOCHS) \
		--imgsz $(IMG_SIZE) \
		--batch $(BATCH) \
		--device $(DEVICE) \
		--workers $(WORKERS) \
		--cache ram > train.log 2>&1 &
	@echo "Training started in background (PID $$!). Watch: tail -f train.log"

train-small: ## Train YOLOv8 small (faster, less accurate)
	$(VENV)/bin/python src/train.py train --model yolov8s.pt --data $(DATA) --epochs $(EPOCHS) --imgsz 1024 --batch -1

train-large: ## Train YOLOv8 large (slower, more accurate)
	$(VENV)/bin/python src/train.py train --model yolov8l.pt --data $(DATA) --epochs $(EPOCHS) --imgsz 1280 --batch -1

train-fast: ## Quick experiment: fewer epochs, smaller imgsz
	$(VENV)/bin/python src/train.py train --model $(MODEL) --data $(DATA) --epochs 20 --imgsz 640 --batch -1

train-multi: ## Multi-GPU training (make train-multi DEVICE="0,1" EPOCHS=100)
	$(VENV)/bin/python src/train.py train --model $(MODEL) --data $(DATA) --epochs $(EPOCHS) --device "$(DEVICE)" --workers $(WORKERS) --cache ram

eval: ## Evaluate trained model
	$(VENV)/bin/python src/train.py eval --model runs/autorickshaw/weights/best.pt --data $(DATA)

predict: ## Run inference on image/folder (make predict SOURCE=path/to/image)
	$(VENV)/bin/python src/train.py predict --model runs/autorickshaw/weights/best.pt --source $(SOURCE)

dashboard: ## Launch Gradio web dashboard for testing (Open http://SERVER:7860)
	$(VENV)/bin/python app.py --model runs/autorickshaw/weights/best.pt --host 0.0.0.0 --port $(PORT)

dashboard-bg: ## Run dashboard in background on port 7860 (survives disconnect)
	nohup $(VENV)/bin/python app.py --model runs/autorickshaw/weights/best.pt --host 0.0.0.0 --port 7860 > dashboard.log 2>&1 &
	@echo "Dashboard started at http://$(HOST):7860  (log: dashboard.log)"

export: ## Export model to ONNX (portable default)
	$(VENV)/bin/python src/export.py --model runs/autorickshaw/weights/best.pt --format $(FORMAT) --imgsz $(IMG_SIZE)

export-onnx: ## Export to ONNX (universal, deployable anywhere)
	$(VENV)/bin/python src/export.py --model runs/autorickshaw/weights/best.pt --format onnx --imgsz $(IMG_SIZE)

export-tensorrt: ## Export to TensorRT engine (NVIDIA Jetson/edge GPUs)
	$(VENV)/bin/python src/export.py --model runs/autorickshaw/weights/best.pt --format engine --device 0 --imgsz $(IMG_SIZE)

export-int8: ## Export to int8 quantized ONNX (4x smaller, runs fast on CPU/NPU)
	$(VENV)/bin/python src/export.py --model runs/autorickshaw/weights/best.pt --format onnx --int8 --imgsz $(IMG_SIZE)

benchmark: ## Benchmark exported model (make benchmark FORMAT=onnx)
	$(VENV)/bin/python src/export.py --model runs/autorickshaw/weights/best.pt --benchmark --format $(FORMAT) --device 0

clean: ## Remove training runs and cache
	rm -rf runs/autorickshaw/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
