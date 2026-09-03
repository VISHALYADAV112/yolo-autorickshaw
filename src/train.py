import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    model_size="yolov8m.pt",
    data_config="dataset/data.yaml",
    epochs=100,
    imgsz=1280,
    batch=-1,
    patience=20,
    device="0",
    workers=16,
    cache="ram",
    project="runs",
    name="autorickshaw",
):
    print(f"Training YOLOv8 for autorickshaw detection")
    print(f"  Model: {model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}  ({'auto-detected' if batch == -1 else batch})")
    print(f"  Device: {device}")
    print(f"  Workers: {workers}")
    print(f"  Cache: {cache}")
    print()

    model = YOLO(model_size)

    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,          # -1 = auto-select largest batch that fits VRAM
        patience=patience,
        device=device,        # "0", "0,1,2,3" for multi-GPU, "cpu"
        workers=workers,      # parallel data loading
        cache=cache,          # "ram" to prefetch entire dataset into RAM
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        amp=True,
        verbose=True,
        seed=42,
        plots=True,
    )

    print()
    print(f"Training complete!")
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    print(f"Last model saved to: {project}/{name}/weights/last.pt")
    print(f"Results plot: {project}/{name}/results.png")

    return results


def evaluate(model_path, data_config="dataset/data.yaml", device="0", imgsz=1280, batch=32):
    print(f"Evaluating model: {model_path}")
    model = YOLO(model_path)
    results = model.val(data=data_config, device=device, imgsz=imgsz, batch=batch)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    return results


def predict(model_path, source, device="0", conf=0.25, imgsz=1280):
    print(f"Running inference on: {source}")
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        device=device,
        conf=conf,
        imgsz=imgsz,
        save=True,
        project="runs",
        name="predict",
        exist_ok=True,
    )
    print(f"Results saved to: runs/predict/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Autorickshaw Training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model", default="yolov8m.pt", help="Base model (default: yolov8m.pt)")
    train_parser.add_argument("--data", default="dataset/data.yaml", help="Data config path")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    train_parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto)")
    train_parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    train_parser.add_argument("--device", default="0", help="Device: '0', '0,1,2,3' for multi-GPU, 'cpu'")
    train_parser.add_argument("--workers", type=int, default=16, help="Data loader workers")
    train_parser.add_argument("--cache", default="ram", help="Cache mode: 'ram', 'disk', False")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--model", required=True, help="Path to model weights")
    eval_parser.add_argument("--data", default="dataset/data.yaml", help="Data config path")
    eval_parser.add_argument("--device", default="0", help="Device")
    eval_parser.add_argument("--imgsz", type=int, default=1280, help="Image size")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--model", required=True, help="Path to model weights")
    predict_parser.add_argument("--source", required=True, help="Image/video/folder path")
    predict_parser.add_argument("--device", default="0", help="Device")
    predict_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    predict_parser.add_argument("--imgsz", type=int, default=1280, help="Image size")

    args = parser.parse_args()

    if args.command == "train":
        train(
            model_size=args.model,
            data_config=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
        )
    elif args.command == "eval":
        evaluate(args.model, args.data, args.device, args.imgsz)
    elif args.command == "predict":
        predict(args.model, args.source, args.device, args.conf, args.imgsz)
    else:
        parser.print_help()
