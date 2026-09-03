import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    model_size="yolov8m.pt",
    data_config="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device="0",
    project="runs",
    name="autorickshaw",
):
    print(f"Training YOLOv8 for autorickshaw detection")
    print(f"  Model: {model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print()

    model = YOLO(model_size)

    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,
    )

    print()
    print(f"Training complete!")
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    print(f"Last model saved to: {project}/{name}/weights/last.pt")
    print(f"Results plot: {project}/{name}/results.png")

    return results


def evaluate(model_path, data_config="dataset/data.yaml", device="0"):
    print(f"Evaluating model: {model_path}")
    model = YOLO(model_path)
    results = model.val(data=data_config, device=device)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    return results


def predict(model_path, source, device="0", conf=0.25):
    print(f"Running inference on: {source}")
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        device=device,
        conf=conf,
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
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    train_parser.add_argument("--device", default="0", help="Device (0, 1, cpu)")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--model", required=True, help="Path to model weights")
    eval_parser.add_argument("--data", default="dataset/data.yaml", help="Data config path")
    eval_parser.add_argument("--device", default="0", help="Device")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--model", required=True, help="Path to model weights")
    predict_parser.add_argument("--source", required=True, help="Image/video/folder path")
    predict_parser.add_argument("--device", default="0", help="Device")
    predict_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

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
        )
    elif args.command == "eval":
        evaluate(args.model, args.data, args.device)
    elif args.command == "predict":
        predict(args.model, args.source, args.device, args.conf)
    else:
        parser.print_help()
