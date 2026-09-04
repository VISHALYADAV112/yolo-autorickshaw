import argparse
import subprocess
from pathlib import Path

from ultralytics import YOLO


def resolve_model(model):
    p = Path(model)
    if p.exists():
        return str(p)
    # fall back to the ultralytics default logging location used during training
    alt = Path.home() / "model_comparison_lab/runs/detect/runs/autorickshaw/weights/best.pt"
    if alt.exists():
        return str(alt)
    raise FileNotFoundError(f"Model not found: {model}")


def export(model_path, fmt, imgsz, int8):
    model = YOLO(model_path)
    kwargs = dict(format=fmt, imgsz=imgsz, simplify=True)
    if int8:
        kwargs["int8"] = True
        kwargs["dynamic"] = False
    elif fmt in ("onnx", "torchscript", "openvino", "tflite"):
        # fp32/fp16 keep dynamic-less for edge simplicity; onnx stays default
        kwargs["half"] = False
    print(f"Exporting {fmt} ...")
    exported = model.export(**kwargs)
    print(f"Exported: {exported}")
    return exported


def benchmark(model_path, fmt, device):
    # benchmark the exported model against its own weights on the provided device
    cmd = [
        "yolo",
        "benchmark",
        "model=" + model_path,
        "format=" + fmt,
        "device=" + device,
        "imgsz=640",
        "half=False",
    ]
    print("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Deploy/export the YOLO autorickshaw model for edge devices")
    parser.add_argument("--model", default="runs/autorickshaw/weights/best.pt")
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "torchscript", "tflite", "openvino", "ncnn"])
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--int8", action="store_true", help="Quantize to int8 (ONNX)")
    parser.add_argument("--device", default="0", help="cuda device id or 'cpu'")
    parser.add_argument("--benchmark", action="store_true", help="Run yolo benchmark")
    args = parser.parse_args()

    model = resolve_model(args.model)
    print(f"Using model: {model}")

    if args.benchmark:
        benchmark(model, args.format, args.device)
    else:
        export(model, args.format, args.imgsz, args.int8)


if __name__ == "__main__":
    main()
