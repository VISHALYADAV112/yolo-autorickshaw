import argparse
from pathlib import Path

import gradio as gr
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/autorickshaw/weights/best.pt"

# Resolve the same model path used by train.py
CANDIDATES = [
    DEFAULT_WEIGHTS,
    Path.home() / "model_comparison_lab/runs/detect/runs/autorickshaw/weights/best.pt",
]


def load_model(weights):
    return YOLO(weights)


def detect(model, image, conf, iou):
    if image is None:
        return None
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=1280,
        device="0",
        verbose=False,
    )
    r = results[0]
    annotated = r.plot()
    counts = len(r.boxes) if r.boxes is not None else 0
    return annotated, counts


def run_tab(model):
    def fn(image, conf, iou):
        out, counts = detect(model, image, conf, iou)
        return out, counts

    with gr.Blocks() as demo:
        gr.Markdown(
            "## Auto Rickshaw Detection\n"
            "Upload an image or use your webcam to detect auto rickshaws with the trained YOLOv8 model."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Input image")
                conf_slider = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="Confidence threshold")
                iou_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="IoU threshold")
                run_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                image_output = gr.Image(label="Detected auto rickshaws")
                count_output = gr.Number(label="Number of auto rickshaws detected")
        run_btn.click(fn, inputs=[image_input, conf_slider, iou_slider], outputs=[image_output, count_output])
    return demo


def main():
    parser = argparse.ArgumentParser(description="Auto Rickshaw detection dashboard")
    parser.add_argument("--model", default=DEFAULT_WEIGHTS, help="Path to model weights")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (0.0.0.0 exposes to network)")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        for cand in CANDIDATES:
            if Path(cand).exists():
                model_path = Path(cand)
                break
    print(f"Loading model: {model_path}")

    model = load_model(str(model_path))
    demo = run_tab(model)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
