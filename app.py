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


def get_class_names(model):
    names = model.names
    # names is dict {0:'...', 1:'...'} -> ordered list for display
    return {int(k): v for k, v in names.items()}


def detect(model, image, conf, iou, selected_names):
    if image is None:
        return None, 0
    names = get_class_names(model)
    # selected_names is a list of class name strings from the UI
    if selected_names:
        classes = [k for k, v in names.items() if v in selected_names]
    else:
        classes = None  # detect all classes
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        classes=classes,
        imgsz=1280,
        device="0",
        verbose=False,
    )
    r = results[0]
    annotated = r.plot()
    counts = len(r.boxes) if r.boxes is not None else 0
    return annotated, counts


def run_tab(model):
    class_names = list(get_class_names(model).values())

    def fn(image, conf, iou, selected):
        out, counts = detect(model, image, conf, iou, selected)
        return out, counts

    with gr.Blocks() as demo:
        gr.Markdown(
            "## Auto Rickshaw Detection\n"
            "Upload an image or use your webcam to detect auto rickshaws with the trained YOLOv8 model.\n"
            "Select the classes you want to detect (leave empty to detect all)."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Input image")
                class_dropdown = gr.Dropdown(
                    choices=class_names,
                    value=class_names,
                    multiselect=True,
                    label="Classes to detect",
                    info="Select one or more classes. Empty = detect all.",
                )
                conf_slider = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="Confidence threshold")
                iou_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="IoU threshold")
                run_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                image_output = gr.Image(label="Detected auto rickshaws")
                count_output = gr.Number(label="Number of detections")
        run_btn.click(
            fn,
            inputs=[image_input, conf_slider, iou_slider, class_dropdown],
            outputs=[image_output, count_output],
        )
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
