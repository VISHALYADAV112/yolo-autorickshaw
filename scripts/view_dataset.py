#!/usr/bin/env python3
"""Inspect a YOLO dataset: sample images with ground-truth boxes and per-class stats.

Usage:
  python scripts/view_dataset.py --data dataset81/data.yaml --out runs/dataset_preview.jpg [--split val] [--samples 12]
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_labels(label_path):
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.split())
            boxes.append((int(cls), xc, yc, w, h))
    return boxes


def draw_boxes(img, boxes, names):
    H, W = img.shape[:2]
    for cls, xc, yc, w, h in boxes:
        x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
        x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
        color = (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, names.get(cls, str(cls)), (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def build_grid(images, cols=4):
    rows = (len(images) + cols - 1) // cols
    cell = images[0].shape[1]
    grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img[:cell, :cell]
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out", default="runs/dataset_preview.jpg")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_yaml(args.data)
    names = {int(k): v for k, v in cfg["names"].items()} if isinstance(cfg["names"], dict) \
        else {i: v for i, v in enumerate(cfg["names"])}
    img_root = Path(cfg[args.split])

    images = sorted(img_root.glob("*.jpg")) + sorted(img_root.glob("*.png")) + sorted(img_root.glob("*.jpeg"))
    if not images:
        print(f"No images found in {img_root}")
        return
    random.seed(args.seed)
    sample = random.sample(images, min(args.samples, len(images)))

    per_class = defaultdict(lambda: [0, []])  # cls -> [count, [relative areas]]
    for ip in images:
        lp = ip.parent.parent / "labels" / (ip.stem + ".txt")
        for cls, xc, yc, w, h in read_labels(lp):
            per_class[cls][0] += 1
            per_class[cls][1].append(w * h)

    print(f"Split: {args.split}  |  images: {len(images)}")
    print(f"{'class':<22} {'#boxes':>7} {'avg rel area':>12} {'area<1%':>9}")
    for cls in sorted(per_class):
        count, areas = per_class[cls]
        avg = np.mean(areas)
        tiny = sum(1 for a in areas if a < 0.01)
        print(f"{names.get(cls, str(cls)):<22} {count:>7} {avg:>12.5f} {tiny:>9}")

    panels = []
    first = None
    to_resize = []
    for ip in sample:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        lp = ip.parent.parent / "labels" / (ip.stem + ".txt")
        boxes = read_labels(lp)
        img = draw_boxes(img, boxes, names)
        panels.append(img)
        if first is None:
            first = img

    if panels:
        cell = 480
        resized = [cv2.resize(p, (cell, cell)) for p in panels]
        grid = build_grid(resized)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), grid)
        print(f"\nPreview saved to {out}  (shows {len(panels)} images with GT boxes)")


if __name__ == "__main__":
    main()