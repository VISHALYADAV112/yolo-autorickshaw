#!/usr/bin/env bash
# Build an 81-class dataset = 10k custom (Auto Rickshaw, class 80) + representative COCO subset (classes 0-79).
# Uses val2017 (5,000 images, ~1GB) as the lightweight COCO rehearsal subset instead of the 19GB train2017.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/venv/bin/python}"
[ -x "$PY" ] || PY="python3"

CUSTOM="${CUSTOM:-$ROOT/dataset}"          # normalized custom dataset dir with train/ + val/
OUT="${OUT:-$ROOT/dataset81}"              # merged 81-class dataset output dir
WORK="${WORK:-$ROOT/.coco_cache}"          # scratch dir for downloads
COCO_SUBSET="${COCO_SUBSET:-5000}"         # # val2017 images to use as rehearsal subset
COCO_VAL="${COCO_VAL:-500}"                # # of those held out for validation
LABELS_URL="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"  # stable YOLO-format labels
LABELS_URL_FALLBACK="https://ultralytics.com/assets/coco2017labels.zip"
IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"      # 1GB, 5000 images, covers all 80 classes
CURL_OPTS="--retry 3 --connect-timeout 20 --retry-delay 3 -fL"

echo "[1/4] Custom dataset check"
if [ ! -d "$CUSTOM/train/images" ] || [ ! -d "$CUSTOM/val/images" ]; then
  echo "ERROR: expected custom dataset at $CUSTOM with train/images and val/images (run make download first)" >&2
  exit 1
fi

echo "[2/4] Downloading COCO labels + val2017 images"
mkdir -p "$WORK"
cd "$WORK"
COCO_LABELS="$WORK/coco/labels/val2017"   # nested under coco/ in coco2017labels.zip
[ ! -d "$COCO_LABELS" ] && COCO_LABELS="$WORK/labels/val2017"   # legacy flat fallback
if [ ! -d "$COCO_LABELS" ]; then
  curl $CURL_OPTS "$LABELS_URL" -o coco2017labels.zip || \
    curl $CURL_OPTS "$LABELS_URL_FALLBACK" -o coco2017labels.zip
  unzip -q -o coco2017labels.zip -d .
  # resolve whichever layout was extracted
  if [ -d "$WORK/coco/labels/val2017" ]; then COCO_LABELS="$WORK/coco/labels/val2017"; fi
fi
if [ ! -d "val2017" ]; then
  curl $CURL_OPTS "$IMAGES_URL" -o val2017.zip
  unzip -q -o val2017.zip -d .
fi
echo "  labels: $(ls "$COCO_LABELS" | wc -l) txt | images: $(ls val2017 | wc -l) jpg"
echo "  labels dir: $COCO_LABELS"

echo "[3/4] Merging into 81-class dataset ($OUT)"
"$PY" - "$CUSTOM" "$OUT" "$COCO_LABELS" "$WORK/val2017" "$COCO_SUBSET" "$COCO_VAL" <<'PYEOF'
import random
import shutil
import sys
from pathlib import Path

coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]  # 80 COCO classes, ids 0-79

RICK = 80  # Auto Rickshaw class id

def remap_custom(labels_dir, out_labels):
    """Rewrite every custom label from class 0 -> RICK and copy to out_labels."""
    labels_dir, out_labels = Path(labels_dir), Path(out_labels)
    out_labels.mkdir(parents=True, exist_ok=True)
    for txt in sorted(labels_dir.glob("*.txt")):
        lines = [f"{RICK}" + line[1:] for line in txt.read_text().splitlines() if line.strip()]
        (out_labels / txt.name).write_text("\n".join(lines) + ("\n" if lines else ""))

def copy_images(images_dir, out_images, stem):
    """Copy a single image file (any ext) plus an accompanying empty label if needed."""
    images_dir, out_images = Path(images_dir), Path(out_images)
    out_images.mkdir(parents=True, exist_ok=True)
    srcs = sorted(images_dir.glob(f"{stem}.*"))
    if not srcs:
        return False
    shutil.copy2(srcs[0], out_images / srcs[0].name)
    return True

def main(custom, out, coco_labels, coco_imgs, subset_n, val_n):
    custom, out = Path(custom), Path(out)
    coco_labels, coco_imgs = Path(coco_labels), Path(coco_imgs)
    subset_n, val_n = int(subset_n), int(val_n)
    if val_n > subset_n:
        val_n = subset_n

    train_img, train_lab = out / "train" / "images", out / "train" / "labels"
    val_img, val_lab = out / "val" / "images", out / "val" / "labels"
    for d in (train_img, train_lab, val_img, val_lab):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    # 1) custom -> merge (remap 0 -> 80)
    remap_custom(custom / "train" / "labels", train_lab)
    remap_custom(custom / "val" / "labels", val_lab)
    for stem in (p.stem for p in (custom / "train" / "images").glob("*.*")):
        copy_images(custom / "train" / "images", train_img, stem)
    for stem in (p.stem for p in (custom / "val" / "images").glob("*.*")):
        copy_images(custom / "val" / "images", val_img, stem)
    n_custom_tr = len(list(train_img.glob("*.*")))
    n_custom_va = len(list(val_img.glob("*.*")))

    # 2) COCO rehearsal subset (deterministic shuffle, seeded)
    all_stems = sorted(p.stem for p in coco_imgs.glob("*.jpg"))[:subset_n]
    if len(all_stems) < subset_n:
        print(f"WARNING: only {len(all_stems)} COCO images available, using those")
    rng = random.Random(42)
    rng.shuffle(all_stems)
    coco_train, coco_val = all_stems[val_n:], all_stems[:val_n]

    n_coco_tr = n_coco_va = 0
    for stem in coco_train:
        txt = coco_labels / f"{stem}.txt"
        if txt.exists():
            shutil.copy2(txt, train_lab / txt.name)
            n_coco_tr += 1
        copy_images(coco_imgs, train_img, stem)
    for stem in coco_val:
        txt = coco_labels / f"{stem}.txt"
        if txt.exists():
            shutil.copy2(txt, val_lab / txt.name)
            n_coco_va += 1
        copy_images(coco_imgs, val_img, stem)

    # 3) 81-class data.yaml
    names = coco_names + ["Auto Rickshaw"]
    yaml = ["# 81-class: 80 COCO + Auto Rickshaw (class 80)\n",
            f"train: {out}/train/images\n",
            f"val: {out}/val/images\n",
            f"nc: {len(names)}\n",
            "names:\n"]
    for i, n in enumerate(names):
        yaml.append(f"  {i}: {n}\n")
    (out / "data.yaml").write_text("".join(yaml))

    print(f"  train: {len(coco_train) + n_custom_tr} images "
          f"(custom {n_custom_tr} + coco {len(coco_train)})")
    print(f"  val:   {len(coco_val) + n_custom_va} images "
          f"(custom {n_custom_va} + coco {len(coco_val)})")
    print(f"  data.yaml: nc={len(names)}, class 80 = Auto Rickshaw")

main(*sys.argv[1:])
PYEOF

echo "[4/4] Done. Next: make build-dataset81-train"