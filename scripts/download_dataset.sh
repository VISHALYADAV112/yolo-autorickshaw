#!/bin/bash
set -e

echo "========================================="
echo "  Downloading Auto-Rickshaw Dataset"
echo "  Source: Roboflow Universe (CC BY 4.0)"
echo "  Dataset: Auto-Rickshaw - 4514 images"
echo "  Project: autorickshaw-detection/auto-rickshaw-9fpsm (v3)"
echo "========================================="

# Use the project venv directly (no activation needed)
PYTHON="python3"
if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
fi

echo "  Using Python: $($PYTHON --version 2>&1) at $PYTHON"

# Check roboflow package (show the real error if import fails)
if ! "$PYTHON" -c "import roboflow; print('  roboflow', roboflow.__version__)" 2>&1; then
    echo ""
    echo "Error: roboflow could not be imported with $PYTHON."
    echo "Run 'make setup' to (re)install dependencies into the venv."
    exit 1
fi

# Check API key
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "Error: ROBOFLOW_API_KEY not set."
    echo ""
    echo "  1. Create a free account at https://app.roboflow.com"
    echo "  2. Click your username -> Settings -> API Key (copy it)"
    echo "  3. Run: export ROBOFLOW_API_KEY=your_key_here"
    echo "     # add to ~/.bashrc to persist"
    exit 1
fi

# Download dataset
echo "[1/3] Downloading dataset from Roboflow..."
rm -rf dataset
mkdir -p dataset

"$PYTHON" << 'EOF'
from roboflow import Roboflow
import os
from pathlib import Path

key = os.environ["ROBOFLOW_API_KEY"]
# Use an absolute location so extraction lands predictably.
# IMPORTANT: overwrite=True is required by roboflow>=1.4 -- if the dir already
# exists and overwrite=False, it silently returns without downloading anything.
target = os.path.abspath("./dataset")

rf = Roboflow(api_key=key)
project = rf.workspace("autorickshaw-detection").project("auto-rickshaw-9fpsm")
version = project.version(3)
dataset = version.download("yolov8", location=target, overwrite=True)

loc = os.path.abspath(dataset.location)
print(f"Dataset downloaded to: {loc}")
print(f"  files in location: {len(list(Path(loc).rglob('*')))}")
# Write location to a file directly (roboflow emits \r progress on stdout)
Path("/tmp/rf_location.txt").write_text(loc)
EOF

RF_LOCATION="$(cat /tmp/rf_location.txt)"

echo ""
echo "  Raw layout (location, max depth 3):"
find "$RF_LOCATION" -maxdepth 3 2>/dev/null | head -60
echo ""

# If the returned location is empty, search the whole filesystem for the data
if [ -z "$(find "$RF_LOCATION" -mindepth 1 2>/dev/null)" ]; then
    echo "  NOTE: $RF_LOCATION is empty! Searching filesystem for the dataset..."
    echo "  (candidate data.yaml files:)"
    find / -name "data.yaml" -path "*uto*" 2>/dev/null | head -10
    echo "  (candidate train/image dirs:)"
    find / -type d -path "*train/images" 2>/dev/null | head -10
    echo ""
    echo "  -> If the above find nothing in /home/vishal, the download produced"
    echo "     no files. Re-run with 'make setup' if needed, or report the"
    echo "     'files in location' count from the download step."
fi

echo "[2/3] Normalizing dataset structure..."
RF_LOCATION="$RF_LOCATION" "$PYTHON" << 'EOF'
import os
from pathlib import Path

dataset_dir = Path(os.environ["RF_LOCATION"])
print(f"  Normalizing: {dataset_dir}")

# Roboflow may extract into a nested project subfolder, e.g. <slug>-<version>/
# Detect it: the folder that itself contains data.yaml
nested = None
for p in dataset_dir.iterdir():
    if p.is_dir() and (p / "data.yaml").exists():
        nested = p
        break

if nested is not None:
    print(f"  Flattening nested folder: {nested.name}")
    for item in nested.iterdir():
        dest = dataset_dir / item.name
        if not dest.exists():
            item.rename(dest)
    nested.rmdir()

# Rename valid -> val (YOLO/ultralytics default)
valid_dir = dataset_dir / "valid"
val_dir = dataset_dir / "val"
if valid_dir.exists() and not val_dir.exists():
    valid_dir.rename(val_dir)
    print("  Renamed valid -> val")

# Rewrite data.yaml paths to be relative to the dataset directory
yaml_file = dataset_dir / "data.yaml"
if yaml_file.exists():
    lines = yaml_file.read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith("train:"):
            out.append("train: train/images")
        elif line.startswith("val:"):
            out.append("val: val/images")
        else:
            out.append(line)
    yaml_file.write_text("\n".join(out) + "\n")
    print("  data.yaml paths fixed")
    print("  data.yaml:")
    print(yaml_file.read_text())
else:
    print("  WARNING: no data.yaml found in dataset dir")
    for p in dataset_dir.rglob("data.yaml"):
        print(f"    found nested yaml: {p}")

print("  Structure:")
for p in sorted(dataset_dir.iterdir()):
    if p.is_dir():
        print(f"    {p.name}/")
    else:
        print(f"    {p.name}")
EOF

echo "[3/3] Dataset ready!"
echo ""
echo "  Location: $RF_LOCATION"
echo "  Train images: $(find "$RF_LOCATION/train/images" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) 2>/dev/null | wc -l | tr -d ' ')"
echo "  Val images:   $(find "$RF_LOCATION/val/images" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "Run 'make train' to start training."
