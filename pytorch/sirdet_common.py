"""Local runtime and paired-dataset helpers shared by SIRDet entry points."""

from __future__ import annotations

import os
import shutil
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent
CONFIG_DIR = ROOT / ".runtime"
CONFIG_DIR.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(CONFIG_DIR)

import yaml
import ultralytics
from ultralytics.utils import USER_CONFIG_DIR

if Path(ultralytics.__file__).resolve().parent != ROOT / "ultralytics":
    raise RuntimeError("Import the Ultralytics copy shipped in this repository, not the stock pip package.")
font = ROOT / "ultralytics/Arial.ttf"
if font.is_file():
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    target = USER_CONFIG_DIR / "Arial.ttf"
    if not target.exists():
        shutil.copy2(font, target)

MODEL = ROOT / "configs/sirdet.yaml"
WEIGHTS = BUNDLE / "weights/sirdet_antiuav_320.pt"
DATA = ROOT / "configs/anti_uav_rgbt.yaml"


def resolve_data(data_file=DATA, data_root=None, split="test"):
    data_file = Path(data_file).resolve()
    data = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    if data.get("channels") != 4 or str(data.get("use_simotm", "")).upper() != "RGBT":
        raise ValueError("Set channels: 4 and use_simotm: RGBT in the dataset YAML.")
    if data.get("pairs_rgb_ir") != ["visible", "infrared"]:
        raise ValueError("This release expects pairs_rgb_ir: [visible, infrared].")
    root = Path(data_root if data_root is not None else data.get("path", "."))
    if not root.is_absolute():
        root = data_file.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root missing: {root}. Supply --data-root or --data.")
    splits = ("train", "val") if split == "train" else (split,)
    for name in splits:
        folder = root / data[name]
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset {name} directory missing: {folder}")
    data["path"] = str(root)
    output = CONFIG_DIR / f"data_{os.getpid()}.yaml"
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def verify_test_collection(resolved_data):
    """Fail on a different collection rather than silently label it a reproduction."""
    manifest = BUNDLE / "data/test_pairs.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    data = yaml.safe_load(Path(resolved_data).read_text(encoding="utf-8"))
    root = Path(data["path"])
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {row["visible"] for row in rows}
    actual = {p.relative_to(root).as_posix() for p in (root / data["test"]).rglob("*")
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    if expected != actual:
        raise ValueError("Test membership differs from data/test_pairs.csv. Use --allow-different-test-set only for a new experiment.")
    for row in rows:
        if not (root / row["infrared"]).is_file():
            raise FileNotFoundError(root / row["infrared"])
        label = root / row["visible_label"]
        if not label.is_file() or hashlib.sha256(label.read_bytes()).hexdigest() != row["visible_label_sha256"]:
            raise ValueError(f"Visible label missing or changed: {row['visible_label']}")
    return dict(pairs=len(rows), boxes=sum(int(row["boxes"]) for row in rows),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
