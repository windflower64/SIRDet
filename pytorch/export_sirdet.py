"""Export the released checkpoint to decoded ONNX; not the raw-DFL RKNN graph."""

import argparse
from pathlib import Path

from sirdet_common import WEIGHTS
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="?", type=Path, default=WEIGHTS)
    parser.add_argument("--imgsz", nargs=2, type=int, default=(192, 320), metavar=("HEIGHT", "WIDTH"))
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    YOLO(str(args.weights)).export(format="onnx", imgsz=tuple(args.imgsz), batch=1,
                                  dynamic=False, nms=False, simplify=False, device="cpu")


if __name__ == "__main__":
    main()
