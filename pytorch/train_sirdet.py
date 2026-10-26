"""Train the paper architecture; checkpoint evaluation is a separate workflow."""

import argparse
from pathlib import Path

from sirdet_common import DATA, MODEL, ROOT, resolve_data
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="sirdet_seed0")
    args = parser.parse_args()
    data = resolve_data(args.data, args.data_root, "train")
    YOLO(str(MODEL)).train(data=str(data), imgsz=320, epochs=args.epochs, batch=args.batch,
                          device=args.device, workers=args.workers, optimizer="SGD", lr0=0.005,
                          lrf=0.01, cos_lr=True, momentum=0.937, weight_decay=0.0005,
                          warmup_epochs=3.0, close_mosaic=min(10, args.epochs),
                          seed=args.seed, deterministic=True, pretrained=False, amp=True,
                          use_wiseiou=True, use_official_wiou=False, use_ema_wiseiou=False,
                          use_st_iou=False, project=str(ROOT / "runs/train"), name=args.name,
                          exist_ok=False, plots=False, compile=False)


if __name__ == "__main__":
    main()
