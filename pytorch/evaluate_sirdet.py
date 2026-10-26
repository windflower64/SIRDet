"""Reproduce test metrics from the unchanged released SIRDet checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from sirdet_common import DATA, ROOT, WEIGHTS, resolve_data, verify_test_collection
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="?", type=Path, default=WEIGHTS)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/evaluate")
    parser.add_argument("--name", default="sirdet_test")
    parser.add_argument("--allow-different-test-set", action="store_true",
                        help="Explicitly evaluate a new collection, without claiming exact test-set reproduction.")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    torch.set_num_threads(4)
    data = resolve_data(args.data, args.data_root, args.split)
    collection = verify_test_collection(data) if args.split == "test" and not args.allow_different_test_set else None
    model = YOLO(str(args.weights))
    checkpoint_params = sum(p.numel() for p in model.model.parameters())
    if int(model.model.yaml.get("ch", 0)) != 4:
        raise ValueError("Expected a four-channel RGBT checkpoint.")
    validators = []

    def validator_factory(**kwargs):
        validator = DetectionValidator(**kwargs)
        validators.append(validator)
        return validator

    metrics = model.val(validator=validator_factory, data=str(data), imgsz=args.imgsz, batch=args.batch, device=args.device,
                        split=args.split, conf=0.001, iou=0.7, max_det=300, rect=True,
                        half=False, workers=args.workers, augment=False, plots=False,
                        save_json=False, project=str(args.project), name=args.name,
                        exist_ok=False, verbose=False)
    precision, recall = float(metrics.box.mp), float(metrics.box.mr)
    speed = {key: float(metrics.speed.get(key, 0)) for key in ("preprocess", "inference", "postprocess")}
    total_ms = sum(speed.values())
    validator = validators[0]
    params = sum(p.numel() for p in model.model.parameters())
    report = dict(model="SIRDet", checkpoint_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  ultralytics_version=ultralytics.__version__, torch_version=torch.__version__,
                  P=precision, R=recall, F1=2 * precision * recall / (precision + recall + 1e-16),
                  mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                  parameters=params, Params_M=params / 1e6, parameters_checkpoint=checkpoint_params,
                  parameters_definition="after standard inference Conv/BN fusion, matching paper profiling",
                  images=int(validator.seen), ground_truth_boxes=int(metrics.nt_per_class.sum()), speed_ms=speed,
                  test_collection=collection,
                  FPS_total=1000 / total_ms if total_ms else None,
                  FPS_inference=1000 / speed["inference"] if speed["inference"] else None,
                  fps_definition="1000 / (preprocess_ms + inference_ms + postprocess_ms)",
                  evaluation=dict(split=args.split, imgsz=args.imgsz, batch=args.batch, conf=0.001,
                                  iou=0.7, max_det=300, rect=True, half=False, augment=False,
                                  channels=4, workers=args.workers, device=args.device))
    try:
        import thop
        net = deepcopy(model.model).cpu().float().eval()
        macs, _ = thop.profile(net, inputs=(torch.zeros(1, 4, args.imgsz, args.imgsz),), verbose=False)
        report["GFLOPs"] = float(macs * 2 / 1e9)
    except Exception as error:
        report["GFLOPs"] = None
        report["profiling_error"] = str(error)
    output = Path(validator.save_dir) / "metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
