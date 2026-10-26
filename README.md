# SIRDet

**SIRDet: Lightweight RGB Thermal UAV Detection with Infrared Guidance**

Official implementation and trained checkpoint for **SIRDet: Lightweight RGB Thermal UAV Detection with Infrared Guidance**, including paired RGB-thermal evaluation and RK3566 deployment source.

## Model and terminology

SIRDet retains visible-image coordinates as the detection reference. **Local Context Recalibration (LCR)** performs local encoding and channel recalibration independently in each modality. **Scale-wise Infrared Modulation (SIM)** spatially averages each scale's learned infrared response into a sample-dependent scalar gain and modulates the corresponding visible feature. Predictions use P2-P5 and the project WIoU regression implementation.

## Released checkpoint

| File | Parameters | Checkpoint evaluation |
|---|---:|---|
| [`weights/sirdet_antiuav_320.pt`](weights/sirdet_antiuav_320.pt) | 4,314,732 | P 0.9621, R 0.8864, F1 0.9227, mAP50 0.9255, mAP50-95 0.5191 |

SHA-256: `10e743adeab01cdadd4e539a5e05adf0e7bd37df3f619c810c7a56a347345c62`. Only load PyTorch checkpoints from trusted sources. See [reproduction notes](docs/REPRODUCIBILITY.md) for implementation details and evaluation settings.

## Setup

Enter `pytorch/`. Use Python 3.10-3.12 and install a matching CUDA torch/torchvision pair appropriate for your GPU. Reference environment: Python 3.12.3, torch 2.3.1+cu118, torchvision 0.18.1+cu118, Ultralytics 8.4.56. Keep a working server CUDA stack rather than blindly replacing it.

```bash
cd pytorch
python -m pip install -r requirements.txt
python -c "import ultralytics; print(ultralytics.__file__)"
```

The imported path must be this repository's `pytorch/ultralytics/`. Do not install/upgrade a stock Ultralytics package over these modules. Weights are stored directly in Git; Git LFS is not required.

## Dataset

Obtain original Anti-UAV data under its access terms. Images/annotations are not redistributed here. The evaluation collection contains 1,820 pairs and 1,691 visible ground-truth boxes. Exact membership, not merely matching counts, is needed for exact reproduction.

```text
Anti_UAV_6K/
  train/visible/{images,labels}/
  train/infrared/{images,labels}/
  val/visible/{images,labels}/
  val/infrared/{images,labels}/
  test/visible/{images,labels}/
  test/infrared/{images,labels}/
```

Paired image filenames/extensions must match. Visible labels use normalized YOLO center-x, center-y, width, height. Evaluation uses visible labels. The loader reads BGR + grayscale IR, resizes IR to the visible grid, then applies shared transforms.

## Evaluate

From `pytorch/`, substitute your dataset root:

```bash
python evaluate_sirdet.py --data-root /path/to/Anti_UAV_6K --device 0
```

Defaults: test split, size 320, batch 16, FP32, confidence 0.001, NMS IoU 0.7, max_det 300, rectangular batches, no augmentation, workers 0. Results, checkpoint hash, settings, speed and FLOPs are saved to a new `runs/evaluate/sirdet_test*/metrics.json`. FPS depends on hardware and system load.

The evaluator checks test membership and visible-label byte hashes against [`data/test_pairs.csv`](data/test_pairs.csv). For an intentionally different dataset, add `--allow-different-test-set` and do not describe that run as exact test-set reproduction. No source image pixels are included in the manifest.

Full-precision evaluation results are available in [`results/sirdet_test_metrics.json`](results/sirdet_test_metrics.json).

## Training and export

```bash
python train_sirdet.py --data-root /path/to/Anti_UAV_6K --device 0
python export_sirdet.py --imgsz 192 320
```

Training uses the paper architecture and a user-supplied train/val split, starting from random initialization. It is not required for checkpoint evaluation and is not a promise to regenerate the same checkpoint or accuracy. The ONNX export is a decoded detector graph, NOT the raw-DFL graph expected by the RKNN programs. Install ONNX dependencies separately when exporting.

The supplied configuration uses the project's `use_wiseiou=True` implementation. See the reproduction notes for the regression-loss settings.

## RK3566 source

`deployment/board/` contains paired-image inference and timing programs. `deployment/video/` contains paired-video processing. Both target the documented NHWC `1x192x320x4` B,G,R,IR INT8 raw-DFL interface with CPU DFL decoding and NMS, RK3566, RKNN Runtime 2.3.2 and OpenCV 4.x development files.

The supplied checkpoint is for PyTorch. RK3566 execution requires a separately converted RKNN graph with matching preprocessing and runtime settings. See the [board setup](deployment/board/README.md) for build and inference commands.

## Checks and license

```bash
cd ..
python scripts/verify_manifest.py
cd pytorch
python -m pytest tests -q
```

The modified Ultralytics runtime remains subject to [AGPL-3.0](LICENSE). Preserve upstream notices when using or redistributing the implementation.
