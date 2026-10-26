# Checkpoint evaluation and implementation mapping

## Scope

Load `weights/sirdet_antiuav_320.pt` and evaluate the specified Anti-UAV test collection. A training entry point is not an independently verified reconstruction of the original training/model-selection process. Independent retraining may produce different results.

## Paper-to-code mapping

| Manuscript | Config name | Checkpoint implementation |
|---|---|---|
| Local Context Recalibration | LCR | DualModalRefine in rgbt_guidance.py |
| Scale-wise Infrared Modulation | SIM | TargetAwareGuide, default scalar mode |
| P2-P5 predictions | Detect | Strides 4, 8, 16, 32 |
| WIoU regression | use_wiseiou=True | Project bbox_iou WIoU='v3' mode and 1-score loss |

Aliases preserve parameter names and old pickle imports. LCR is pointwise projection, depthwise encoding, SE-style weights and `x + tanh(gamma) * update`. SIM is a learned one-channel IR response, sigmoid, spatial mean and `visible * (1 + beta * mean)`. It does not transmit an IR spatial map. The unused legacy gate_reduction argument is retained for config compatibility; it is not an extra forward operation.

Loss gains for box, classification and DFL default to 7.5, 0.5, 1.5. The project WIoU retains batch-local normalization, bounded quality gains and its historical distance modulation, and is not numerically identical to every official WIoU implementation. No alternative mode is silently enabled. P5 also retains standard SPPF and C2PSA processing even when abbreviated in the overview figure.

## Evaluation protocol

- Channels: B,G,R,IR; IR resized to visible geometry before shared transforms.
- Coordinate reference and ground truth: visible image and normalized center-format visible YOLO labels.
- Split: test; expected collection 1,820 pairs, 1,691 ground-truth boxes.
- Requested size: 320. With rect=True, padded batch height may be less than 320; requested size does not guarantee square inference tensors.
- Batch 16; FP32; augment=False; confidence floor 0.001; NMS IoU 0.7; max_det 300; workers 0.
- P/R are evaluator operating-point results; F1 = 2PR/(P+R).
- mAP50-95 averages thresholds 0.50:0.05:0.95 for the single class.
- The checkpoint contains 4,325,460 parameters before Conv/BN fusion. The paper's 4,314,732 parameters are measured after standard inference Conv/BN fusion. Both are recorded separately. GFLOPs use explicit `(1,4,320,320)` input and THOP MACs*2, not rectangular inference geometry.
- FPS_total = 1000/(preprocess_ms+inference_ms+postprocess_ms). Inference-only FPS is separately labeled. Evaluator stage timing is not live-camera end-to-end latency.

Report the checkpoint hash, split membership, evaluator version and evaluation settings alongside measured results. Full-precision results are included under `results/`.

Local repository verification on the reference test collection completed successfully: P 0.9621266047, R 0.8863523788, F1 0.9226863950, mAP50 0.9254592027 and mAP50-95 0.5191065542, 4.060228352 GFLOPs. The supplied config with paper aliases loads the checkpoint state dict strictly and produces bit-identical predictions to the original named model on the compatibility-test input. Six automated tests cover aliases, checkpoint hash, strict loading, scalar modulation, training gradients, and dataset-label integrity.

