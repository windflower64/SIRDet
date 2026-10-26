# RKNN model placement

Place the matching SIRDet RKNN model generated from the 320x192 raw-DFL export in this directory, or pass its absolute path to the command-line programs. The repository includes a PyTorch checkpoint under the root weights directory, but no RKNN binary is bundled here.

Expected contract for the formal deployment result:

- input: NHWC `1x192x320x4`, channel order B, G, R, IR
- numeric format: INT8
- outputs: merged raw-DFL regression and score tensors
- postprocessing: sparse INT8 score filtering, local dequantization, DFL decoding, and NMS
