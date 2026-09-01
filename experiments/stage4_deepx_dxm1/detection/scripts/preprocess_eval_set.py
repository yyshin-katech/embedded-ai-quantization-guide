#!/usr/bin/env python3
"""Build the COCO val2017 detection eval subset for the DX-M1 accuracy axis.

Letterboxes the first N val2017 images (sorted by filename, deterministic) to
640x640 uint8 NHWC RGB and stacks them into ONE .npy that transfers to the Pi.
Both the FP32 reference (onnxruntime here) and the INT8 NPU (dx_engine on Pi) read
this identical array -> letterbox geometry is bit-identical across backends, so any
mAP gap isolates quantization (same rationale as the accuracy axis' rpi_sub_u8.npy).

Emits:
  <out_npy>            [N,640,640,3] uint8 RGB (large, scratchpad; transferred to Pi)
  results/eval_meta.json   per-image {image_id, file, r, left, top, w0, h0} + order
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from yolo_det_common import letterbox

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="/home/yuyeong/embedded-ai-quantization-guide/_workspace/coco")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out_npy", required=True)
    args = ap.parse_args()

    img_dir = os.path.join(args.coco, "val2017")
    files = sorted(os.listdir(img_dir))[: args.n]
    arrs, meta = [], []
    for fn in files:
        im = Image.open(os.path.join(img_dir, fn)).convert("RGB")
        canvas, r, left, top, w0, h0 = letterbox(im)
        arrs.append(canvas)
        meta.append({"image_id": int(os.path.splitext(fn)[0]), "file": fn,
                     "r": r, "left": left, "top": top, "w0": w0, "h0": h0})
    u8 = np.stack(arrs).astype(np.uint8)  # [N,640,640,3] RGB
    np.save(args.out_npy, u8)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "eval_meta.json"), "w") as f:
        json.dump({"n": len(meta), "layout": "NHWC_RGB_uint8_letterbox640_pad114_centered",
                   "images": meta}, f)
    print(f"saved {args.out_npy} shape={u8.shape} dtype={u8.dtype} "
          f"({u8.nbytes/1e6:.1f} MB); meta n={len(meta)}")


if __name__ == "__main__":
    main()
