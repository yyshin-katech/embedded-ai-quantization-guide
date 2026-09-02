#!/usr/bin/env python3
"""Build the COCO val2017 DETR eval subset for the DX-M1 axis.

Force-resizes the first N val2017 images (sorted by filename, deterministic) to
800x1066 uint8 NHWC RGB and stacks them into ONE .npy that transfers to the Pi. Both
the FP32 reference (onnxruntime here, detr_2out.onnx) and the INT8 NPU (dx_engine on Pi,
detr_2out.dxnn) read this identical array -> the pixels are bit-identical across backends,
so any mAP gap isolates quantization (same rationale as the detection axis' eval_u8.npy).

Emits:
  <out_npy>              [N,800,1066,3] uint8 RGB (large; transferred to Pi)
  results/eval_meta.json  per-image {image_id, file, w0, h0} + order (w0,h0 = ORIGINAL
                          COCO size, needed by the DETR box postprocess)
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from detr_det_common import load_u8

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
        path = os.path.join(img_dir, fn)
        w0, h0 = Image.open(path).size                      # ORIGINAL (W,H) for box scaling
        arrs.append(load_u8(path))                          # (800,1066,3) uint8 RGB
        meta.append({"image_id": int(os.path.splitext(fn)[0]), "file": fn, "w0": w0, "h0": h0})
    u8 = np.stack(arrs).astype(np.uint8)                    # [N,800,1066,3] RGB
    np.save(args.out_npy, u8)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "eval_meta.json"), "w") as f:
        json.dump({"n": len(meta), "layout": "NHWC_RGB_uint8_forceresize_800x1066",
                   "images": meta}, f)
    print(f"saved {args.out_npy} shape={u8.shape} dtype={u8.dtype} "
          f"({u8.nbytes/1e6:.1f} MB); meta n={len(meta)}")


if __name__ == "__main__":
    main()
