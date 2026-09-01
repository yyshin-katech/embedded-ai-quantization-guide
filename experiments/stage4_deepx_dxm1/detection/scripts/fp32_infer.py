#!/usr/bin/env python3
"""FP32 reference: run yolo26n.onnx (float) over the eval .npy via onnxruntime.

Reads the SAME letterboxed uint8 array the NPU reads, converts to the float
[1,3,640,640] RGB/255 the ONNX graph expects, decodes the shared [1,300,6] end2end
output. Emits predictions_fp32.json (COCO-format) — the baseline the INT8 .dxnn mAP
is subtracted from (quantization isolated by identical decode).
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort

from yolo_det_common import decode

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "predictions_fp32.json"))
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--cpu", action="store_true", help="force CPUExecutionProvider")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(ROOT, "results", "eval_meta.json")))["images"]
    u8 = np.load(args.npy)  # [N,640,640,3] RGB
    assert u8.shape[0] == len(meta), (u8.shape, len(meta))

    prov = ["CPUExecutionProvider"] if args.cpu else \
        (["CUDAExecutionProvider", "CPUExecutionProvider"]
         if "CUDAExecutionProvider" in ort.get_available_providers()
         else ["CPUExecutionProvider"])
    so = ort.InferenceSession(args.onnx, providers=prov)
    used = so.get_providers()[0]

    dets, lat = [], []
    for i, m in enumerate(meta):
        x = (u8[i].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # RGB NCHW
        t = time.perf_counter()
        out = so.run(None, {"images": x})[0][0]  # [300,6]
        lat.append((time.perf_counter() - t) * 1e3)
        dets += decode(out, m["image_id"], m["r"], m["left"], m["top"],
                       m["w0"], m["h0"], conf=args.conf)
    json.dump(dets, open(args.out, "w"))
    lat = np.array(lat)
    print(f"provider={used} images={len(meta)} dets={len(dets)} "
          f"lat_p50={np.percentile(lat,50):.2f}ms mean={lat.mean():.2f}ms -> {args.out}")


if __name__ == "__main__":
    main()
