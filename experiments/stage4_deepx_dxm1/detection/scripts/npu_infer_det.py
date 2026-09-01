#!/usr/bin/env python3
"""DX-M1 INT8 detection inference (run ON the Pi). Reads the SAME letterboxed eval
.npy the FP32 reference read, feeds each frame to the .dxnn via dx_engine, decodes the
shared [1,300,6] end2end output to COCO-format detections. Emits predictions_npu_*.json
(scored off-device with the same eval_map.py -> quantization isolated) + per-image
latency. dx_engine input contract = uint8 NHWC [1,640,640,3] (preprocessing folded).

--color rgb (default): feed the RGB array as-is (dx_com folds Div/Transpose but the
convertColor step is NOT folded, so the caller supplies final RGB — same as the
resnet50 accuracy axis). --color bgr reverses channels (sanity control)."""
import argparse
import json
import os
import time

import numpy as np

from yolo_det_common import decode
from dx_engine import InferenceEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--color", choices=["rgb", "bgr"], default="rgb")
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    meta = json.load(open(args.meta))["images"]
    u8 = np.load(args.npy)  # [N,640,640,3] RGB
    assert u8.shape[0] == len(meta), (u8.shape, len(meta))
    ie = InferenceEngine(args.model)

    dets, lat = [], []
    for i, m in enumerate(meta):
        frame = u8[i]
        if args.color == "bgr":
            frame = frame[:, :, ::-1]
        buf = np.ascontiguousarray(frame[None], dtype=np.uint8)  # [1,640,640,3]
        t = time.perf_counter()
        out = ie.run([buf])
        lat.append((time.perf_counter() - t) * 1e3)
        arr = np.asarray(out[0]).reshape(300, 6)
        dd = decode(arr, m["image_id"], m["r"], m["left"], m["top"],
                    m["w0"], m["h0"], conf=args.conf)
        dets += dd
        if i == 0:
            cls = sorted({int(d["category_id"]) for d in dd})
            print(f"[img0] color={args.color} dets={len(dd)} cat_ids={cls[:12]}")
    json.dump(dets, open(args.out, "w"))
    lat = np.array(lat)
    summ = {"model": os.path.basename(args.model), "color": args.color,
            "images": len(meta), "dets": len(dets),
            "lat_p50_ms": round(float(np.percentile(lat, 50)), 3),
            "lat_mean_ms": round(float(lat.mean()), 3),
            "lat_min_ms": round(float(lat.min()), 3)}
    json.dump(summ, open(args.out.replace(".json", "_lat.json"), "w"), indent=2)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
