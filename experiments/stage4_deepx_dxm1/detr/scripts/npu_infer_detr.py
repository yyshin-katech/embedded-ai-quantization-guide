#!/usr/bin/env python3
"""DX-M1 INT8 DETR inference (run ON the Pi). Reads the SAME force-resized eval .npy the
FP32 reference read, feeds each frame to the .dxnn via dx_engine, and dumps raw
logits/boxes to detr_npu_raw.npz for the shared host scorer (analyze_detr_map.py) — so
quantization is the only variable. dx_engine input contract = uint8 NHWC [1,800,1066,3]
RGB (Div/normalize/transpose folded; convertColor/expandDim not folded -> caller supplies
final RGB, same as the detection axis).

DETR has TWO outputs (logits [1,100,92], pred_boxes [1,100,4]); dx_engine returns a list
whose order is not guaranteed, so we disambiguate by element count (9200 vs 400)."""
import argparse
import json
import os
import time

import numpy as np

from dx_engine import InferenceEngine

N_LOGITS = 100 * 92   # 9200
N_BOXES = 100 * 4     # 400


def split_outputs(out):
    a0 = np.asarray(out[0]).ravel()
    a1 = np.asarray(out[1]).ravel()
    if a0.size == N_LOGITS and a1.size == N_BOXES:
        return a0.reshape(100, 92), a1.reshape(100, 4)
    if a1.size == N_LOGITS and a0.size == N_BOXES:
        return a1.reshape(100, 92), a0.reshape(100, 4)
    raise ValueError(f"unexpected output sizes: {a0.size}, {a1.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True, help="detr_npu_raw.npz")
    ap.add_argument("--color", choices=["rgb", "bgr"], default="rgb")
    args = ap.parse_args()

    meta = json.load(open(args.meta))["images"]
    u8 = np.load(args.npy)  # [N,800,1066,3] RGB
    assert u8.shape[0] == len(meta), (u8.shape, len(meta))
    ie = InferenceEngine(args.model)

    logits, boxes, ids, lat = [], [], [], []
    for i, m in enumerate(meta):
        frame = u8[i]
        if args.color == "bgr":
            frame = frame[:, :, ::-1]
        buf = np.ascontiguousarray(frame[None], dtype=np.uint8)  # [1,800,1066,3]
        t = time.perf_counter()
        out = ie.run([buf])
        lat.append((time.perf_counter() - t) * 1e3)
        lg, bx = split_outputs(out)
        logits.append(lg.astype(np.float32))
        boxes.append(bx.astype(np.float32))
        ids.append(m["image_id"])
        if i == 0:
            print(f"[img0] color={args.color} n_out={len(out)} "
                  f"logits{lg.shape} boxes{bx.shape} logit_absmax={np.abs(lg).max():.3f}")

    np.savez(args.out,
             logits=np.stack(logits), boxes=np.stack(boxes),
             img_ids=np.asarray(ids, dtype=np.int64))
    lat = np.array(lat)
    summ = {"model": os.path.basename(args.model), "color": args.color, "images": len(meta),
            "lat_p50_ms": round(float(np.percentile(lat, 50)), 3),
            "lat_mean_ms": round(float(lat.mean()), 3),
            "lat_min_ms": round(float(lat.min()), 3)}
    json.dump(summ, open(args.out.replace(".npz", "_lat.json"), "w"), indent=2)
    print("wrote", args.out, "|", json.dumps(summ))


if __name__ == "__main__":
    main()
