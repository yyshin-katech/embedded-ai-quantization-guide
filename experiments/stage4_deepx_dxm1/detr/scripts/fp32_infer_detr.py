#!/usr/bin/env python3
"""FP32 DETR reference (x86 onnxruntime) on the SAME eval .npy the NPU reads. Runs the
identical graph the .dxnn was compiled from (detr_2out.onnx) so the only difference vs the
NPU is quantization. Applies Div/255 + ImageNet norm EXPLICITLY (folded on the NPU side).
Dumps raw logits/boxes to detr_fp32_raw.npz for the shared host scorer (analyze_detr_map.py)."""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort

from detr_det_common import to_fp32_nchw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/yuyeong/dxm1_detr/detr_2out.onnx")
    ap.add_argument("--npy", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True, help="detr_fp32_raw.npz")
    args = ap.parse_args()

    meta = json.load(open(args.meta))["images"]
    u8 = np.load(args.npy)  # [N,800,1066,3] RGB
    assert u8.shape[0] == len(meta), (u8.shape, len(meta))

    so = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    in_name = so.get_inputs()[0].name
    out_names = [o.name for o in so.get_outputs()]
    li, bi = out_names.index("logits"), out_names.index("pred_boxes")
    print("providers:", so.get_providers(), "| in:", in_name, "| out:", out_names)

    logits, boxes, ids = [], [], []
    for i, m in enumerate(meta):
        x = to_fp32_nchw(u8[i])                       # (1,3,800,1066) float32
        outs = so.run(None, {in_name: x})
        logits.append(np.asarray(outs[li]).reshape(100, 92))
        boxes.append(np.asarray(outs[bi]).reshape(100, 4))
        ids.append(m["image_id"])
    np.savez(args.out,
             logits=np.stack(logits).astype(np.float32),
             boxes=np.stack(boxes).astype(np.float32),
             img_ids=np.asarray(ids, dtype=np.int64))
    print(f"wrote {args.out}  logits{np.stack(logits).shape} boxes{np.stack(boxes).shape} n={len(ids)}")


if __name__ == "__main__":
    main()
