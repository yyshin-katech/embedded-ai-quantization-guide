#!/usr/bin/env python3
"""HOST scorer (emb-ai venv) — turn the FP32 (x86) and NPU (Pi) raw DETR outputs into
COCO mAP + numerical divergence. Postprocess = the stage2 s2_07 / stage3 formula EXACTLY
(softmax -> drop no-object idx 91 -> cxcywh*W,H -> xywh), scaled by each image's ORIGINAL
W,H from the COCO annotations (DETR boxes are resolution-normalised, so the fixed 800x1066
resize does not enter the box math). category_id = class index (DETR COCO-91).

FP32 and NPU read the same eval pixels -> same imgIds, same scorer -> the mAP gap isolates
quantization. Also reports NPU-vs-FP32 logits/boxes correlation + max-abs-delta."""
import argparse
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANN = "/home/yuyeong/embedded-ai-quantization-guide/_workspace/coco/annotations/instances_val2017.json"


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def postprocess(logits, boxes, W, H):
    """logits [100,92], boxes [100,4] cxcywh-normalised -> list of (cid, score, [x,y,w,h])."""
    prob = softmax(logits.astype(np.float64), -1)[:, :91]     # drop no-object (index 91)
    labels = prob.argmax(-1)
    scores = prob.max(-1)
    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x0 = (xc - 0.5 * w) * W
    y0 = (yc - 0.5 * h) * H
    ww = w * W
    hh = h * H
    return [(int(labels[i]), float(scores[i]),
             [float(x0[i]), float(y0[i]), float(ww[i]), float(hh[i])])
            for i in range(logits.shape[0])]


def score(coco, npz):
    d = np.load(npz)
    logits, boxes, img_ids = d["logits"], d["boxes"], d["img_ids"]
    dets = []
    for k in range(logits.shape[0]):
        iid = int(img_ids[k])
        info = coco.loadImgs(iid)[0]
        W, H = info["width"], info["height"]
        for cid, sc, box in postprocess(logits[k], boxes[k], W, H):
            dets.append({"image_id": iid, "category_id": cid, "bbox": box, "score": sc})
    dt = coco.loadRes(dets)
    ev = COCOeval(coco, dt, "bbox")
    ev.params.imgIds = [int(i) for i in img_ids]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    m = {"mAP": round(float(s[0]), 4), "mAP50": round(float(s[1]), 4), "mAP75": round(float(s[2]), 4),
         "mAP_s": round(float(s[3]), 4), "mAP_m": round(float(s[4]), 4), "mAP_l": round(float(s[5]), 4),
         "n_images": int(len(img_ids))}
    return m, logits, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True, help="detr_fp32_raw.npz")
    ap.add_argument("--npu", required=True, help="detr_npu_raw.npz")
    ap.add_argument("--out", required=True, help="detr_map_summary.json")
    args = ap.parse_args()

    coco = COCO(ANN)
    m_fp32, lg0, bx0 = score(coco, args.fp32)
    print("[fp32    ] " + json.dumps(m_fp32), flush=True)
    m_npu, lg1, bx1 = score(coco, args.npu)
    print("[npu_int8] " + json.dumps(m_npu), flush=True)

    delta = {k: round(m_fp32[k] - m_npu[k], 4) for k in ("mAP", "mAP50", "mAP75", "mAP_s", "mAP_m", "mAP_l")}
    rel = round(-delta["mAP"] / m_fp32["mAP"] * 100, 1) if m_fp32["mAP"] else None
    div = {"logits_corr": round(float(np.corrcoef(lg0.ravel(), lg1.ravel())[0, 1]), 6),
           "logits_absmax": round(float(np.abs(lg0 - lg1).max()), 4),
           "boxes_corr": round(float(np.corrcoef(bx0.ravel(), bx1.ravel())[0, 1]), 6),
           "boxes_absmax": round(float(np.abs(bx0 - bx1).max()), 4)}
    summary = {"fp32": m_fp32, "npu_int8": m_npu,
               "delta_fp32_minus_npu": delta, "rel_mAP_drop_pct": rel,
               "divergence_npu_vs_fp32": div}
    json.dump(summary, open(args.out, "w"), indent=2)
    print("delta:", json.dumps(delta), "| rel_mAP_drop_pct:", rel, flush=True)
    print("div:", json.dumps(div), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
