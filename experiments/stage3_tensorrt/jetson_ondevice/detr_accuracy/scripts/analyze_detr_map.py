#!/usr/bin/env python3
# analyze_detr_map.py (HOST, emb-ai venv) — turn the board's raw DETR outputs into COCO
# mAP + numerical divergence vs FP32. Postprocess = the stage2 s2_07 formula exactly
# (softmax -> drop no-object -> cxcywh*W,H -> xywh), scaled by each image's ORIGINAL W,H
# from the COCO annotations (DETR boxes are resolution-normalised, so the fixed-800x1066
# input resize does not enter the box math). category_id = class index (DETR COCO-91).
import json, os, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

RAW = "experiments/stage3_tensorrt/jetson_ondevice/detr_accuracy/results"   # npz pulled from board
ANN = "_workspace/coco/annotations/instances_val2017.json"
OUT = os.path.join(RAW, "detr_map_summary.json")

TAGS = [("gpu_fp32", True), ("gpu_fp16", True),
        ("gpu_int8_sym", True), ("gpu_int8_implicit", False)]

coco = COCO(ANN)


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
    out = []
    for i in range(logits.shape[0]):
        out.append((int(labels[i]), float(scores[i]),
                    [float(x0[i]), float(y0[i]), float(ww[i]), float(hh[i])]))
    return out


def eval_tag(tag):
    d = np.load(os.path.join(RAW, "detr_%s_raw.npz" % tag))
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
    ev.params.imgIds = [int(i) for i in img_ids]              # restrict to the evaluated subset
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP": round(float(s[0]), 4), "mAP50": round(float(s[1]), 4),
            "mAP75": round(float(s[2]), 4), "mAP_s": round(float(s[3]), 4),
            "mAP_m": round(float(s[4]), 4), "mAP_l": round(float(s[5]), 4)}, logits, boxes


def main():
    summary, raw = {}, {}
    for tag, acc_valid in TAGS:
        p = os.path.join(RAW, "detr_%s_raw.npz" % tag)
        if not os.path.exists(p):
            print("MISSING", p); continue
        m, lg, bx = eval_tag(tag)
        m["accuracy_valid"] = acc_valid
        summary[tag] = m
        raw[tag] = (lg, bx)
        print("[%-18s] mAP=%.4f mAP50=%.4f mAP_s=%.4f mAP_m=%.4f mAP_l=%.4f %s"
              % (tag, m["mAP"], m["mAP50"], m["mAP_s"], m["mAP_m"], m["mAP_l"],
                 "" if acc_valid else "[implicit auto-range: accuracy NOT claimed]"), flush=True)

    # numerical divergence vs FP32 (logits/boxes correlation + max abs delta), capstone-style
    if "gpu_fp32" in raw:
        lg0, bx0 = raw["gpu_fp32"]
        div = {}
        for tag in raw:
            if tag == "gpu_fp32":
                continue
            lg, bx = raw[tag]
            div[tag] = {
                "logits_corr": round(float(np.corrcoef(lg0.ravel(), lg.ravel())[0, 1]), 6),
                "logits_absmax": round(float(np.abs(lg0 - lg).max()), 4),
                "boxes_corr": round(float(np.corrcoef(bx0.ravel(), bx.ravel())[0, 1]), 6),
                "boxes_absmax": round(float(np.abs(bx0 - bx).max()), 4),
            }
        summary["_divergence_vs_fp32"] = div
        for tag, v in div.items():
            print("  div[%s] logits_corr=%.5f absmax=%.3f | boxes_corr=%.5f absmax=%.3f"
                  % (tag, v["logits_corr"], v["logits_absmax"], v["boxes_corr"], v["boxes_absmax"]), flush=True)

    json.dump(summary, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
