#!/usr/bin/env python3
# analyze_detr_sq_map.py (HOST, emb-ai venv) — does on-device SmoothQuant recover the DETR
# INT8 collapse that the accuracy-valid symmetric engine reproduced (58ba518: 0.4237->0.2383)?
#
# The A/B is clean: sq_a10/sq_a05 differ from the committed detr_gpu_int8_sym.plan ONLY by
# the ONNX-level SmoothQuant activation migration (identical sym QDQ recipe, same board, same
# trtexec flags, same 1000-image head subset, same detr_prep). To rule out any pipeline drift
# we RE-EVALUATE fp32 + sym from their committed npz with the SAME postprocess here — they must
# reproduce 0.4237 / 0.2383, which is itself the cross-check.
import json, os, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SQ  = "experiments/stage3_tensorrt/jetson_ondevice/detr_smoothquant/results"   # sq npz (this run)
ACC = "experiments/stage3_tensorrt/jetson_ondevice/detr_accuracy/results"      # committed fp32/sym npz
ANN = "_workspace/coco/annotations/instances_val2017.json"
OUT = os.path.join(SQ, "detr_sq_map_summary.json")

# (tag, npz_dir, accuracy_valid)
TAGS = [("gpu_fp32", ACC, True), ("gpu_int8_sym", ACC, True),
        ("sq_a10", SQ, True), ("sq_a05", SQ, True)]

coco = COCO(ANN)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def postprocess(logits, boxes, W, H):
    prob = softmax(logits.astype(np.float64), -1)[:, :91]     # drop no-object (index 91)
    labels = prob.argmax(-1); scores = prob.max(-1)
    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x0 = (xc - 0.5 * w) * W; y0 = (yc - 0.5 * h) * H
    ww = w * W; hh = h * H
    return [(int(labels[i]), float(scores[i]),
             [float(x0[i]), float(y0[i]), float(ww[i]), float(hh[i])]) for i in range(logits.shape[0])]


def eval_tag(tag, d):
    z = np.load(os.path.join(d, "detr_%s_raw.npz" % tag))
    logits, boxes, img_ids = z["logits"], z["boxes"], z["img_ids"]
    dets = []
    for k in range(logits.shape[0]):
        iid = int(img_ids[k]); info = coco.loadImgs(iid)[0]
        W, H = info["width"], info["height"]
        for cid, sc, box in postprocess(logits[k], boxes[k], W, H):
            dets.append({"image_id": iid, "category_id": cid, "bbox": box, "score": sc})
    ev = COCOeval(coco, coco.loadRes(dets), "bbox")
    ev.params.imgIds = [int(i) for i in img_ids]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP": round(float(s[0]), 4), "mAP50": round(float(s[1]), 4), "mAP75": round(float(s[2]), 4),
            "mAP_s": round(float(s[3]), 4), "mAP_m": round(float(s[4]), 4), "mAP_l": round(float(s[5]), 4)}, logits, boxes


def main():
    summary, raw = {}, {}
    for tag, d, acc_valid in TAGS:
        p = os.path.join(d, "detr_%s_raw.npz" % tag)
        if not os.path.exists(p):
            print("MISSING", p); continue
        m, lg, bx = eval_tag(tag, d)
        m["accuracy_valid"] = acc_valid
        summary[tag] = m; raw[tag] = (lg, bx)
        print("[%-14s] mAP=%.4f mAP50=%.4f mAP_s=%.4f mAP_m=%.4f mAP_l=%.4f"
              % (tag, m["mAP"], m["mAP50"], m["mAP_s"], m["mAP_m"], m["mAP_l"]), flush=True)

    fp32, sym = summary["gpu_fp32"]["mAP"], summary["gpu_int8_sym"]["mAP"]
    fp32_s, sym_s = summary["gpu_fp32"]["mAP_s"], summary["gpu_int8_sym"]["mAP_s"]
    gap = fp32 - sym                                              # the collapse SmoothQuant must recover
    rec = {}
    for tag in ("sq_a10", "sq_a05"):
        if tag not in summary:
            continue
        mp, mps = summary[tag]["mAP"], summary[tag]["mAP_s"]
        rec[tag] = {
            "mAP": mp, "mAP_s": mps,
            "delta_vs_sym_abs": round(mp - sym, 4),
            "delta_vs_fp32_abs": round(mp - fp32, 4),
            "delta_vs_fp32_pct": round((mp - fp32) / fp32 * 100, 1),
            "recovery_pct_of_gap": round((mp - sym) / gap * 100, 1) if gap else None,
            "maps_recovery_pct_of_gap": round((mps - sym_s) / (fp32_s - sym_s) * 100, 1) if (fp32_s - sym_s) else None,
        }
        print("  RECOVERY[%s] mAP %.4f  Δvs_sym=%+.4f  recovery=%.1f%% of the %.4f gap  (Δvs_fp32 %+.1f%%)"
              % (tag, mp, rec[tag]["delta_vs_sym_abs"], rec[tag]["recovery_pct_of_gap"], gap,
                 rec[tag]["delta_vs_fp32_pct"]), flush=True)

    # divergence vs FP32 (sanity: SQ engines are real INT8, not a silent FP32 clone)
    lg0, bx0 = raw["gpu_fp32"]; div = {}
    for tag in ("gpu_int8_sym", "sq_a10", "sq_a05"):
        if tag not in raw:
            continue
        lg, bx = raw[tag]
        div[tag] = {"logits_corr": round(float(np.corrcoef(lg0.ravel(), lg.ravel())[0, 1]), 6),
                    "logits_absmax": round(float(np.abs(lg0 - lg).max()), 4),
                    "boxes_corr": round(float(np.corrcoef(bx0.ravel(), bx.ravel())[0, 1]), 6),
                    "boxes_absmax": round(float(np.abs(bx0 - bx).max()), 4)}
    # SQ-vs-sym prediction change (confirms SmoothQuant actually changed the engine)
    lgs, bxs = raw["gpu_int8_sym"]
    sq_vs_sym = {}
    for tag in ("sq_a10", "sq_a05"):
        if tag in raw:
            lg, bx = raw[tag]
            sq_vs_sym[tag] = {"logits_absmax": round(float(np.abs(lgs - lg).max()), 4),
                              "boxes_absmax": round(float(np.abs(bxs - bx).max()), 4)}

    summary["_recovery"] = {"fp32_mAP": fp32, "sym_mAP": sym, "gap": round(gap, 4),
                            "fp32_mAP_s": fp32_s, "sym_mAP_s": sym_s, **rec}
    summary["_divergence_vs_fp32"] = div
    summary["_sq_vs_sym_absmax"] = sq_vs_sym
    json.dump(summary, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
