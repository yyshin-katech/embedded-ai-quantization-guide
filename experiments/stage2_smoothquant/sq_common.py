#!/usr/bin/env python3
# sq_common.py — §4.4 SmoothQuant 검증 공통 모듈 (torch fake-quant 경로).
# DETR을 torch에서 직접 돌려 modelopt fake-quant(INT8_DEFAULT vs INT8_SMOOTHQUANT)의
# COCO mAP를 자기일관 3원(FP32/per-tensor INT8/SmoothQuant INT8)으로 측정한다.
# 평가 프로토콜(전처리·후처리)은 s2_07_coco_eval.py와 동일하게 맞춰 FP32가 0.4207을 재현하게 한다.
#   전처리: HF DetrImageProcessor (shortest side 800)
#   후처리: softmax(92) → no-object(마지막) 드롭 → [:, :91] max → cxcywh→xyxy × (W,H)
#   category_id = 클래스 인덱스(DETR = COCO 91-scheme, index==coco id)
import os, sys, time, json, numpy as np, torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MID = "facebook/detr-resnet-50"
ANN = "_workspace/coco/annotations/instances_val2017.json"
IMG = "_workspace/coco/val2017"
DEV = "cuda"

_proc = None
_coco = None

def proc():
    global _proc
    if _proc is None:
        _proc = DetrImageProcessor.from_pretrained(MID)
    return _proc

def coco():
    global _coco
    if _coco is None:
        _coco = COCO(ANN)
    return _coco

def load_model():
    m = DetrForObjectDetection.from_pretrained(MID).to(DEV).eval()
    return m

def img_ids(limit=None):
    ids = sorted(coco().getImgIds())
    return ids[:limit] if limit else ids

def pixel_values(iid):
    info = coco().loadImgs(iid)[0]
    im = Image.open(os.path.join(IMG, info["file_name"])).convert("RGB")
    W, H = im.size
    pv = proc()(images=im, return_tensors="pt")["pixel_values"].to(DEV)
    return pv, W, H, info["file_name"]

def postprocess(logits, boxes, W, H):
    # logits [1,100,92], boxes [1,100,4] (torch, any device)
    lg = logits.detach().float().cpu()
    bx = boxes.detach().float().cpu()
    prob = lg.softmax(-1)[0][:, :91]          # no-object(92번째) 드롭
    scores, labels = prob.max(-1)
    xc, yc, w, h = bx[0].unbind(-1)
    x0 = (xc - 0.5 * w) * W; y0 = (yc - 0.5 * h) * H
    ww = w * W; hh = h * H
    out = []
    for i in range(scores.shape[0]):
        out.append((int(labels[i]), float(scores[i]),
                    [float(x0[i]), float(y0[i]), float(ww[i]), float(hh[i])]))
    return out

@torch.no_grad()
def evaluate(model, ids, tag="", log_every=1000):
    """model을 ids 위에서 돌려 COCO mAP 요약 dict 반환."""
    results = []
    t0 = time.time()
    for n, iid in enumerate(ids):
        pv, W, H, _ = pixel_values(iid)
        o = model(pixel_values=pv)
        for cid, sc, box in postprocess(o.logits, o.pred_boxes, W, H):
            results.append({"image_id": iid, "category_id": cid, "bbox": box, "score": sc})
        if (n + 1) % log_every == 0:
            print(f"  [{tag}] {n+1}/{len(ids)}  ({(time.time()-t0)/(n+1)*1000:.0f} ms/img)", flush=True)
    dt = coco().loadRes(results)
    ev = COCOeval(coco(), dt, "bbox")
    ev.params.imgIds = list(ids)      # 평가한 이미지에만 한정(부분 실행 시 희석 방지; 전체면 기본과 동일)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP": round(float(s[0]), 4), "mAP50": round(float(s[1]), 4),
            "mAP75": round(float(s[2]), 4), "mAP_s": round(float(s[3]), 4),
            "mAP_m": round(float(s[4]), 4), "mAP_l": round(float(s[5]), 4),
            "n_images": len(ids), "sec": round(time.time() - t0, 1)}

def calib_pixel_values(n=100):
    """대표 COCO val n장(균등 간격)을 캘리브용 pixel_values 리스트로. 양 팔 동일 입력."""
    ids = img_ids()
    step = max(1, len(ids) // n)
    picked = ids[::step][:n]
    pvs = [pixel_values(iid)[0] for iid in picked]
    return pvs, picked

def forward_loop_factory(calib_pvs):
    @torch.no_grad()
    def forward_loop(m):
        for pv in calib_pvs:
            m(pixel_values=pv)
    return forward_loop
