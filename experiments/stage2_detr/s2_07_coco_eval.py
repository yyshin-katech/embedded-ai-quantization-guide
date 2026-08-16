#!/usr/bin/env python3
# s2_07_coco_eval.py — DETR ONNX의 COCO val2017 전량 mAP 실측.
# 한 번의 전처리 패스로 여러 구성(FP32/INT8/mixed)을 동시에 평가한다.
# 후처리는 HF API 대신 표준 DETR 수식(softmax→no-object 드롭→cxcywh→xyxy)으로
# 직접 구현(버전 독립). category_id = 클래스 인덱스(DETR=COCO 91-scheme, index==coco id).
#   사용법: s2_07_coco_eval.py [name=path ...] [--limit N]
#   기본: fp32=detr_dyn.onnx int8=detr_dyn_int8.onnx
import json, os, sys, time, numpy as np, torch
from PIL import Image
from transformers import DetrImageProcessor
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANN = "_workspace/coco/annotations/instances_val2017.json"
IMG = "_workspace/coco/val2017"

specs, limit = {}, None
for a in sys.argv[1:]:
    if a.startswith("--limit"):
        limit = int(a.split("=")[1]) if "=" in a else None
    elif "=" in a:
        k, v = a.split("=", 1); specs[k] = v
if not specs:
    specs = {"fp32": "_workspace/stage2/detr_dyn.onnx",
             "int8": "_workspace/stage2/detr_dyn_int8.onnx"}

proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
coco = COCO(ANN)
img_ids = sorted(coco.getImgIds())
if limit:
    img_ids = img_ids[:limit]
print("평가 이미지:", len(img_ids), "| 구성:", list(specs), flush=True)

sessions = {k: ort.InferenceSession(v, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            for k, v in specs.items()}
for k, s in sessions.items():
    print(f"  [{k}] providers={s.get_providers()[:1]} size={round(os.path.getsize(specs[k])/1e6,1)}MB", flush=True)
in_name = {k: s.get_inputs()[0].name for k, s in sessions.items()}
out_names = {k: [o.name for o in s.get_outputs()] for k, s in sessions.items()}

def postprocess(lg, bx, W, H):
    lg = torch.from_numpy(np.asarray(lg)); bx = torch.from_numpy(np.asarray(bx))
    prob = lg.softmax(-1)[0][:, :91]          # no-object(마지막 92번째) 드롭
    scores, labels = prob.max(-1)
    xc, yc, w, h = bx[0].unbind(-1)
    x0 = (xc - 0.5 * w) * W; y0 = (yc - 0.5 * h) * H
    ww = w * W; hh = h * H
    out = []
    for i in range(scores.shape[0]):
        out.append((int(labels[i]), float(scores[i]),
                    [float(x0[i]), float(y0[i]), float(ww[i]), float(hh[i])]))
    return out

results = {k: [] for k in specs}
t0 = time.time()
for n, iid in enumerate(img_ids):
    info = coco.loadImgs(iid)[0]
    im = Image.open(os.path.join(IMG, info["file_name"])).convert("RGB")
    W, H = im.size
    pv = proc(images=im, return_tensors="np")["pixel_values"].astype(np.float32)
    for k, s in sessions.items():
        outs = dict(zip(out_names[k], s.run(None, {in_name[k]: pv})))
        lg = outs.get("logits"); bx = outs.get("pred_boxes")
        if lg is None or bx is None:
            for a in outs.values():
                if a.shape[-2:] == (100, 92): lg = a
                if a.shape[-2:] == (100, 4): bx = a
        for cid, sc, box in postprocess(lg, bx, W, H):
            results[k].append({"image_id": iid, "category_id": cid,
                               "bbox": box, "score": sc})
    if (n + 1) % 500 == 0:
        print(f"  {n+1}/{len(img_ids)}  ({(time.time()-t0)/(n+1)*1000:.0f} ms/img)", flush=True)

summary = {}
for k in specs:
    json.dump(results[k], open(f"_workspace/stage2/dets_{k}.json", "w"))
    dt = coco.loadRes(results[k])
    ev = COCOeval(coco, dt, "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    summary[k] = {"mAP": round(float(ev.stats[0]), 4), "mAP50": round(float(ev.stats[1]), 4),
                  "mAP75": round(float(ev.stats[2]), 4), "mAP_s": round(float(ev.stats[3]), 4),
                  "mAP_m": round(float(ev.stats[4]), 4), "mAP_l": round(float(ev.stats[5]), 4)}
    print(f"=== {k}: mAP={summary[k]['mAP']:.4f}  mAP50={summary[k]['mAP50']:.4f} ===", flush=True)

json.dump(summary, open("_workspace/stage2/coco_map_summary.json", "w"), indent=2)
print("SUMMARY:", json.dumps(summary))
print("COCO_EVAL_DONE")
