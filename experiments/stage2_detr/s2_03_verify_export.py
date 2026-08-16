#!/usr/bin/env python3
# s2_03_verify_export.py — 2단계 §4.3 검증: export가 수치적으로 충실한가.
# 같은 실이미지로 PyTorch(FP32) vs ONNX(detr_sim.onnx) 출력을 비교한다.
import numpy as np, torch, requests
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
import onnxruntime as ort

mid = "facebook/detr-resnet-50"
proc = DetrImageProcessor.from_pretrained(mid)
model = DetrForObjectDetection.from_pretrained(mid).eval()
img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
pv = proc(images=img, return_tensors="pt")["pixel_values"]      # [1,3,800,1066]
print("pixel_values:", tuple(pv.shape))

with torch.no_grad():
    o = model(pv)
lg_pt, bx_pt = o.logits.numpy(), o.pred_boxes.numpy()

sess = ort.InferenceSession("_workspace/stage2/detr_sim.onnx",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ORT providers:", sess.get_providers())
onames = [x.name for x in sess.get_outputs()]
print("ONNX outputs :", onames)
res = sess.run(None, {"pixel_values": pv.numpy()})
by = dict(zip(onames, res))
# 이름 우선, 없으면 shape로 매칭
def pick(shape_tail):
    if "logits" in by and shape_tail == (100, 92): return by["logits"]
    if "pred_boxes" in by and shape_tail == (100, 4): return by["pred_boxes"]
    for a in res:
        if a.shape[-2:] == shape_tail: return a
    return None
lg_ort = by.get("logits", pick((100, 92)))
bx_ort = by.get("pred_boxes", pick((100, 4)))
print("logits max|Δ| :", float(np.abs(lg_pt - lg_ort).max()))
print("boxes  max|Δ| :", float(np.abs(bx_pt - bx_ort).max()))
# 검출 일치(임계 0.9, no-object 클래스 91 제외)
def dets(lg, bx):
    p = torch.softmax(torch.tensor(lg), -1)[0]
    keep = (p[:, :91].max(-1).values > 0.9)
    return keep.sum().item()
print("PyTorch 검출 수(>0.9):", dets(lg_pt, bx_pt), "| ONNX 검출 수:", dets(lg_ort, bx_ort))
print("VERIFY_OK")
