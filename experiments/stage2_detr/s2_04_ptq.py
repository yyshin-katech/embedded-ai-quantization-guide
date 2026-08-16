#!/usr/bin/env python3
# s2_04_ptq.py — 2단계 §4.5 검증(기계적 경로 + 발산 프록시).
# 문서의 quantize_static(전부 INT8·QDQ·per-channel)을 현행 ORT 1.23.2로 실제 실행하고,
# INT8이 DETR 출력을 얼마나 흔드는지 '검출 일치/발산'으로 본다.
# 주의: 캘리브는 단일 이미지 증강 12장(경량) — mAP가 아니라 '기계적 성립 + 방향성' 확인용.
import numpy as np, torch, requests
from PIL import Image, ImageEnhance
from transformers import DetrForObjectDetection, DetrImageProcessor
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

mid = "facebook/detr-resnet-50"
proc = DetrImageProcessor.from_pretrained(mid)
img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw).convert("RGB")

def make_samples(n):
    out = []
    for i in range(n):
        im = img.transpose(Image.FLIP_LEFT_RIGHT) if i % 2 else img
        im = ImageEnhance.Brightness(im).enhance(0.8 + 0.05 * (i % 5))
        pv = proc(images=im, return_tensors="pt")["pixel_values"].numpy().astype(np.float32)
        # 고정 shape로 통일 (첫 샘플 기준 리사이즈 회피 위해 동일 이미지 계열 사용)
        out.append(pv)
    return out

samples = make_samples(12)
base_pv = samples[0]
print("calib 샘플:", len(samples), "| shape:", samples[0].shape)

class Reader(CalibrationDataReader):
    def __init__(self, s): self.it = iter(s)
    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"pixel_values": b}

print("=== quantize_static: 전부 INT8 (QDQ, per-channel) ===", flush=True)
quantize_static("_workspace/stage2/detr_sim.onnx", "_workspace/stage2/detr_int8.onnx",
                calibration_data_reader=Reader(samples),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True)
import os
print("detr_int8.onnx =", round(os.path.getsize("_workspace/stage2/detr_int8.onnx")/1e6, 1), "MB")

def run(path, pv):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [x.name for x in s.get_outputs()]
    r = dict(zip(names, s.run(None, {"pixel_values": pv})))
    lg = r.get("logits"); bx = r.get("pred_boxes")
    if lg is None:
        for a in r.values():
            if a.shape[-2:] == (100, 92): lg = a
            if a.shape[-2:] == (100, 4): bx = a
    return lg, bx

def dets(lg):
    p = torch.softmax(torch.tensor(lg), -1)[0]
    return int((p[:, :91].max(-1).values > 0.9).sum())

lg32, bx32 = run("_workspace/stage2/detr_sim.onnx", base_pv)
lg8,  bx8  = run("_workspace/stage2/detr_int8.onnx", base_pv)
print("FP32 검출(>0.9):", dets(lg32), "| INT8 검출(>0.9):", dets(lg8))
print("logits max|Δ|(FP32 vs INT8):", float(np.abs(lg32 - lg8).max()))
print("boxes  max|Δ|(FP32 vs INT8):", float(np.abs(bx32 - bx8).max()))
# 상위 검출의 클래스/박스가 유지되는지 (soft one-hot 붕괴 신호)
def top(lg, bx, k=5):
    p = torch.softmax(torch.tensor(lg), -1)[0][:, :91]
    v, idx = p.max(-1)
    order = v.argsort(descending=True)[:k]
    return [(int(idx[i]), round(float(v[i]), 3)) for i in order]
print("FP32 top5:", top(lg32, bx32))
print("INT8 top5:", top(lg8, bx8))
print("PTQ_DONE")
