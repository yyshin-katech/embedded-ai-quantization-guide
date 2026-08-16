#!/usr/bin/env python3
# s2_01_baseline.py — 2단계 §4.1 검증: DETR 로드 + FP32 baseline 추론.
# 문서의 detr_load.py를 그대로 재현해 기대 shape([1,100,92]/[1,100,4])를 확인한다.
import sys, torch, requests
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

model_id = "facebook/detr-resnet-50"
print("transformers 로 DETR 로드 중...", flush=True)
processor = DetrImageProcessor.from_pretrained(model_id)
model = DetrForObjectDetection.from_pretrained(model_id).eval()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 고양이 2마리
img = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=img, return_tensors="pt")
print("pixel_values shape :", tuple(inputs["pixel_values"].shape))

with torch.no_grad():
    out = model(**inputs)
print("logits shape       :", tuple(out.logits.shape), "(기대 [1,100,92])")
print("pred_boxes shape   :", tuple(out.pred_boxes.shape), "(기대 [1,100,4])")

# DETR 설정 실측: 문서 2.1이 GELU를 4대 문제로 들지만 DETR FFN 활성은?
cfg = model.config
print("DETR activation_fn :", getattr(cfg, "activation_function", "?"), "(문서 GELU 가정 검증용)")
print("num_queries        :", getattr(cfg, "num_queries", "?"))
print("BASELINE_OK")
