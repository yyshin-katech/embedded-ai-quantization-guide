#!/usr/bin/env python3
# s2_06_quantize_dynamic.py — COCO 실측용 INT8 양자화.
# ⚠️ 파일명의 "dynamic"은 ORT의 quantize_dynamic이 아니라 대상 모델이
#    '동적 shape'(detr_dyn.onnx, H/W/batch 동적)라는 뜻이다. 양자화 방식은
#    캘리브 기반 정적 PTQ(quantize_static)가 맞다 — 동적 shape 모델에 static PTQ.
# detr_dyn.onnx(FP32 동적)을 실제 COCO val 이미지 N장으로 캘리브 →
# 전부 INT8(QDQ, per-channel) detr_dyn_int8.onnx 산출.
import os, glob, sys, numpy as np
from PIL import Image
from transformers import DetrImageProcessor
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
COCO_IMG = "_workspace/coco/val2017"
proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
files = sorted(glob.glob(COCO_IMG + "/*.jpg"))[:N]
print("calib images:", len(files), flush=True)

class Reader(CalibrationDataReader):
    def __init__(self, files):
        self.files = list(files); self.i = 0
    def get_next(self):
        if self.i >= len(self.files):
            return None
        f = self.files[self.i]; self.i += 1
        im = Image.open(f).convert("RGB")
        pv = proc(images=im, return_tensors="np")["pixel_values"].astype(np.float32)
        if self.i % 25 == 0:
            print(f"  calib {self.i}/{len(self.files)}", flush=True)
        return {"pixel_values": pv}

src = "_workspace/stage2/detr_dyn.onnx"
dst = "_workspace/stage2/detr_dyn_int8.onnx"
print("=== quantize_static: 전부 INT8 (QDQ, per-channel) — 동적 shape ===", flush=True)
quantize_static(src, dst,
                calibration_data_reader=Reader(files),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True)
print("detr_dyn_int8.onnx =", round(os.path.getsize(dst) / 1e6, 1), "MB")
print("QUANT_DONE")
