#!/usr/bin/env python3
# s2_08_quantize_mixed.py — mixed precision(§4.5 회복 팔).
# DETR의 '가장 어려운' op = attention score matmul(activation×activation):
#   /model/*/self_attn/MatMul, /MatMul_1  (Q·Kᵀ, attn·V)
#   /model/*/encoder_attn/MatMul, /MatMul_1  (cross-attn)
# 이들만 FP로 남기고(nodes_to_exclude) 나머지(backbone Conv, *_proj, FFN)는 INT8.
# 주의: *_proj/MatMul(activation×weight)은 제외하지 않는다(쉬운 축).
import onnx, re, os, glob, sys, numpy as np
from PIL import Image
from transformers import DetrImageProcessor
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
src = "_workspace/stage2/detr_dyn.onnx"
dst = "_workspace/stage2/detr_dyn_mixed.onnx"

m = onnx.load(src, load_external_data=False)
excl = [n.name for n in m.graph.node
        if n.op_type == "MatMul" and re.search(r"/(self_attn|encoder_attn)/MatMul(_1)?$", n.name)]
print(f"제외(FP 유지) attention score MatMul: {len(excl)}개", flush=True)
for e in excl[:4]:
    print("  ", e)

proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
files = sorted(glob.glob("_workspace/coco/val2017/*.jpg"))[:N]
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
        return {"pixel_values": pv}

print("=== quantize_static: mixed (attention score matmul FP 유지) ===", flush=True)
quantize_static(src, dst,
                calibration_data_reader=Reader(files),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True, nodes_to_exclude=excl)
print("detr_dyn_mixed.onnx =", round(os.path.getsize(dst) / 1e6, 1), "MB")
print("MIXED_QUANT_DONE")
