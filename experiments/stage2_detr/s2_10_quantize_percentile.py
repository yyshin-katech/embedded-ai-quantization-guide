#!/usr/bin/env python3
# s2_10_quantize_percentile.py — calibration이 진짜 레버인가(§2.1.1 outlier 명제 직접 검증).
# 전부 INT8을 MinMax(기본) 대신 Percentile 캘리브로 다시 만들어, outlier를 clip하면
# per-tensor scale이 본체를 되찾아 mAP가 회복되는지 본다. (ORT 내장, 무의존)
# 두 percentile(99.99, 99.9)로 clip 강도를 bracket.
import os, glob, sys, numpy as np
from PIL import Image
from transformers import DetrImageProcessor
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantType, QuantFormat, CalibrationMethod)

# 사용법: s2_10_quantize_percentile.py [N] [pct ...]
# Percentile/Entropy 히스토그램 캘리브는 고해상 DETR activation에서 메모리를 크게 먹어
# N=100은 31GB OOM. N을 줄여 맞춘다(히스토그램 통계라 소표본도 유효).
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
pcts = [float(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [99.99, 99.9]
src = "_workspace/stage2/detr_dyn.onnx"
proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
files = sorted(glob.glob("_workspace/coco/val2017/*.jpg"))[:N]
print(f"calib N={len(files)} | percentiles={pcts}", flush=True)

# ⚠️ 동적 shape 모델 + Percentile/Entropy 히스토그램 캘리브의 충돌:
# 이미지마다 H×W가 달라 activation 텐서 shape가 제각각 → ORT 히스토그램 수집기가
# per-image 배열을 하나로 stack하려다 "inhomogeneous shape" ValueError로 죽고,
# per-image 데이터를 쌓아 메모리도 폭발(N=100 OOM). MinMax(스칼라 min/max)는 무관.
# 해결: 캘리브 입력을 고정 shape로. quant param은 shape 무관 스칼라라 결과 모델은 동적 유지.
FIXED_WH = (1066, 800)  # (W, H) — DETR 기본 최단변 800 근방의 대표 해상도
class Reader(CalibrationDataReader):
    def __init__(self, files): self.files = list(files); self.i = 0
    def get_next(self):
        if self.i >= len(self.files): return None
        f = self.files[self.i]; self.i += 1
        im = Image.open(f).convert("RGB").resize(FIXED_WH)
        pv = proc(images=im, return_tensors="np", do_resize=False)["pixel_values"]
        return {"pixel_values": pv.astype(np.float32)}

for pct in pcts:
    dst = f"_workspace/stage2/detr_dyn_int8_pct{str(pct).replace('.','p')}.onnx"
    print(f"=== Percentile {pct} 캘리브 → 전부 INT8 ===", flush=True)
    quantize_static(src, dst, calibration_data_reader=Reader(files),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"percentile": pct})
    print(f"{dst} =", round(os.path.getsize(dst) / 1e6, 1), "MB", flush=True)
print("PCT_QUANT_DONE")
