#!/usr/bin/env python3
# s2_09_quantize_ablation.py — 결정적 절제: 폭락의 범인이 backbone인가 transformer인가.
# mixed(attention score matmul FP)가 +0.36 mAP만 회복 → "attention이 범인" 반증.
# 진짜 범인 위치를 두 구성으로 분리:
#   bb_fp : backbone(Conv 53) FP 유지, transformer(137)만 INT8  → 회복되면 '범인=backbone'
#   tf_fp : transformer(137) FP 유지, backbone(Conv 53)만 INT8  → 회복되면 '범인=transformer'
# 캘리브는 int8/mixed와 동일한 COCO val 앞 100장(일관성).
import onnx, os, glob, numpy as np
from PIL import Image
from transformers import DetrImageProcessor
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

src = "_workspace/stage2/detr_dyn.onnx"
m = onnx.load(src, load_external_data=False)
q = [n for n in m.graph.node if n.op_type in ("Conv", "MatMul", "Gemm")]
bb = [n.name for n in q if n.name.startswith("/model/backbone")]
rest = [n.name for n in q if not n.name.startswith("/model/backbone")]
print(f"backbone 노드 {len(bb)} | 비-backbone 노드 {len(rest)}", flush=True)

proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
files = sorted(glob.glob("_workspace/coco/val2017/*.jpg"))[:100]

class Reader(CalibrationDataReader):
    def __init__(self, files): self.files = list(files); self.i = 0
    def get_next(self):
        if self.i >= len(self.files): return None
        f = self.files[self.i]; self.i += 1
        im = Image.open(f).convert("RGB")
        return {"pixel_values": proc(images=im, return_tensors="np")["pixel_values"].astype(np.float32)}

def quant(dst, exclude, tag):
    print(f"=== {tag}: exclude {len(exclude)} nodes → quantize {len(q)-len(exclude)} ===", flush=True)
    quantize_static(src, dst, calibration_data_reader=Reader(files),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True, nodes_to_exclude=exclude)
    print(f"{tag} =", round(os.path.getsize(dst) / 1e6, 1), "MB", flush=True)

quant("_workspace/stage2/detr_bb_fp.onnx", bb, "bb_fp(backbone FP, transformer INT8)")
quant("_workspace/stage2/detr_tf_fp.onnx", rest, "tf_fp(transformer FP, backbone INT8)")
print("ABLATION_QUANT_DONE")
