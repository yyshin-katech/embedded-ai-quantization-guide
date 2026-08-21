#!/usr/bin/env python3
# detr_sym_export.py (HOST, emb-ai venv) — symmetric re-export of DETR to clear the
# stage3 case-C + case-B parser blockers, mirroring the ResNet50 recipe that DID build
# on trtexec (t02_latency_3point.py):
#     nodes_to_exclude=["/model/backbone/model/conv1/Conv"]        # case D (stem 3ch7x7)
#     extra_options={ActivationSymmetric:True, WeightSymmetric:True, QuantizeBias:False}
#         ActivationSymmetric/WeightSymmetric -> zp=0 everywhere  (clears case C shiftIsAllZeros)
#         QuantizeBias:False                  -> no INT32 bias DQ  (clears case B)
# The stage2 detr_int8.onnx used none of these -> zp!=0 (1085/1485) + 179 bias DQ -> case C.
#
# Calibration: 100 COCO val images (tail 4900:5000, DISJOINT from the head eval subset),
# preprocessed with the SAME detr_prep.preprocess used at eval time (fixed 800x1066).
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from detr_prep import preprocess
from pycocotools.coco import COCO
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

SRC = "_workspace/stage2/detr_sim.onnx"
DST = "_workspace/stage3_jetson_detr_acc/detr_int8_sym.onnx"
ANN = "_workspace/coco/annotations/instances_val2017.json"
IMG = "_workspace/coco/val2017"
N_CALIB = 100

coco = COCO(ANN)
img_ids = sorted(coco.getImgIds())
calib_ids = img_ids[-N_CALIB:]                       # tail, disjoint from head eval
paths = [os.path.join(IMG, coco.loadImgs(i)[0]["file_name"]) for i in calib_ids]
print(f"calib images: {len(paths)} (COCO val tail {N_CALIB}) | input name = pixel_values", flush=True)


class Reader(CalibrationDataReader):
    def __init__(self, paths):
        self.it = iter(paths)
    def get_next(self):
        p = next(self.it, None)
        if p is None:
            return None
        return {"pixel_values": preprocess(p)}


print("=== quantize_static: symmetric QInt8 QDQ (zp=0), QuantizeBias=False, per-channel ===", flush=True)
# op_types_to_quantize=["Conv","Gemm"] — the weight-bearing ops (mirrors ResNet50's
# Conv+final-Gemm). The default (quantize *everything*) put Q/DQ on the transformer's
# attention Constant/Mul/MatMul/Softmax and TRT's builder rejects the quantized-constant
# pattern (qdqGraphOptimizer: "Quantized constant is only allowed before DQ or PLUGIN").
# Leaving attention matmuls + LayerNorm + Softmax in FP16 is exactly stage2 §4.5's finding
# that op-selection mixed barely moves mAP — the lever is activation granularity, not op set.
quantize_static(
    SRC, DST,
    calibration_data_reader=Reader(paths),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    op_types_to_quantize=["Conv", "Gemm"],
    nodes_to_exclude=["/model/backbone/model/conv1/Conv"],
    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False},
)
print("wrote", DST, round(os.path.getsize(DST) / 1e6, 1), "MB", flush=True)

# ---- verify it mirrors the ResNet50 parseable QDQ (zp all zero, 0 INT32 bias DQ) ----
import onnx
from onnx import numpy_helper
m = onnx.load(DST); g = m.graph
inits = {i.name: i for i in g.initializer}
qdq = [n for n in g.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
zp_total = zp_nonzero = int32_dq = 0
for n in qdq:
    if len(n.input) >= 3 and n.input[2] in inits:
        arr = numpy_helper.to_array(inits[n.input[2]]); zp_total += 1
        if np.any(arr != 0):
            zp_nonzero += 1
for n in g.node:
    if n.op_type == "DequantizeLinear" and n.input[0] in inits:
        if numpy_helper.to_array(inits[n.input[0]]).dtype == np.int32:
            int32_dq += 1
print(f"VERIFY  zp_total={zp_total}  zp_nonzero={zp_nonzero}  int32_bias_DQ={int32_dq}", flush=True)
assert zp_nonzero == 0, "zp not all zero -> case C would still fire"
assert int32_dq == 0, "INT32 bias DQ present -> case B would still fire"
print("SYM_EXPORT_OK  (symmetric, no bias DQ — mirrors resnet50_int8_qdq.onnx)", flush=True)
